"""RoboCar drive link: the ESP8266 two-wheel chassis the Pi now rides on.

Ported from `robot/robocar_server.py` into the panel server so that driving,
localization, the camera and the IMU can all be exercised from one page. The
standalone script's own HTTP server is deliberately NOT carried over — it
listened on 8080, which is the panel's port; the panel's own routes
(`/api/robot/*` in preview.py) replace it.

Wire protocol, unchanged from the sketch (`robot/robocar_esp.ino`):

* Discovery — the ESP broadcasts `ROBOCAR?<SECRET>` on UDP 5001; we answer
  `ROBOCAR!<SECRET>:5000` and also beacon that every 2 s, so the ESP finds
  us whichever comes up first. The token is a "this is my server" label,
  NOT security: the link is plaintext and anyone on the LAN can drive the
  robot.
* Commands — newline-terminated ASCII over TCP 5000. `M <left> <right>`
  with each in [-255, 255], `STOP`, `BRAKE`, `MAX <pwm>`, `PING`, `STATE`,
  and the rest of the sketch's vocabulary passed through verbatim.
* The sketch stops the motors if it hears nothing for `FAILSAFE_MS` (400 ms
  by default), so a held key has to be retransmitted; `_repeater` does that
  every `REPEAT_INTERVAL`.

**Added here: a server-side deadline on top of that repeat.** In the
original script `repeat_cmd` persists until something explicitly clears it,
so the failsafe only protects against the *network* dropping — if the
browser tab crashes or the operator closes the laptop mid-throttle, the
server keeps faithfully retransmitting "full speed ahead" and the ESP keeps
obeying. A driving robot with nobody holding the key is exactly the failure
the sketch's failsafe exists to prevent, so the same idea is applied one
hop up: a drive command is only valid for `COMMAND_TTL_S`, after which this
sends STOP by itself.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SECRET = "ROBOCAR-2026"
DISCOVERY_PORT = 5001
COMMAND_PORT = 5000

BEACON_INTERVAL_S = 2.0
REPEAT_INTERVAL_S = 0.15   # must be well inside the sketch's 400 ms failsafe
COMMAND_TTL_S = 0.6        # how long a drive command stays valid unrefreshed
PWM_LIMIT = 255

# The sketch boots with MAX_PWM = 80, which was tuned on a bare chassis.
# Loaded with the Pi 5, the camera and the battery, two small motors need
# more than that to move at all, so the link raises it on connect.
DEFAULT_MAX_PWM = 150

# Localization spin: rotate in place so the place-index sees the whole room.
# Turning in place has to overcome far more friction than driving straight
# (both wheels fight the carpet sideways), and the sketch maps a request of
# v onto MIN_PWM..MAX_PWM — so 110 against MAX_PWM 150 came out at ~86 and
# did not break static friction at all. Measured on the real chassis: it
# simply sat there. 160 lands near 108, which moves it.
SPIN_SPEED = 160
SPIN_TARGET_DEG = 360.0
SPIN_TIMEOUT_S = 45.0      # backstop if the heading never accumulates
SPIN_BLIND_S = 12.0        # how long to spin when there is no IMU heading at all

# The spin is PULSED, not continuous. Two independent reasons, both real:
#   * The place index is matched on a CLIP embedding of a single frame. A
#     frame grabbed mid-rotation is motion-blurred, and a blurred frame
#     embeds to something that matches nothing in an index built from a
#     slow, sharp walk. Stopping to be photographed is the whole point.
#   * The nav loop only ticks at ~1.5 Hz, so during a continuous spin
#     consecutive yaw samples are tens of degrees apart and each one is
#     taken while the chassis is still turning. Sampling at rest instead
#     makes the accumulated angle far cleaner.
# The settle has to outlast one nav tick or a step can go unphotographed.
SPIN_PULSE_S = 0.40
SPIN_SETTLE_S = 1.20
SPIN_BLIND_STEPS = 14      # ~one revolution's worth when nothing measures it

# Camera pan. The tripod carries a servo, and the user's stated use is two or
# three fixed look positions — not a sweep, not filming around.
#
# **Why panning matters more than it sounds.** Turning the chassis in place is
# the single most expensive thing this robot does: both wheels scrub sideways
# against the floor, which is why it needs motor PWM ~108 to move at all while
# driving straight needs far less. Looking left and right with a servo costs
# milliamps. Three views from a parked robot triple the chance of matching the
# place index without the chassis rotating once.
#
# **Why panning only happens while stopped.** On the ESP8266 the Servo library
# and `analogWrite` both want timer1, and the motors are driven by
# `analogWrite`. Rather than fight that, the sketch attaches the servo, moves
# it, waits, and detaches — and refuses while the motors are running. That
# suits the use exactly: park, look around, drive on. The server enforces the
# same order so a refusal is never a surprise.
PAN_ANGLES_DEG = (-40.0, 0.0, 40.0)
PAN_SETTLE_S = 0.9         # servo travel plus a still frame for the camera
PAN_CENTRE_DEG = 0.0

# Commands the panel is allowed to pass through. Everything else is refused
# rather than forwarded: `FORGET` wipes the ESP's stored Wi-Fi credentials
# and `REBOOT` drops the link, neither of which should be one stray fetch
# away in a browser UI.
ALLOWED_COMMANDS = frozenset({
    "STOP", "BRAKE", "TEST", "PING", "STATE", "ID", "IP",
    "RSSI", "SSID", "UPTIME", "HEAP", "VER", "HELP",
    "LED 1", "LED 0", "LED AUTO",
})


# Turning scrubs both wheels sideways against the floor while driving straight
# does not, so a turn needs materially more effort than the same speed in a
# line. Measured on this chassis (see the localization spin in CLAUDE.md): a
# request of 110 would not rotate it at all while manual driving was fine, and
# 160 was needed — about 1.45x. This is the fraction of that extra effort
# applied in proportion to how hard you are turning.
TURN_BOOST = 0.45

# Below this the motor buzzes and heats without turning the wheel. A stalled
# DC motor draws its STALL current — the largest it ever draws, all of it heat
# — so a wheel commanded into this band is worse than a wheel commanded to
# zero: it costs the most battery and delivers no motion. Mirrors the ESP's
# own MIN_PWM; drive_profile.py carries the same constant for the VLA path.
WHEEL_DEADBAND = 40

# Множители на левое и правое колесо, выравнивающие РАЗНЫЕ моторы.
#
# Замер одометрии показал это косвенно и убедительно: команда была строго
# симметричной (разница колёс 0 во всех шести проездах), а робота всё равно
# уводило — и настолько, что проезды длиннее 4 с занижали измеренную скорость
# на 24%, потому что рулетка меряет прямую до финиша, а робот ехал по дуге.
#
# Одинаковый PWM на два дешёвых мотора N20 не даёт одинаковых оборотов:
# отличаются щётки, редуктор, приработка. Лечится только программно —
# постоянным множителем на более быстрое колесо.
#
# 1.0/1.0 = выключено. Не подбирать на глаз: измерять
# `scripts/measure_wheel_trim.py`, иначе робот будет уводить в другую сторону.
WHEEL_TRIM_LEFT = 1.0
WHEEL_TRIM_RIGHT = 1.0


def _snap_out_of_deadband(value: int, deadband: int = WHEEL_DEADBAND) -> int:
    """Round a wheel command out of the stall band, never leave it inside.

    Up rather than down, and this is deliberate: a command just under the
    threshold spends the most current of any command and moves nothing, so the
    cheap-looking choice (round down to zero) throws away a wheel that was
    asked to contribute. Rounding up honours the intent.
    """
    if value == 0:
        return 0
    if abs(value) >= deadband:
        return value
    return deadband if value > 0 else -deadband


def mix_drive(throttle: float, steer: float, speed: int,
              turn_boost: float = TURN_BOOST,
              deadband: int = WHEEL_DEADBAND,
              trim_left: float = WHEEL_TRIM_LEFT,
              trim_right: float = WHEEL_TRIM_RIGHT) -> tuple[int, int]:
    """Differential mix: (throttle, steer) in [-1, 1] -> (left, right) PWM.

    Three things happen here that a naive `throttle +/- steer` does not do,
    each fixing a measured failure on this chassis.

    **1. Saturation costs speed, not the turn.** The old mix divided both
    wheels by the larger one, which preserves their RATIO and shrinks their
    DIFFERENCE — and the difference is the turn. Full forward plus full left
    came out as (200, 0): one wheel driving, the other dead. A geared DC motor
    with no power on it does not freewheel, it resists, so that wheel became a
    brake the other one had to drag around while the caster scrubbed. Now the
    pair is shifted bodily into range, which preserves `left - right` exactly:
    the same command becomes (200, -200), both motors pushing the rotation.

    **2. A turn is given more total effort than a straight line.** Rotating
    scrubs the tyres sideways; driving does not. See TURN_BOOST.

    **3. No wheel is left in the stall band.** See _snap_out_of_deadband.

    **The wheels come out swapped on purpose.** On this chassis the motor
    leads are wired mirrored relative to `M <l> <r>`, which is why steering
    read backwards (press A, turn right) while forward/back was fine — a
    swapped pair cancels out when both wheels get the same sign and only
    shows up on a turn. Correcting it here, in the one place that turns
    *intent* into wheels, means every caller inherits the fix: the panel,
    the localization spin, and anything added later. Explicit `left`/`right`
    values passed to `drive()` are NOT touched — those are raw wheel
    commands, and silently rewriting them would make the low-level path
    untestable against the hardware.
    """
    if throttle == 0.0 and steer == 0.0:
        return 0, 0

    left = throttle + steer
    right = throttle - steer

    # Turn-priority clamp: slide BOTH wheels into [-1, 1] by the same amount,
    # so their difference — the turn — survives untouched. Dividing instead
    # would keep the ratio and shrink the turn, which is the bug above.
    over = max(left, right) - 1.0
    if over > 0.0:
        left -= over
        right -= over
    under = -1.0 - min(left, right)
    if under > 0.0:
        left += under
        right += under

    speed = max(0, min(PWM_LIMIT, int(speed)))
    # Выравнивание моторов — ДО мёртвой зоны и до ограничения, чтобы
    # подстройка меняла то, что реально уйдёт на мотор.
    #
    # Множители СКРЕЩЕНЫ намеренно. Эта функция возвращает колёса
    # переставленными (`return out_right, out_left`) — у шасси зеркальная
    # проводка. Значит внутреннее `left` попадает в правый слот команды
    # `M <l> <r>`. Подстройка называется по ФИЗИЧЕСКОМУ колесу, которое вы
    # видите и меряете, поэтому здесь она перекрещивается ровно один раз,
    # в том же месте, где перекрещиваются сами колёса.
    left *= trim_right
    right *= trim_left
    # Extra effort for the scrub, proportional to how hard the turn is, and
    # capped so a boosted turn cannot exceed what the ESP will accept.
    effort = min(float(PWM_LIMIT), speed * (1.0 + turn_boost * min(1.0, abs(steer))))

    out_left = _snap_out_of_deadband(int(round(left * effort)), deadband)
    out_right = _snap_out_of_deadband(int(round(right * effort)), deadband)
    out_left = max(-PWM_LIMIT, min(PWM_LIMIT, out_left))
    out_right = max(-PWM_LIMIT, min(PWM_LIMIT, out_right))
    return out_right, out_left


def clamp_pwm(value: Any) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(-PWM_LIMIT, min(PWM_LIMIT, number))


class _Link:
    """One connected ESP. Replaced wholesale when a new one dials in."""

    def __init__(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        self.conn = conn
        self.addr = addr
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self.repeat_cmd: str | None = None
        self.repeat_until = 0.0
        self.last_hb = ""
        self.lines: list[str] = []
        self.connected_at = time.monotonic()

    def send(self, command: str) -> bool:
        with self._lock:
            try:
                self.conn.sendall((command + "\n").encode("ascii", "replace"))
                return True
            except OSError:
                self.stop.set()
                return False

    def close(self) -> None:
        self.stop.set()
        try:
            self.conn.close()
        except OSError:
            pass


class RobocarService:
    """Discovery + command link, owned by the panel server."""

    def __init__(
        self,
        secret: str = DEFAULT_SECRET,
        discovery_port: int = DISCOVERY_PORT,
        command_port: int = COMMAND_PORT,
        max_pwm: int = DEFAULT_MAX_PWM,
        action_log: Any | None = None,
        drive_profile: Any | None = None,
    ) -> None:
        # How this chassis actually behaves: below the stall threshold the
        # motors draw their PEAK current and do not move, so a command in
        # that band costs more battery than driving. See drive_profile.py.
        from mac_server.drive_profile import DriveProfile

        self.drive_profile = drive_profile or DriveProfile()
        self.last_plan: Any | None = None
        # Every wheel command funnels through drive(), so one hook here
        # captures the complete action stream for behaviour cloning with no
        # risk of a second path being added later and forgotten.
        self.action_log = action_log
        self.max_pwm = max(20, min(PWM_LIMIT, int(max_pwm)))
        self.secret = secret
        self.discovery_port = discovery_port
        self.command_port = command_port
        self.probe = f"ROBOCAR?{secret}"
        self.reply = f"ROBOCAR!{secret}:{command_port}"

        self._link: _Link | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._tcp: socket.socket | None = None
        self._errors: list[str] = []
        self._last_drive = (0, 0)
        # Localization spin state.
        self._spin_thread: threading.Thread | None = None
        self._spin_cancel = threading.Event()
        # Same shape the spin loop publishes, so the panel and the API never
        # see a half-populated dict before the first spin has ever run.
        self._spin: dict[str, Any] = {
            "active": False, "steps": 0,
            "reason": None, "target_deg": SPIN_TARGET_DEG, "mode": None,
        }
        # Camera pan. `None` means "never successfully commanded", which is
        # distinct from "at 0 degrees" — an unknown pan angle must not be
        # reported as a centred one.
        self._pan_deg: float | None = None
        self._pan_supported: bool | None = None

    # ------------------------------------------------------------ lifecycle

    def start(self) -> bool:
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            tcp.bind(("0.0.0.0", self.command_port))
            tcp.listen(1)
        except OSError as exc:
            # Not fatal to the panel — everything else still works, the
            # robot card just reports why it cannot drive.
            self._errors.append(f"TCP {self.command_port}: {exc}")
            logger.error("RoboCar command port %d unavailable: %s", self.command_port, exc)
            tcp.close()
            return False
        self._tcp = tcp
        tcp.settimeout(0.5)

        for target, name in (
            (self._accept_loop, "robocar-accept"),
            (self._discovery_loop, "robocar-discovery"),
            (self._repeat_loop, "robocar-repeat"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        logger.info("RoboCar: discovery on UDP %d, commands on TCP %d (token %s)",
                    self.discovery_port, self.command_port, self.secret)
        return True

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            link = self._link
            self._link = None
        if link is not None:
            link.repeat_cmd = None
            link.send("STOP")   # never leave the motors running on shutdown
            link.close()
        if self._tcp is not None:
            try:
                self._tcp.close()
            except OSError:
                pass
            self._tcp = None
        for thread in self._threads:
            thread.join(timeout=1.5)
        self._threads.clear()

    # --------------------------------------------------------------- threads

    def _accept_loop(self) -> None:
        while not self._stop.is_set() and self._tcp is not None:
            try:
                conn, addr = self._tcp.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info("RoboCar connected from %s:%d", addr[0], addr[1])
            link = _Link(conn, addr)
            with self._lock:
                previous, self._link = self._link, link
            if previous is not None:
                previous.close()
            threading.Thread(target=self._read_loop, args=(link,),
                             name="robocar-read", daemon=True).start()
            # The sketch's own MAX_PWM default (80) was tuned on a bare
            # chassis and barely moves this one. Raise it every time an ESP
            # dials in — the setting lives in the sketch's RAM, so a reset
            # silently reverts it.
            link.send(f"MAX {self.max_pwm}")
            logger.info("RoboCar: MAX_PWM set to %d", self.max_pwm)

    def _read_loop(self, link: _Link) -> None:
        buffer = b""
        while not self._stop.is_set() and not link.stop.is_set():
            try:
                chunk = link.conn.recv(1024)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                text = raw.decode("utf-8", "replace").strip()
                if not text:
                    continue
                with self._lock:
                    if text.startswith("ERR UNKNOWN PAN"):
                        self.note_pan_supported(False)
                    elif text.startswith("OK PAN"):
                        self.note_pan_supported(True)
                    if text.startswith("HB "):
                        link.last_hb = text
                        if self.action_log is not None:
                            # The ESP reports its APPLIED post-ramp PWM here.
                            # It should lag the command by ~one ramp time and
                            # nothing more, which makes it an independent
                            # check on the whole timestamp join.
                            self.action_log.record_heartbeat(text)
                    link.lines.append(text)
                    del link.lines[:-40]
        link.stop.set()
        logger.info("RoboCar disconnected (%s)", link.addr[0])

    def _discovery_loop(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock.bind(("0.0.0.0", self.discovery_port))
        except OSError as exc:
            self._errors.append(f"UDP {self.discovery_port}: {exc}")
            logger.error("RoboCar discovery port %d unavailable: %s — the ESP can still "
                         "reach us if it already knows the address",
                         self.discovery_port, exc)
            sock.close()
            return
        sock.settimeout(0.5)
        reply = self.reply.encode()
        last_beacon = 0.0

        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(256)
                if data.decode("utf-8", "ignore").strip() == self.probe:
                    sock.sendto(reply, addr)
            except (socket.timeout, OSError):
                pass
            now = time.monotonic()
            if now - last_beacon >= BEACON_INTERVAL_S:
                last_beacon = now
                for target in ("255.255.255.255", _broadcast_address()):
                    if not target:
                        continue
                    try:
                        sock.sendto(reply, (target, self.discovery_port))
                    except OSError:
                        pass
        sock.close()

    def _repeat_loop(self) -> None:
        """Retransmit the held command, and expire it when nobody refreshes.

        The expiry is the safety half — see the module docstring. Without
        it a dead browser leaves the robot driving.
        """
        while not self._stop.is_set():
            # `command` is rebound every iteration rather than relying on a
            # value left over from the previous one — a stale command sent to
            # a robot is not a bug you want to find by watching it drive away.
            command: str | None = None
            expired = False
            with self._lock:
                link = self._link
                if link is not None and not link.stop.is_set() and link.repeat_cmd:
                    if time.monotonic() > link.repeat_until:
                        link.repeat_cmd = None
                        self._last_drive = (0, 0)
                        expired = True
                    else:
                        command = link.repeat_cmd
                else:
                    link = None
            if link is not None:
                if expired:
                    logger.warning("RoboCar: no drive command for %.1fs — stopping",
                                   COMMAND_TTL_S)
                    link.send("STOP")
                elif command is not None:
                    link.send(command)
            time.sleep(REPEAT_INTERVAL_S)

    # ----------------------------------------------------------------- public

    def drive(
        self,
        left: int,
        right: int,
        _internal: bool = False,
        source: str = "api",
        throttle: float | None = None,
        steer: float | None = None,
        speed: int | None = None,
    ) -> bool:
        """Set the wheel PWMs. Returns False when no ESP is connected.

        Any call from outside (the panel, the HTTP API) aborts a running
        localization spin: the operator grabbing the keys must always win
        over an automatic manoeuvre, immediately and without a separate
        cancel step.
        """
        if not _internal:
            self.cancel_spin("operator took over")
        left, right = clamp_pwm(left), clamp_pwm(right)

        # Shape the request out of the stall band. Recorded before the
        # connection check so the dataset sees what was ASKED for even when
        # the robot was not listening.
        from mac_server.drive_profile import energy_note, shape

        plan = shape(left, right, self.drive_profile)
        with self._lock:
            self.last_plan = {"requested": [left, right],
                               "sent": [plan.left, plan.right],
                               "reason": plan.reason,
                               **energy_note(plan, self.drive_profile)}
        left, right = plan.left, plan.right
        # Logged BEFORE the connection check: "the operator asked for full
        # throttle while the robot was disconnected" is a real event, and a
        # dataset that quietly omits it has a gap it cannot explain.
        if self.action_log is not None:
            self.action_log.record_drive(
                left, right, source=source, max_pwm=self.max_pwm,
                throttle=throttle, steer=steer, speed=speed,
            )
        with self._lock:
            link = self._link
            if link is None or link.stop.is_set():
                return False
            self._last_drive = (left, right)
            if left == 0 and right == 0:
                had = link.repeat_cmd is not None
                link.repeat_cmd = None
                command = "STOP" if had else None
            else:
                command = f"M {left} {right}"
                link.repeat_cmd = command
                link.repeat_until = time.monotonic() + COMMAND_TTL_S
        return link.send(command) if command else True

    def send_command(self, command: str) -> tuple[bool, str]:
        """Pass one of `ALLOWED_COMMANDS` (or `MAX <n>`) through to the ESP."""
        command = (command or "").strip().upper()
        if command.startswith("MAX "):
            try:
                value = max(20, min(PWM_LIMIT, int(command[4:])))
            except ValueError:
                return False, "MAX needs a number"
            command = f"MAX {value}"
        elif command not in ALLOWED_COMMANDS:
            return False, f"command not allowed: {command!r}"
        with self._lock:
            link = self._link
            if link is None or link.stop.is_set():
                return False, "no robot connected"
            if command in ("STOP", "BRAKE"):
                link.repeat_cmd = None
                self._last_drive = (0, 0)
                stop_spin = True
            else:
                stop_spin = False
        if stop_spin:
            self.cancel_spin("STOP pressed")
        return (True, "sent") if link.send(command) else (False, "send failed")

    # ------------------------------------------------------------------ pan

    def set_pan(self, angle_deg: float) -> tuple[bool, str]:
        """Point the camera at a fixed angle. Only while the wheels are stopped.

        The stop requirement is not politeness: on the ESP8266 the servo and
        the motor PWM contend for the same timer, so the sketch detaches the
        servo while driving. Refusing here rather than there means the caller
        gets a clear reason instead of a silently ignored command.
        """
        angle = max(-90.0, min(90.0, float(angle_deg)))
        with self._lock:
            link = self._link
            if link is None or link.stop.is_set():
                return False, "no robot connected"
            if link.repeat_cmd:
                return False, "refusing to pan while the wheels are driving"
            if self._pan_supported is False:
                return False, "this firmware has no PAN command — reflash the sketch"
        if not link.send(f"PAN {angle:.0f}"):
            return False, "send failed"
        with self._lock:
            self._pan_deg = angle
        return True, f"pan {angle:+.0f}"

    def pan_survey(
        self,
        angles: tuple[float, ...] = PAN_ANGLES_DEG,
        settle_s: float = PAN_SETTLE_S,
    ) -> list[float]:
        """Step the camera through the look positions, pausing at each.

        Returns the angles actually reached. The pause is what makes the
        view usable: the place index is matched on a CLIP embedding of one
        frame, and a frame taken mid-travel is motion-blurred into something
        that matches nothing — the same reason the chassis spin is pulsed.
        """
        reached = []
        for angle in angles:
            ok, _detail = self.set_pan(angle)
            if not ok:
                break
            if self._stop.wait(settle_s):
                break
            reached.append(angle)
        if reached:
            self.set_pan(PAN_CENTRE_DEG)
        return reached

    def note_pan_supported(self, supported: bool) -> None:
        """Learned from the ESP's reply: `ERR UNKNOWN PAN` means old firmware.

        Recorded so the survey stops asking and falls back to the chassis
        spin, rather than pausing pointlessly at every angle.
        """
        with self._lock:
            self._pan_supported = supported

    # ------------------------------------------------------- localization spin

    def start_spin(
        self,
        speed: int = SPIN_SPEED,
        target_deg: float = SPIN_TARGET_DEG,
        timeout_s: float = SPIN_TIMEOUT_S,
    ) -> tuple[bool, str]:
        """Rotate in place so the camera gets a look at the whole room.

        Timed, not measured: this chassis has no wheel encoders, and the IMU
        heading that used to terminate the spin came from the metric
        localization stack, which is gone. `SPIN_BLIND_STEPS` pulses is what
        was measured to come out near a revolution on this robot.
        """
        with self._lock:
            if self._link is None or self._link.stop.is_set():
                return False, "no robot connected"
            if self._spin_thread is not None and self._spin_thread.is_alive():
                return False, "already spinning"
            # Blind fallback rather than a refusal. Requiring a heading
            # deadlocked this: the spin exists to GET localized, but the Pi
            # only reported heading once it already was. That specific
            # ordering bug is fixed, but the principle stands — a survey
            # spin is useful even with no IMU at all, and "the button does
            # nothing" is the worst possible failure mode.
            pan_first = self._pan_supported is not False
            self._spin_cancel.clear()
            self._spin = {"active": True, "reason": None,
                          "target_deg": target_deg, "steps": 0,
                          "mode": "timed"}
        self._spin_thread = threading.Thread(
            target=self._survey_loop, args=(speed, target_deg, timeout_s, pan_first),
            name="robocar-spin", daemon=True,
        )
        self._spin_thread.start()
        return True, f"spinning: {SPIN_BLIND_STEPS} timed steps"

    def cancel_spin(self, reason: str = "cancelled") -> None:
        if self._spin_thread is not None and self._spin_thread.is_alive():
            with self._lock:
                self._spin["reason"] = reason
            self._spin_cancel.set()

    def _survey_loop(self, speed: int, target_deg: float, timeout_s: float,
                     pan_first: bool) -> None:
        """Look around the cheap way first, then rotate only if needed.

        Panning the camera costs milliamps; rotating the chassis scrubs both
        wheels sideways and is the most expensive manoeuvre this robot has.
        With three pan positions the query gets three distinct views before a
        single wheel turns, and on a robot whose battery dies in minutes that
        ordering is worth more than the code it takes.
        """
        if pan_first:
            reached = self.pan_survey()
            if reached:
                logger.info("Pan survey covered %s deg without moving the chassis",
                            [f"{a:+.0f}" for a in reached])
                with self._lock:
                    self._spin["pan_angles"] = reached
        if self._spin_cancel.is_set() or self._stop.is_set():
            with self._lock:
                self._spin = {**self._spin, "active": False, "reason": "cancelled"}
            return
        self._spin_loop(speed, target_deg, timeout_s)

    def _spin_loop(self, speed: int, target_deg: float, timeout_s: float) -> None:
        # No turn boost here. SPIN_SPEED was MEASURED against this exact
        # scrub — 110 would not rotate the chassis at all and 160 does — so
        # it already contains the compensation TURN_BOOST exists to add.
        # Applying both would put ~232 on the motors, past what the pulsed
        # spin was tuned for.
        left, right = mix_drive(0.0, 1.0, speed, turn_boost=0.0)
        deadline = time.monotonic() + timeout_s
        steps = 0
        reason = "done"

        while not self._spin_cancel.is_set() and not self._stop.is_set():
            if time.monotonic() > deadline:
                reason = f"timed out after {timeout_s:.0f}s at {steps} steps"
                break
            if steps >= SPIN_BLIND_STEPS:
                reason = f"finished after {steps} steps"
                break

            # Pulse.
            pulse_until = time.monotonic() + SPIN_PULSE_S
            while time.monotonic() < pulse_until:
                if self._spin_cancel.is_set() or self._stop.is_set():
                    break
                if not self.drive(left, right, _internal=True, source="spin"):
                    reason = "robot disconnected"
                    break
                time.sleep(0.1)
            if reason != "done":
                break
            self.drive(0, 0, _internal=True, source="spin")
            steps += 1

            # Settle, then read the heading while the chassis is at rest.
            if self._spin_cancel.wait(SPIN_SETTLE_S):
                break
            with self._lock:
                self._spin["steps"] = steps

        if self._spin_cancel.is_set():
            with self._lock:
                reason = self._spin["reason"] or "cancelled"
        self.drive(0, 0, _internal=True, source="spin")
        with self._lock:
            self._spin = {"active": False, "reason": reason,
                          "target_deg": target_deg, "steps": steps,
                          "mode": "timed"}
        logger.info("RoboCar spin finished: %d steps (%s)", steps, reason)

    def status(self) -> dict[str, Any]:
        with self._lock:
            link = self._link
            connected = link is not None and not link.stop.is_set()
            return {
                "connected": connected,
                "address": link.addr[0] if connected and link else None,
                "uptime_s": round(time.monotonic() - link.connected_at, 1)
                if connected and link else None,
                "heartbeat": link.last_hb if link else "",
                "recent": list(link.lines[-8:]) if link else [],
                "drive": list(self._last_drive),
                "holding": bool(link.repeat_cmd) if link else False,
                "spin": dict(self._spin),
                "pan_deg": self._pan_deg,
                "pan_supported": self._pan_supported,
                "pan_angles_deg": list(PAN_ANGLES_DEG),
                "drive_plan": self.last_plan,
                "drive_profile_measured": self.drive_profile.measured,
                "max_pwm": self.max_pwm,
                "command_port": self.command_port,
                "discovery_port": self.discovery_port,
                "errors": list(self._errors),
            }


def _broadcast_address() -> str:
    """Crude /24 broadcast guess, same as the original script."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        parts = probe.getsockname()[0].split(".")
    except OSError:
        return ""
    finally:
        probe.close()
    return ".".join(parts[:3] + ["255"]) if len(parts) == 4 else ""


@dataclass(frozen=True)
class DriveCommand:
    """What was asked for, and what the wheels were told to do.

    Both halves are kept because they answer different questions and one
    cannot be recovered from the other. `left`/`right` are ground truth for
    what the hardware did and are what a learned policy must predict.
    `throttle`/`steer` are the operator's *intent*, and they survive a change
    in wiring: `mix_drive` currently swaps the wheels to compensate this
    chassis' mirrored motor leads, so if that wiring is ever fixed, a
    wheels-only recording becomes silently wrong while an intent recording
    stays valid.
    """

    left: int
    right: int
    throttle: float | None = None
    steer: float | None = None
    speed: int | None = None
    form: str = "wheels"        # "wheels" (explicit) or "mixed" (from intent)


_TRIM_CACHE: dict[str, float] | None = None


def _wheel_trim() -> dict[str, float]:
    """Подстройка колёс из конфига, читается один раз за процесс.

    Файл, а не аргумент: `parse_drive_payload` — функция модуля, её зовут из
    HTTP-обработчика без доступа к конфигу сервера, и протаскивать его через
    всю цепочку ради двух чисел значит менять сигнатуры на пути.
    Кэш за процесс, поэтому после правки подстройки сервер надо перезапустить
    — скрипт замера об этом прямо предупреждает.
    """
    global _TRIM_CACHE
    if _TRIM_CACHE is None:
        _TRIM_CACHE = {"left": 1.0, "right": 1.0}
        try:
            path = Path(__file__).resolve().parents[3] / "config/default.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            trim = (data.get("robot") or {}).get("wheel_trim") or {}
            _TRIM_CACHE = {"left": float(trim.get("left", 1.0)),
                           "right": float(trim.get("right", 1.0))}
        except Exception:  # noqa: BLE001 — без конфига едем без подстройки
            pass
    return _TRIM_CACHE


def parse_drive_payload(body: bytes) -> DriveCommand | None:
    """Accept either explicit wheel PWMs or a (throttle, steer, speed) trio.

    The panel sends the latter — it holds the keys, not the mixing rules —
    but keeping the raw form makes the endpoint scriptable for tests.
    """
    try:
        payload = json.loads(body or b"{}")
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    if "left" in payload or "right" in payload:
        return DriveCommand(
            left=clamp_pwm(payload.get("left")),
            right=clamp_pwm(payload.get("right")),
            form="wheels",
        )
    try:
        throttle = max(-1.0, min(1.0, float(payload.get("throttle", 0.0))))
        steer = max(-1.0, min(1.0, float(payload.get("steer", 0.0))))
        speed = int(payload.get("speed", 160))
    except (TypeError, ValueError):
        return None
    trim = _wheel_trim()
    left, right = mix_drive(throttle, steer, speed,
                            trim_left=trim["left"], trim_right=trim["right"])
    return DriveCommand(left=left, right=right, throttle=throttle,
                        steer=steer, speed=speed, form="mixed")
