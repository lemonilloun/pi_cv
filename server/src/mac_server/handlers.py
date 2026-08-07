"""Message handlers for the MacBook server."""

from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
import time
import zipfile
from io import BytesIO

from mac_server.protocol import make_ack_response, make_cv_result_ack_response, make_image_ack_response
from mac_server.preview import LatestFrameStore
from mac_server.registry import ClientHandle, ClientRegistry, TelemetryStore
from shared.messages import Message


logger = logging.getLogger(__name__)

QUIET_MESSAGE_TYPES = {
    "camera_stream_frame",
    "system_telemetry",
    "nav_query",
    "scene_keyframe",
    "imu_cal_state",  # ~5 Hz while the calibration wizard runs
    "time_sync",      # a 21-message burst at each end of every episode
    "vla_frame",      # 10 Hz for the whole of a recording episode
}


@dataclass
class SessionContext:
    """Per-connection context so handlers can register session clients."""

    registry: ClientRegistry
    telemetry_store: TelemetryStore
    client_socket: socket.socket
    client_address: tuple[str, int]
    send_lock: threading.Lock


def handle_message(
    message: Message,
    binary_payload: bytes = b"",
    storage_dir: Path | None = None,
    preview_store: LatestFrameStore | None = None,
    session_context: SessionContext | None = None,
) -> Message:
    log = logger.debug if message.type in QUIET_MESSAGE_TYPES else logger.info
    log(
        "Handling message from device_id=%s type=%s payload=%s binary_payload_bytes=%s",
        message.device_id,
        message.type,
        message.payload,
        len(binary_payload),
    )

    if message.type == "image":
        if storage_dir is None:
            raise ValueError("storage_dir is required for image messages")
        return _handle_image_message(message, binary_payload, storage_dir)

    if message.type == "camera_frame":
        if storage_dir is None:
            raise ValueError("storage_dir is required for camera frame messages")
        return _handle_camera_frame_message(message, binary_payload, storage_dir)

    if message.type == "camera_stream_frame":
        return _handle_camera_stream_frame_message(message, binary_payload, storage_dir, preview_store)

    if message.type == "cv_result":
        if storage_dir is None:
            raise ValueError("storage_dir is required for CV result messages")
        return _handle_cv_result_message(message, binary_payload, storage_dir)

    if message.type == "session_hello" and session_context is not None:
        return _handle_session_hello_message(message, session_context)

    if message.type == "system_telemetry" and session_context is not None:
        return _handle_system_telemetry_message(message, session_context)

    if message.type == "command_result" and session_context is not None:
        return _handle_command_result_message(message, session_context)

    if message.type in {"scene_session_start", "scene_keyframe", "scene_session_end"}:
        if storage_dir is None:
            raise ValueError("storage_dir is required for scene messages")
        return _handle_scene_message(message, binary_payload, storage_dir)

    if message.type == "nav_query":
        if storage_dir is None:
            raise ValueError("storage_dir is required for nav queries")
        return _handle_nav_query(message, storage_dir)

    if message.type == "imu_cal_state":
        return _handle_imu_cal_state(message)

    if message.type == "time_sync":
        return _handle_time_sync(message)

    if message.type in {"vla_session_start", "vla_frame", "vla_session_end"}:
        if storage_dir is None:
            raise ValueError("storage_dir is required for VLA episode messages")
        return _handle_vla_message(message, binary_payload, storage_dir)

    return make_ack_response(message)


# Latest state of the Pi's guided IMU calibration wizard, served to the
# panel by GET /api/imu_cal/status. Deliberately just the last message:
# the wizard is the authority on the measurement and writes its own CSVs
# and summary.json on the Pi — this is a live view, not a record.
_imu_cal_state: dict[str, object] | None = None
_imu_cal_lock = threading.Lock()


def _handle_imu_cal_state(message: Message) -> Message:
    global _imu_cal_state
    with _imu_cal_lock:
        _imu_cal_state = dict(message.payload)
    return make_ack_response(message)


def _handle_time_sync(message: Message) -> Message:
    """Answer with this machine's monotonic clock, as promptly as possible.

    Deliberately does nothing else — no logging, no locks, no lookups. Every
    microsecond spent here lands inside the measured round trip and inflates
    the offset's error bar. `t0_pi_ns` is echoed so the Pi can match replies
    to requests without keeping state.
    """
    return Message(
        device_id="mac_server",
        type="time_sync_reply",
        payload={
            "mac_monotonic_ns": time.monotonic_ns(),
            "mac_wall_ns": time.time_ns(),
            "t0_pi_ns": message.payload.get("t0_pi_ns"),
            "seq": message.payload.get("seq", 0),
        },
    )


# The robot link, registered by server.py. nav_query is the only place the
# server sees the Pi's live heading, and the localization spin needs it to
# know when a full revolution is done — see robocar.RobocarService.note_heading.
_robot_service: object | None = None


def set_robot_service(service: object | None) -> None:
    global _robot_service
    _robot_service = service


# The action log, registered by server.py. Armed automatically when the Pi
# announces an episode, rather than by a separate panel button: the recorder
# is the authority on when an episode starts, so making the server follow it
# removes the whole class of "recorded 200 frames, forgot to arm the actions"
# failures — which the very first smoke episode already hit.
_action_log: object | None = None


def set_action_log(log: object | None) -> None:
    global _action_log
    _action_log = log


def get_imu_cal_state() -> dict[str, object] | None:
    with _imu_cal_lock:
        return dict(_imu_cal_state) if _imu_cal_state else None


def _handle_nav_query(message: Message, storage_dir: Path) -> Message:
    """Localize the Pi: match its live CLIP embedding against the place
    indexes of all reconstructed sessions (scene3d/navindex)."""
    from mac_server.scene3d import navindex

    embedding = message.payload.get("clip_emb")
    if not isinstance(embedding, list) or len(embedding) < 8:
        raise ValueError("nav_query needs a clip_emb list")
    sessions_dir = storage_dir.parent / "scene_sessions"
    result = navindex.query(sessions_dir, embedding)
    depth = message.payload.get("depth_center_m")
    if depth is not None:
        result["depth_center_m"] = depth

    # The Pi's own fused EKF state, if it is running one. Converted here
    # rather than on the Pi because the metres->plan_frac mapping needs the
    # session's plan_frame, which lives with the index on this side.
    # Mutating `result` in place is deliberate: navindex.query already
    # stored this same object as `last_result`, which is what
    # GET /api/nav/last serves to the panel.
    fused = message.payload.get("fused")
    if isinstance(fused, dict) and result.get("located"):
        position_m = fused.get("position_m")
        if isinstance(position_m, list) and len(position_m) >= 2:
            projected = navindex.fused_to_plan(
                sessions_dir,
                result["best"]["session_id"],
                position_m,
                heading_deg=fused.get("heading_deg"),
                std_m=fused.get("std_m"),
            )
            if projected:
                projected["filter"] = fused.get("filter")
                # Pass the filter's own confidence through untouched. For the
                # RVC heading filter, `yaw_bias_std_deg` falling is the signal
                # that IMU yaw has been tied to this room's plan frame and can
                # carry heading between visual fixes; the panel shows it so a
                # drifting or unconverged filter is visible rather than
                # silently producing a confident-looking arrow.
                for key in ("heading_std_deg", "yaw_bias_deg", "yaw_bias_std_deg", "bias",
                            "imu_yaw_deg", "imu_pitch_deg", "imu_roll_deg"):
                    if fused.get(key) is not None:
                        projected[key] = fused[key]
                result["fused"] = projected

    # Raw attitude, forwarded regardless of whether the fix landed. The
    # localization spin needs the heading exactly when the robot is NOT
    # localized yet, so gating this on `located` deadlocked it.
    imu = message.payload.get("imu")
    if isinstance(imu, dict):
        result["imu"] = imu
        if _robot_service is not None:
            _robot_service.note_heading(imu.get("yaw_deg"))
    return Message(
        device_id="mac_server",
        type="nav_result",
        payload=result,
    )


# ------------------------------------------------------------- scene3d
# The scene recorder streams keyframes online (docs/scene3d.md) so nothing
# accumulates on the Pi's microSD; the server materializes the exact same
# session directory layout the offline/rsync path would produce.


def _scene_session_dir(storage_dir: Path, scene_session: str) -> Path:
    safe = "".join(c for c in scene_session if c.isalnum() or c == "_")
    if not safe:
        raise ValueError(f"Invalid scene session name: {scene_session!r}")
    # storage_dir is data/received; sessions live next to it in data/.
    return storage_dir.parent / "scene_sessions" / safe


def _handle_scene_message(message: Message, binary_payload: bytes, storage_dir: Path) -> Message:
    import json

    payload = message.payload
    session_dir = _scene_session_dir(storage_dir, str(payload.get("scene_session", "")))

    if message.type == "scene_session_start":
        (session_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        intrinsics = payload.get("intrinsics")
        if intrinsics:
            (session_dir / "intrinsics.json").write_text(
                json.dumps(intrinsics, indent=2), encoding="utf-8"
            )
        logger.info("Scene session started: %s (intrinsics: %s)",
                    session_dir.name, "yes" if intrinsics else "no")
        return make_ack_response(message)

    if message.type == "scene_keyframe":
        rgb_bytes = int(payload.get("rgb_bytes", 0))
        masks_bytes = int(payload.get("masks_bytes", 0))
        if rgb_bytes <= 0 or rgb_bytes + masks_bytes != len(binary_payload):
            raise ValueError(
                f"scene_keyframe payload mismatch: rgb={rgb_bytes} masks={masks_bytes} "
                f"actual={len(binary_payload)}"
            )
        frame_idx = int(payload.get("frame_idx", 0))
        kf_dir = session_dir / "keyframes" / f"{frame_idx:06d}"
        kf_dir.mkdir(parents=True, exist_ok=True)
        (kf_dir / "rgb.jpg").write_bytes(binary_payload[:rgb_bytes])
        if masks_bytes > 0:
            (kf_dir / "masks.png").write_bytes(binary_payload[rgb_bytes:])
        (kf_dir / "meta.json").write_text(
            json.dumps(payload.get("meta", {}), indent=1), encoding="utf-8"
        )
        return make_ack_response(message)

    # scene_session_end
    (session_dir).mkdir(parents=True, exist_ok=True)
    (session_dir / "session_meta.json").write_text(
        json.dumps(payload.get("session_meta", {}), indent=2), encoding="utf-8"
    )
    logger.info("Scene session finished: %s", session_dir.name)
    return make_ack_response(message)


def _handle_session_hello_message(message: Message, context: SessionContext) -> Message:
    handle = ClientHandle(
        device_id=message.device_id,
        session_id=str(message.payload.get("session_id", "")),
        sock=context.client_socket,
        send_lock=context.send_lock,
        address=context.client_address,
        hello_payload=dict(message.payload),
        mode=str(message.payload.get("mode", "idle")),
    )
    context.registry.register(handle)
    return make_ack_response(message)


def _handle_system_telemetry_message(message: Message, context: SessionContext) -> Message:
    context.telemetry_store.add(message.device_id, message.payload)
    handle = context.registry.find_by_socket(context.client_socket)
    if handle is not None and message.payload.get("mode"):
        handle.mode = str(message.payload["mode"])
    return make_ack_response(message)


def _handle_command_result_message(message: Message, context: SessionContext) -> Message:
    handle = context.registry.find_by_socket(context.client_socket)
    if handle is not None:
        handle.last_command_result = dict(message.payload)
        if message.payload.get("ok") and message.payload.get("mode"):
            handle.mode = str(message.payload["mode"])
    logger.info(
        "Command result from device_id=%s command_id=%s ok=%s mode=%s error=%s",
        message.device_id,
        message.payload.get("command_id"),
        message.payload.get("ok"),
        message.payload.get("mode"),
        message.payload.get("error"),
    )
    return make_ack_response(message)


def _handle_image_message(message: Message, binary_payload: bytes, storage_dir: Path) -> Message:
    expected_byte_count = int(message.payload.get("byte_count", -1))
    actual_byte_count = len(binary_payload)

    if expected_byte_count != actual_byte_count:
        raise ValueError(
            f"Image byte count mismatch: expected {expected_byte_count}, got {actual_byte_count}"
        )

    original_filename = Path(str(message.payload.get("filename", "image.jpg"))).name
    timestamp_ms = int(time.time() * 1000)
    saved_path = storage_dir / f"{timestamp_ms}_{original_filename}"

    storage_dir.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(binary_payload)

    logger.info("Saved image to %s", saved_path)
    return make_image_ack_response(message, str(saved_path), actual_byte_count)


def _handle_camera_frame_message(message: Message, binary_payload: bytes, storage_dir: Path) -> Message:
    expected_byte_count = int(message.payload.get("byte_count", -1))
    actual_byte_count = len(binary_payload)

    if expected_byte_count != actual_byte_count:
        raise ValueError(
            f"Camera frame byte count mismatch: expected {expected_byte_count}, got {actual_byte_count}"
        )

    saved_path = _save_camera_frame(message, binary_payload, storage_dir / "camera")
    logger.info("Saved camera frame to %s", saved_path)
    return make_image_ack_response(message, str(saved_path), actual_byte_count)


def _handle_camera_stream_frame_message(
    message: Message,
    binary_payload: bytes,
    storage_dir: Path | None,
    preview_store: LatestFrameStore | None,
) -> Message:
    expected_byte_count = int(message.payload.get("byte_count", -1))
    actual_byte_count = len(binary_payload)

    if expected_byte_count != actual_byte_count:
        raise ValueError(
            f"Camera stream frame byte count mismatch: expected {expected_byte_count}, got {actual_byte_count}"
        )

    if preview_store is not None:
        preview_store.update(binary_payload, message.payload)

    saved_path = ""
    if bool(message.payload.get("save_frame")):
        if storage_dir is None:
            raise ValueError("storage_dir is required to save stream frames")
        saved_path = str(_save_camera_frame(message, binary_payload, storage_dir / "stream"))

    frame_index = int(message.payload.get("frame_index", 0))
    log = logger.info if frame_index == 1 or frame_index % 30 == 0 else logger.debug
    log(
        "Received stream frame session=%s index=%s bytes=%s saved=%s",
        message.payload.get("session_id"),
        frame_index,
        actual_byte_count,
        bool(saved_path),
    )
    return make_image_ack_response(message, saved_path, actual_byte_count)


def _save_camera_frame(message: Message, binary_payload: bytes, target_dir: Path) -> Path:
    frame_id = _safe_filename_part(str(message.payload.get("frame_id", "frame")))
    device_id = _safe_filename_part(message.device_id)
    image_format = str(message.payload.get("format", "jpeg"))
    extension = "jpg" if image_format == "jpeg" else image_format
    timestamp_ms = int(time.time() * 1000)

    saved_path = target_dir / f"{timestamp_ms}_{device_id}_{frame_id}.{extension}"
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(binary_payload)
    return saved_path


def _handle_cv_result_message(message: Message, binary_payload: bytes, storage_dir: Path) -> Message:
    expected_byte_count = int(message.payload.get("byte_count", -1))
    actual_byte_count = len(binary_payload)

    if expected_byte_count != actual_byte_count:
        raise ValueError(f"CV result byte count mismatch: expected {expected_byte_count}, got {actual_byte_count}")

    run_id = _safe_filename_part(str(message.payload.get("run_id", "cv_run")))
    target_dir = storage_dir / "cv" / run_id
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir / "result.zip"
    zip_path.write_bytes(binary_payload)

    try:
        with zipfile.ZipFile(BytesIO(binary_payload)) as archive:
            _safe_extract_zip(archive, target_dir)
    except zipfile.BadZipFile as exc:
        raise ValueError("CV result payload is not a valid zip file") from exc

    logger.info(
        "Saved CV result run_id=%s pipeline=%s bytes=%s dir=%s",
        run_id,
        message.payload.get("pipeline_type"),
        actual_byte_count,
        target_dir,
    )
    return make_cv_result_ack_response(
        request=message,
        saved_path=str(zip_path),
        extracted_dir=str(target_dir),
        byte_count=actual_byte_count,
    )


def _safe_extract_zip(archive: zipfile.ZipFile, target_dir: Path) -> None:
    resolved_target = target_dir.resolve()
    for member in archive.infolist():
        member_path = target_dir / member.filename
        resolved_member = member_path.resolve()
        if resolved_target != resolved_member and resolved_target not in resolved_member.parents:
            raise ValueError(f"Unsafe zip member path: {member.filename}")
    archive.extractall(target_dir)


def _safe_filename_part(value: str) -> str:
    safe_chars = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe_value = "".join(safe_chars).strip("_")
    return safe_value or "unknown"


# --------------------------------------------------------------- VLA episodes
# Behaviour-cloning data. Frames land here from the Pi; the matching actions
# are logged on this machine by mac_server/vla/action_log.py, and
# vla/build_dataset.py joins the two using the clock offset recorded at
# episode start.


def _vla_episode_dir(storage_dir: Path, episode_id: str) -> Path:
    """Reject an unclean episode id rather than sanitizing it.

    `_scene_session_dir` above strips unsafe characters, which is fine there:
    a mangled scene name costs nothing. Here it would be a data-integrity
    bug — sanitizing maps several distinct ids onto one directory, and two
    episodes silently merged into a single folder produce a training example
    whose frames come from two different demonstrations. The recorder
    generates ids itself (`ep_YYYYmmdd_HHMMSS`), so anything that needs
    cleaning means something upstream is wrong and should say so.
    """
    safe = "".join(c for c in episode_id if c.isalnum() or c == "_")
    if not safe or safe != episode_id:
        raise ValueError(
            f"Invalid episode id {episode_id!r}: only letters, digits and underscore. "
            f"Ids are not sanitized here because two ids collapsing onto one "
            f"directory would merge two demonstrations into one episode."
        )
    return storage_dir.parent / "vla_episodes" / safe


def _handle_vla_message(message: Message, binary_payload: bytes, storage_dir: Path) -> Message:
    import json

    payload = message.payload
    episode_dir = _vla_episode_dir(storage_dir, str(payload.get("episode_id", "")))

    if message.type == "vla_session_start":
        (episode_dir / "frames").mkdir(parents=True, exist_ok=True)
        if _action_log is not None:
            _action_log.start_episode(str(payload.get("episode_id")), episode_dir)
        (episode_dir / "episode_meta.json").write_text(
            json.dumps({
                "episode_id": payload.get("episode_id"),
                "task": payload.get("task"),
                "settings": payload.get("settings"),
                "clock": payload.get("clock"),
                "device_id": message.device_id,
                "started_wall": time.time(),
            }, indent=1),
            encoding="utf-8",
        )
        return make_ack_response(message)

    if message.type == "vla_frame":
        expected = int(payload.get("jpeg_bytes", 0))
        if expected != len(binary_payload):
            raise ValueError(
                f"vla_frame byte count mismatch: header {expected}, got {len(binary_payload)}"
            )
        frame_idx = int(payload.get("frame_idx", 0))
        (episode_dir / "frames").mkdir(parents=True, exist_ok=True)
        (episode_dir / "frames" / f"{frame_idx:06d}.jpg").write_bytes(binary_payload)
        # One JSONL line per frame rather than a file per frame: at 10 Hz a
        # 30 s episode is 300 frames, and 300 tiny JSON files cost far more in
        # inode churn than one append-only log.
        with (episode_dir / "frames.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "frame_idx": frame_idx,
                "t_pi_mono_ns": payload.get("t_pi_mono_ns"),
                "imu": payload.get("imu"),
            }) + "\n")
        return make_ack_response(message)

    # vla_session_end
    meta_path = episode_dir / "episode_meta.json"
    existing = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            existing = {}
    existing.update(payload.get("episode_meta") or {})
    existing["ended_wall"] = time.time()
    if _action_log is not None:
        existing["action_log"] = _action_log.status()
        _action_log.stop_episode()
    episode_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(existing, indent=1), encoding="utf-8")
    logger.info("VLA episode %s closed: %s frames",
                payload.get("episode_id"), existing.get("frames_sent"))
    return make_ack_response(message)
