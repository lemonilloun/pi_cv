# Moving the Mac-server role to a new laptop

Context: `CLAUDE.md`'s "Reconstruction quality ceiling on the M2" section —
the Scene3D server role (currently run from this M2) is moving to a
different laptop with SSH access to a 2x A6000 GPU host (`ssh pdfserver`,
run from *that* laptop only — this Mac and the Pi never talk to it
directly). The Pi's recording side does not change at all; only which
machine it streams keyframes to changes.

## 1. Get the new laptop's LAN IP

On the **new laptop**:

```bash
./scripts/check_lan_ip.sh
```

Run this from a clone of this repo, or copy the script over standalone —
it has no dependencies beyond standard `ifconfig`/`ip`. It prints the
laptop's LAN IP(s) and, if `cv-pi.local` resolves from that machine, tries
a quick reachability check against the Pi's TCP port so you know before
editing any config whether the two machines are even on the same network.

## 2. SSH key so the Pi can push to the new laptop

The Pi doesn't need to SSH into the new laptop for the normal camera/scene
protocol (that's a raw TCP connection on port 8765, not SSH) — this step is
only needed if you also want to `ssh` into the new laptop directly for
debugging/deployment, same as this Mac already does for `cv-pi.local`.

On the **new laptop**:

```bash
ssh-keygen -t ed25519 -C "new-laptop-pi-cv" -f ~/.ssh/id_ed25519_pi_cv
ssh-copy-id -i ~/.ssh/id_ed25519_pi_cv.pub first@cv-pi.local
```

`ssh-copy-id` will ask for the Pi's password once. After that, add a
`~/.ssh/config` entry on the new laptop:

```
Host cv-pi.local cv-pi
  HostName cv-pi.local
  User first
  IdentityFile ~/.ssh/id_ed25519_pi_cv
```

Verify:

```bash
ssh cv-pi.local "hostname && whoami"
```

If `cv-pi.local` doesn't resolve (mDNS is flaky across network changes —
already hit this earlier in the project when the Pi moved apartments), find
the Pi's current IP via its MAC address in the new laptop's ARP table
(`arp -a`) and use that IP directly instead of the `.local` hostname, same
workaround already used elsewhere in this project.

## 3. Point the Pi at the new laptop

On the **Pi** (`ssh cv-pi.local`, then):

```bash
cd ~/Desktop/work/pi_cv
nano .env   # or any editor
```

Change:

```env
PI_CV_SERVER_HOST=vedro.local
```

to the new laptop's hostname (if it resolves via mDNS from the Pi) or the
IP address from step 1. Leave `PI_CV_SERVER_PORT=8765` unless the new
laptop's server is configured on a different port.

## 4. Start the server on the new laptop, confirm the Pi connects

On the **new laptop**:

```bash
./scripts/run_server.sh
```

On the **Pi**:

```bash
cd ~/Desktop/work/pi_cv && source .venv/bin/activate
./scripts/run_pi_session.sh
```

Confirm in the new laptop's server log (or `http://<new-laptop-host>:8080/`)
that the Pi's `session_hello` registered — same check as always, just
against the new host.

## What does NOT change

- The Pi's recording code, IMU driver, calibration files
  (`config/imu_calibration_shtp.json`), and scene_recorder invocation are
  all identical regardless of which laptop is the server — the session
  directory contract (`meta.json` schema, keyframe layout) is the interface
  point and isn't touched by this migration.
- `data/scene_sessions/` accumulates on whichever machine is currently
  running the server — sessions recorded against this M2 stay on this M2's
  disk; they don't automatically follow the migration. Copy/sync
  `data/scene_sessions/<id>/` manually if you want to re-run the pipeline
  for an old session on the new laptop.
