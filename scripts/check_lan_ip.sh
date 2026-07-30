#!/usr/bin/env bash
# Run this ON THE NEW LAPTOP that will take over the Mac-server role
# (see docs/new_laptop_setup.md). Prints this machine's LAN IP(s) and, if
# possible, checks whether the Pi (cv-pi.local) is reachable from here
# before you go update its .env to point at this machine.
set -uo pipefail

echo "=== LAN IP(s) on this machine ==="
if command -v ifconfig >/dev/null 2>&1; then
    # macOS/BSD: list IPv4 addresses on active interfaces, skip loopback.
    ifconfig | awk '
        /^[a-z0-9]+:/ { iface=$1 }
        /inet /  && $2 != "127.0.0.1" { print iface, $2 }
    '
elif command -v ip >/dev/null 2>&1; then
    # Linux fallback.
    ip -4 -o addr show | awk '$4 !~ /^127\./ { print $2, $4 }'
else
    echo "Neither ifconfig nor ip found — can't auto-detect. Check System Settings > Network."
fi

echo
echo "=== Hostname (for PI_CV_SERVER_HOST if mDNS resolves from the Pi) ==="
hostname

echo
echo "=== Can this machine reach the Pi? ==="
if ping -c 1 -t 3 cv-pi.local >/dev/null 2>&1; then
    echo "cv-pi.local resolves and responds to ping."
elif ping -c 1 -t 3 cv-pi >/dev/null 2>&1; then
    echo "cv-pi resolves and responds to ping."
else
    echo "cv-pi.local did NOT respond (mDNS may not be reachable from here, or the"
    echo "Pi moved networks — this project has hit that before). Find the Pi by MAC"
    echo "address in the ARP table instead:"
    echo "    arp -a | grep -i <pi-mac-address>"
fi

echo
echo "Next: put this machine's IP (or hostname, if mDNS resolves from the Pi)"
echo "into PI_CV_SERVER_HOST in ~/Desktop/work/pi_cv/.env ON THE PI."
echo "See docs/new_laptop_setup.md for the full walkthrough."
