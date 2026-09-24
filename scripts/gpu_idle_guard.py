#!/usr/bin/env python3
"""Read-only launch preflight. Unknown occupancy is a failure, never permission."""

import csv
import io
import json
import subprocess
import sys


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=20).stdout


def gpu_capable(container):
    host = container["HostConfig"]
    # Include privileged containers and explicit device passthrough: these can
    # claim the GPU after preflight even if no CUDA context exists yet.
    if host.get("Privileged") or host.get("Runtime") == "nvidia":
        return True
    for request in host.get("DeviceRequests") or []:
        caps = {cap for group in request.get("Capabilities", []) for cap in group}
        if request.get("Driver") == "nvidia" or caps.intersection({"gpu", "compute", "utility"}):
            return True
    for device in host.get("Devices") or []:
        if any(token in str(device).lower() for token in ("nvidia", "/dev/dri", "/dev/dxg")):
            return True
    if host.get("DeviceCgroupRules"):
        return True  # Arbitrary device access cannot be safely classified as idle.
    for env in container.get("Config", {}).get("Env") or []:
        if env.startswith("NVIDIA_VISIBLE_DEVICES="):
            if env.partition("=")[2].lower() not in ("", "none", "void"):
                return True
    return False


def assert_idle(port):
    processes = command("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits")
    for row in csv.reader(io.StringIO(processes)):
        if row:
            raise RuntimeError(f"GPU process or unrecognized occupancy: {','.join(row)}")
    ids = command("docker", "ps", "--quiet").split()
    if ids:
        containers = json.loads(command("docker", "inspect", *ids))
        if len(containers) != len(ids):
            raise RuntimeError("Incomplete Docker occupancy inspection")
        busy = [c["Name"] for c in containers if gpu_capable(c)]
        if busy:
            raise RuntimeError(f"GPU-capable containers still running: {', '.join(busy)}")
    listeners = command("ss", "-H", "-ltn", f"sport = :{port}")
    if listeners.strip():
        raise RuntimeError(f"Port {port} already listening")


def main():
    try:
        assert_idle(int(sys.argv[1]))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"FATAL: GPU/port preflight failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
