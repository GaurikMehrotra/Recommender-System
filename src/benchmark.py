"""Measure elapsed time and Linux child-process peak RSS for a real workflow command."""

import argparse
import resource
import subprocess
import sys
import time

from src.utils import atomic_json


def main() -> None:
    """Preserve the wrapped exit status; RUSAGE_CHILDREN measures process peak, not a container."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/resources.json")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a command after --")
    if sys.platform != "linux":
        parser.error("This RSS measurement targets Linux; use WSL, Kaggle, or the cloud")
    started = time.perf_counter()
    result = subprocess.run(command, check=False)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    atomic_json(
        args.output,
        {
            "command": command,
            "exit_code": result.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "max_rss_mib": usage.ru_maxrss / 1024,
            "cpu_user_seconds": usage.ru_utime,
            "cpu_system_seconds": usage.ru_stime,
        },
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
