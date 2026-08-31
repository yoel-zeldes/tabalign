#!/usr/bin/env python3
"""Download experiment results from the Modal volume to local results/ directory.

Downloads the joblib cache and any generated plots, skipping model weights
and OpenML data caches.
"""

import argparse
import os
import subprocess
import sys


from consts import MODAL_VOLUME_NAME as VOLUME_NAME


def run_modal_cmd(args):
    """Run a modal CLI command and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "modal", *args],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def list_volume_root():
    """List top-level non-hidden directory names in the volume."""
    code, stdout, stderr = run_modal_cmd(["volume", "ls", VOLUME_NAME])
    if code != 0:
        print(f"Error listing volume: {stderr}", file=sys.stderr)
        sys.exit(1)
    dirs = []
    for line in stdout.strip().splitlines():
        name = line.strip().split()[-1] if line.strip() else ""
        # "Directory" appears in the `modal volume ls` header line
        if name and not name.startswith(".") and name != "Directory":
            dirs.append(name.rstrip("/"))
    return dirs


def download(remote_dir, local_dir):
    """Download a directory from the Modal volume."""
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading {remote_dir}/ → {local_dir}/")
    code, stdout, stderr = run_modal_cmd(
        ["volume", "get", VOLUME_NAME, remote_dir, "--output", local_dir, "--force"]
    )
    if code != 0:
        print(f"  Warning: {stderr.strip()}")
    else:
        print(f"  Done.")


def main():
    from cache_utils import OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Download results from Modal volume to local results/ directory"
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help=f"Local directory to download results into (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dir", nargs="*", default=None,
        help=(
            "Specific subdirectories to download from the volume. "
            "Default: all non-hidden directories (skips .model_cache, .openml_cache)."
        ),
    )
    args = parser.parse_args()

    if args.dir:
        dirs_to_download = args.dir
    else:
        dirs_to_download = list_volume_root()

    if not dirs_to_download:
        print("No directories to download.")
        return

    print(f"\nDownloading: {', '.join(dirs_to_download)}")
    for dir_name in dirs_to_download:
        download(dir_name, os.path.join(args.output, dir_name))

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
