#!/usr/bin/env python3
"""
download_modelnet.py

Download and extract ModelNet10 / ModelNet40 OFF datasets.

Usage:
  python data/scripts/download_modelnet.py --root data/modelnet --datasets ModelNet10 ModelNet40
  python data/scripts/download_modelnet.py --root ./data --datasets ModelNet10 --force

Notes:
- Default sources are the official Princeton URLs:
    ModelNet10:  http://modelnet.cs.princeton.edu/ModelNet10.zip
    ModelNet40:  http://modelnet.cs.princeton.edu/ModelNet40.zip
- The script:
    1) Downloads the ZIP to <root>/<name>.zip (with a progress bar)
    2) Verifies the ZIP
    3) Extracts to <root>/
    4) Normalizes folder case to exactly "ModelNet10" or "ModelNet40"
"""

import argparse
import contextlib
import os
import sys
import time
import zipfile
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Optional

# Official URLs
URLS: Dict[str, str] = {
    "ModelNet10": "http://modelnet.cs.princeton.edu/ModelNet10.zip",
    "ModelNet40": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
}


def human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    ix = 0
    while n >= 1024 and ix < len(units) - 1:
        n /= 1024.0
        ix += 1
    return f"{n:.2f} {units[ix]}"


def progress_hook_factory(start_time: float):
    last_print = {"t": 0}  # throttle prints

    def hook(block_count: int, block_size: int, total_size: int):
        if total_size <= 0:
            # Unknown total size
            dl = block_count * block_size
            now = time.time()
            if now - last_print["t"] > 0.2:
                print(f"\r  Downloaded {human_bytes(dl)}", end="", flush=True)
                last_print["t"] = now
            return

        downloaded = min(block_count * block_size, total_size)
        pct = downloaded / total_size * 100.0
        speed = downloaded / max(1e-9, (time.time() - start_time))
        bar_len = 30
        filled = int(pct / 100.0 * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = (
            f"\r  [{bar}] {pct:6.2f}%  "
            f"{human_bytes(downloaded)}/{human_bytes(total_size)}  "
            f"@ {human_bytes(speed)}/s"
        )
        # Throttle a bit to avoid flicker
        now = time.time()
        if now - last_print["t"] > 0.05 or downloaded == total_size:
            print(msg, end="", flush=True)
            last_print["t"] = now

    return hook


def download(url: str, dst_zip: Path, overwrite: bool = False, timeout: int = 30) -> None:
    if dst_zip.exists() and not overwrite:
        print(f"• Found existing ZIP: {dst_zip} (use --force to re-download)")
        return

    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"• Downloading from: {url}")
    print(f"• Saving to       : {dst_zip}")

    start = time.time()
    try:
        with contextlib.ExitStack() as _:
            urllib.request.urlretrieve(
                url,
                filename=dst_zip,
                reporthook=progress_hook_factory(start),
                data=None,
            )
        print()  # newline after progress bar
    except Exception as e:
        if dst_zip.exists():
            # Remove incomplete file
            with contextlib.suppress(Exception):
                dst_zip.unlink()
        raise RuntimeError(f"Download failed: {e}") from e


def verify_zip(zip_path: Path) -> None:
    print(f"• Verifying ZIP: {zip_path}")
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"Corrupted member: {bad}")
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"ZIP verification failed: {e}") from e
    print("  ✓ ZIP OK")


def extract_zip(zip_path: Path, root: Path, expected_name: str) -> Path:
    print(f"• Extracting to: {root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    # Normalize to exact expected folder name (handles lowercase archives)
    extracted_dir_upper = root / expected_name
    extracted_dir_lower = root / expected_name.lower()

    if extracted_dir_upper.exists():
        final_dir = extracted_dir_upper
    elif extracted_dir_lower.exists():
        # Move/rename lower to proper case
        if extracted_dir_upper.exists():
            shutil.rmtree(extracted_dir_upper)
        shutil.move(str(extracted_dir_lower), str(extracted_dir_upper))
        final_dir = extracted_dir_upper
    else:
        # Last attempt: try to find one top-level folder to rename
        candidates = [p for p in root.iterdir() if p.is_dir() and expected_name.lower() in p.name.lower()]
        if candidates:
            if extracted_dir_upper.exists():
                shutil.rmtree(extracted_dir_upper)
            shutil.move(str(candidates[0]), str(extracted_dir_upper))
            final_dir = extracted_dir_upper
        else:
            raise RuntimeError(f"Could not locate extracted directory for {expected_name}.")

    print(f"  ✓ Extracted folder: {final_dir}")
    return final_dir


def main():
    parser = argparse.ArgumentParser(description="Download and extract ModelNet10/40 OFF datasets.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/modelnet",
        help="Root directory to place ZIPs and extracted folders (default: data/modelnet)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ModelNet10", "ModelNet40"],
        default=["ModelNet10"],
        help="Which datasets to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if ZIP already exists.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction step (only download).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Network timeout (seconds) for HTTP requests.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # Set global timeout for urllib
    socket_timeout_prev: Optional[float] = None
    try:
        import socket
        socket_timeout_prev = socket.getdefaulttimeout()
        socket.setdefaulttimeout(float(args.timeout))
    except Exception:
        pass

    try:
        for name in args.datasets:
            url = URLS[name]
            zip_path = root / f"{name}.zip"

            print(f"\n=== {name} ===")
            download(url, zip_path, overwrite=args.force, timeout=args.timeout)
            verify_zip(zip_path)

            if not args.skip_extract:
                extract_zip(zip_path, root, expected_name=name)

        print("\nAll done.")
    finally:
        # restore previous timeout
        try:
            if socket_timeout_prev is not None:
                import socket
                socket.setdefaulttimeout(socket_timeout_prev)
        except Exception:
            pass


if __name__ == "__main__":
    main()
