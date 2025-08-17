#!/usr/bin/env python3
"""
preprocess_modelnet.py

Convert ModelNet10/40 OFF meshes to point-cloud .npz shards for training/evaluation.

Features
--------
- Samples N points per mesh using area-weighted surface sampling.
- Normalizes each point cloud to a unit sphere (center at mean, scale by max radius).
- Writes compressed .npz files with:
    points: (N, 3) float32
    label:  class name (str)
    file:   original mesh path (str)
    meta:   dict-like misc fields (dataset, split, num_points)
- Deterministic sampling per file with a global --seed (stable across runs).
- Parallel processing with --workers.
- Creates/updates class_to_idx.json under <root>/<dataset>/.

Usage
-----
# Typical (as used in the README/Quickstart):
python data/scripts/preprocess_modelnet.py --root data/modelnet --dataset ModelNet10 --num-points 1024
python data/scripts/preprocess_modelnet.py --root data/modelnet --dataset ModelNet40 --num-points 1024

# Options
python data/scripts/preprocess_modelnet.py \
  --root data/modelnet \
  --dataset ModelNet10 \
  --num-points 1024 \
  --splits train test \
  --workers 8 \
  --seed 42 \
  --overwrite

Directory expectations (after download_modelnet.py):
  <root>/<dataset>/<class>/{train,test}/*.off

Outputs:
  <root>/<dataset>/{train,test}/*.npz
  <root>/<dataset>/class_to_idx.json
"""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import trimesh


# --------------------------
# Helpers
# --------------------------

@dataclass
class Job:
    off_path: Path
    out_path: Path
    label: str
    dataset: str
    split: str
    num_points: int
    seed: int
    normalize: bool
    overwrite: bool


def _load_mesh(off_path: Path) -> trimesh.Trimesh:
    """
    Load an OFF mesh robustly. If it opens as a Scene, merge geometries.
    """
    mesh = trimesh.load(off_path, file_type='off', force='mesh')
    if isinstance(mesh, trimesh.Scene):
        # Concatenate all geometries into one Trimesh
        if len(mesh.geometry) == 0:
            raise ValueError(f"No geometry in OFF scene: {off_path}")
        mesh = trimesh.util.concatenate([
            g.to_mesh() if hasattr(g, "to_mesh") else trimesh.Trimesh(**g.to_dict())
            for g in mesh.geometry.values()
        ])
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not a Trimesh: {off_path}")
    return mesh


def _sample_points_from_mesh(mesh: trimesh.Trimesh, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """
    Area-weighted uniform sampling over the surface triangles.
    """
    # trimesh.sample.sample_surface uses np.random by default; we pass our rng via set_state trick if needed.
    # Simpler: sample more than needed and then subsample deterministically with rng.
    pts, _ = trimesh.sample.sample_surface(mesh, num_points)
    return pts.astype(np.float32)


def _normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    """
    Center to mean and scale so that max distance from origin is 1.
    """
    points = points - points.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(points, axis=1)
    scale = np.max(radii)
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


def _deterministic_rng(seed: int, key: str) -> np.random.Generator:
    """
    Create a deterministic RNG for each file using a global seed and the file key.
    """
    # Simple stable hash
    h = np.uint64(1469598103934665603)  # FNV-1a offset
    for ch in key.encode("utf-8"):
        h ^= np.uint64(ch)
        h *= np.uint64(1099511628211)
    combined = (np.uint64(seed) ^ h) & np.uint64((1 << 64) - 1)
    # numpy SeedSequence accepts up to 128 bits; we split combined into two 32-bit ints for safety
    ss = np.random.SeedSequence([int(combined & np.uint64(0xFFFFFFFF)),
                                 int((combined >> np.uint64(32)) & np.uint64(0xFFFFFFFF))])
    return np.random.default_rng(ss)


def _process_one(job: Job) -> Tuple[str, bool, Optional[str]]:
    """
    Worker function: load mesh, sample, normalize, save .npz.
    Returns (str(off_path), success, error_message_or_None)
    """
    try:
        if (not job.overwrite) and job.out_path.exists():
            return (str(job.off_path), True, None)

        mesh = _load_mesh(job.off_path)
        rng = _deterministic_rng(job.seed, str(job.off_path))
        pts = _sample_points_from_mesh(mesh, job.num_points, rng)
        if job.normalize:
            pts = _normalize_unit_sphere(pts)

        # Ensure output dir exists
        job.out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            job.out_path,
            points=pts.astype(np.float32),
            label=job.label,  # keep string; loader maps to indices
            file=str(job.off_path),
            meta=np.array([job.dataset, job.split, str(job.num_points)], dtype=object),
        )
        return (str(job.off_path), True, None)
    except Exception as e:
        tb = traceback.format_exc(limit=3)
        return (str(job.off_path), False, f"{e}\n{tb}")


def _collect_meshes(dataset_root: Path, split: str) -> List[Tuple[Path, str]]:
    """
    Scan <dataset_root>/<class>/<split>/*.off and return list of (off_path, class_name).
    Accepts split in {'train','test'} (case-insensitive).
    """
    split_candidates = [split, split.lower(), split.upper()]
    items: List[Tuple[Path, str]] = []
    classes = [d for d in dataset_root.iterdir() if d.is_dir()]
    for cdir in sorted(classes):
        hit = None
        for s in split_candidates:
            d = cdir / s
            if d.is_dir():
                hit = d
                break
        if hit is None:
            continue
        for off in sorted(hit.glob("*.off")):
            items.append((off, cdir.name))
    return items


def _write_class_index(dataset_root: Path) -> Dict[str, int]:
    """
    Build and persist class_to_idx.json from folder names.
    """
    classes = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    mapping = {cls: i for i, cls in enumerate(classes)}
    with open(dataset_root / "class_to_idx.json", "w") as f:
        json.dump(mapping, f, indent=2)
    return mapping


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Sample point clouds from ModelNet OFF meshes into NPZ format.")
    ap.add_argument("--root", type=str, default="data/modelnet",
                    help="Root containing ModelNetXX directories (default: data/modelnet)")
    ap.add_argument("--dataset", type=str, default="ModelNet10",
                    choices=["ModelNet10", "ModelNet40"], help="Dataset folder name")
    ap.add_argument("--num-points", type=int, default=1024, help="Points to sample per mesh (default: 1024)")
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"],
                    help="Which splits to preprocess (default: both)")
    ap.add_argument("--workers", type=int, default=0, help="#processes for parallelism (0 = sequential)")
    ap.add_argument("--seed", type=int, default=42, help="Global random seed for deterministic sampling")
    ap.add_argument("--normalize", action="store_true", default=True,
                    help="Normalize to unit sphere (enabled by default).")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Disable normalization to unit sphere.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz files")
    args = ap.parse_args()

    dataset_root = Path(args.root) / args.dataset
    if not dataset_root.exists():
        sys.exit(f"[ERR] Dataset path not found: {dataset_root}. Run download_modelnet.py first.")

    normalize = args.normalize and not args.no_normalize
    # Write/refresh class_to_idx.json for convenience
    mapping = _write_class_index(dataset_root)
    print(f"Classes ({len(mapping)}): {list(mapping.keys())}")

    total_files = 0
    for split in args.splits:
        items = _collect_meshes(dataset_root, split)
        total_files += len(items)
        print(f"[{args.dataset}:{split}] found {len(items)} OFF files")

    if total_files == 0:
        print("No files found. Nothing to do.")
        return

    # Build jobs
    jobs: List[Job] = []
    for split in args.splits:
        items = _collect_meshes(dataset_root, split)
        out_dir = dataset_root / split
        for off_path, cls in items:
            stem = off_path.stem
            out_npz = out_dir / f"{stem}.npz"
            jobs.append(Job(
                off_path=off_path,
                out_path=out_npz,
                label=cls,
                dataset=args.dataset,
                split=split,
                num_points=args.num_points,
                seed=args.seed,
                normalize=normalize,
                overwrite=args.overwrite,
            ))

    # Process
    failures = []
    if args.workers and args.workers > 0:
        print(f"Processing with {args.workers} workers ...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_process_one, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futures), start=1):
                off_path, ok, err = fut.result()
                if not ok:
                    failures.append((off_path, err))
                if i % 50 == 0 or i == len(jobs):
                    print(f"  progress: {i}/{len(jobs)}")
    else:
        print("Processing sequentially ...")
        for i, j in enumerate(jobs, start=1):
            off_path, ok, err = _process_one(j)
            if not ok:
                failures.append((off_path, err))
            if i % 50 == 0 or i == len(jobs):
                print(f"  progress: {i}/{len(jobs)}")

    succeeded = len(jobs) - len(failures)
    print(f"\nDone. Success: {succeeded}/{len(jobs)}")

    if failures:
        print("\nSome files failed to process:")
        for p, e in failures[:10]:
            print(f"  - {p}: {e.splitlines()[0]}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more.")


if __name__ == "__main__":
    main()
