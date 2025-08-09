#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path
import zipfile
import tarfile
import pandas as pd

"""
Chatgpt made this script in an effort to save time after deleting mine on accident
**OUTPUT FOLDER NEEDS FIXED**
"""

KAGGLE_DATASET = "chrisfilo/urbansound8k"  # API-accessible mirror

# ------------------ archive extraction & normalization ------------------

def _extract_all_archives(data_dir: Path, max_passes: int = 4) -> None:
    for _ in range(max_passes):
        changed = False
        # extract zips
        for z in list(data_dir.glob("*.zip")):
            try:
                with zipfile.ZipFile(z) as zf:
                    print(f"Extracting {z} ...")
                    zf.extractall(path=data_dir)
                try: z.unlink()
                except: pass
                changed = True
            except zipfile.BadZipFile:
                pass

        # extract tarballs
        for pat in ["*.tar.gz", "*.tgz", "*.tar.xz", "*.txz", "*.tar.bz2", "*.tbz2", "*.tar"]:
            for tg in list(data_dir.glob(pat)):
                try:
                    print(f"Extracting {tg} ...")
                    with tarfile.open(tg, "r:*") as tf:
                        tf.extractall(path=data_dir)
                    try: tg.unlink()
                    except: pass
                    changed = True
                except tarfile.TarError:
                    pass

        if not changed:
            break

def _normalize_urbansound_layout(data_dir: Path) -> Path | None:
    """
    If Kaggle dumped a flat layout (UrbanSound8K.csv + fold1..fold10 at top level),
    move into UrbanSound8K/{metadata,audio}.
    """
    flat_csv = data_dir / "UrbanSound8K.csv"
    flat_folds = [p for p in data_dir.glob("fold*") if p.is_dir()]
    if flat_csv.exists() and flat_folds:
        dataset_root = data_dir / "UrbanSound8K"
        meta_dir = dataset_root / "metadata"
        audio_dir = dataset_root / "audio"
        meta_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        flat_csv.rename(meta_dir / "UrbanSound8K.csv")
        for fd in flat_folds:
            fd.rename(audio_dir / fd.name)

        print(f"Normalized flat layout into: {dataset_root}")
        return dataset_root
    return None

def ensure_kaggle_dataset(data_dir: Path) -> Path:
    dataset_root = data_dir / "UrbanSound8K"
    meta_csv = dataset_root / "metadata" / "UrbanSound8K.csv"
    if meta_csv.exists():
        return dataset_root

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError("Install Kaggle: `pip install kaggle`") from e

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        raise RuntimeError("Kaggle auth failed. Put `kaggle.json` in ~/.kaggle and `chmod 600` it.") from e

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} to {data_dir} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(data_dir), quiet=False, unzip=False)

    _extract_all_archives(data_dir, max_passes=4)

    # Normalize flat dump if present
    normalized = _normalize_urbansound_layout(data_dir)
    if normalized:
        return normalized

    # Detect standard layout
    candidates = []
    for p in data_dir.rglob("UrbanSound8K.csv"):
        if p.parent.name == "metadata":
            candidates.append(p.parent.parent)

    if not candidates:
        print("Contents of data dir (debug):")
        for x in sorted(data_dir.iterdir()):
            print(" -", x)
        raise RuntimeError("Could not locate UrbanSound8K after extraction.")

    candidates.sort(key=lambda x: len(str(x)))
    root = candidates[0]
    print(f"Detected dataset root: {root}")
    return root

# ------------------ sampling / copying / cleanup ------------------

def sample_non_dog(meta_csv: Path, n_unknown: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    # exclude dog_bark
    pool = df[df["class"] != "dog_bark"].copy()
    classes = sorted(pool["class"].unique())
    if not classes:
        raise RuntimeError("No non-dog_bark classes found.")

    # even distribution by class
    per_class = max(1, n_unknown // len(classes))
    sampled = (
        pool.groupby("class", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), per_class), random_state=seed))
        .reset_index(drop=True)
    )

    # top-up (if some classes are too small)
    if len(sampled) < n_unknown:
        remaining = pool.drop(sampled.index, errors="ignore")
        extra = remaining.sample(min(len(remaining), n_unknown - len(sampled)), random_state=seed)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    # trim if we overshot (paranoia)
    if len(sampled) > n_unknown:
        sampled = sampled.sample(n_unknown, random_state=seed).reset_index(drop=True)

    return sampled

def copy_to_unknown(sample_df: pd.DataFrame, dataset_root: Path) -> int:
    audio_root = dataset_root / "audio"
    out_dir = dataset_root / "unknown"
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, row in sample_df.iterrows():
        fold = f"fold{int(row['fold'])}"
        src = audio_root / fold / row["slice_file_name"]
        dst = out_dir / row["slice_file_name"]
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            print(f"Failed to copy {src} -> {dst}: {e}", file=sys.stderr)
    return copied

def nuke_everything_except_unknown(dataset_root: Path) -> None:
    """
    Remove all dataset content except the 'unknown/' folder.
    Leaves: <dataset_root>/unknown with your 600 files.
    """
    unknown = dataset_root / "unknown"
    for item in dataset_root.iterdir():
        if item == unknown:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            try: item.unlink()
            except: pass

# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser(description="Sample N non-dog UrbanSound8K clips into <dataset>/unknown and delete the rest.")
    ap.add_argument("--data-dir", default="./data", help="Download/extract location (default: ./data)")
    ap.add_argument("--n", type=int, default=600, help="Number of unknown samples (default: 600)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--delete-after-copy", action="store_true", help="Delete all original dataset files after copying")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    dataset_root = ensure_kaggle_dataset(data_dir)

    meta_csv = dataset_root / "metadata" / "UrbanSound8K.csv"
    audio_root = dataset_root / "audio"
    if not meta_csv.exists() or not audio_root.exists():
        raise RuntimeError(f"Incomplete dataset. Expected {meta_csv} and {audio_root}")

    sampled = sample_non_dog(meta_csv, args.n, args.seed)
    print(f"Sampling {len(sampled)} non-dog files into '{dataset_root / 'unknown'}' ...")
    copied = copy_to_unknown(sampled, dataset_root)
    print(f"✅ Copied {copied} files.")

    # sanity check before deletion
    if args.delete_after_copy:
        # require at least 90% of target to avoid accidental nukes
        if copied < max(10, int(args.n * 0.9)):
            print("Refusing to delete dataset: too few files copied vs requested.", file=sys.stderr)
            sys.exit(2)
        nuke_everything_except_unknown(dataset_root)
        print(f"✅ Cleaned dataset; only '{dataset_root / 'unknown'}' remains.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
