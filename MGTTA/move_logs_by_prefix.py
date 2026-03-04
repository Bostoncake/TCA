#!/usr/bin/env python3
from pathlib import Path
import shutil

OLD_PREFIX = "foa_bp_2025-10-31_13"
NEW_PREFIX = "foa_bp_2025-10-30_21"

# base = Path(".").resolve()
base = Path("/home/xiongyizhe/CVPR2026/MGTTA/outputs")

def unique_dest(dest: Path) -> Path:
    """Return a non-colliding destination path by appending _1, _2, ... if needed."""
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    parent = dest.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

for old_dir in base.iterdir():
    if old_dir.is_dir() and old_dir.name.startswith(OLD_PREFIX):
        new_name = NEW_PREFIX + old_dir.name[len(OLD_PREFIX):]
        new_dir = base / new_name
        new_dir.mkdir(parents=True, exist_ok=True)

        # Move everything inside old_dir to new_dir
        for item in old_dir.iterdir():
            target = new_dir / item.name
            target = unique_dest(target)
            shutil.move(str(item), str(target))

        # Optional: remove the old (now empty) directory
        try:
            old_dir.rmdir()
        except OSError:
            # Not empty (e.g., hidden files left) — skip removal
            pass

        print(f"Moved contents: {old_dir.name} -> {new_dir.name}")
