"""Check a complete downloaded dataset against its SHA-256 file manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> int:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "trajectory-release-files-v1":
        raise ValueError("Unsupported dataset manifest")
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError("Manifest contains a path outside the dataset")
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing file or size mismatch: {relative}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {relative}")
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data/published"))
    args = parser.parse_args()
    print(f"Verified {verify(args.data_dir):,} files.")


if __name__ == "__main__":
    main()
