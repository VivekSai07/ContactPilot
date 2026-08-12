#!/usr/bin/env python3
"""Download the GraspGen Franka-Panda checkpoint from Hugging Face Hub.

Run this once (idempotent) to populate graspgen_checkpoints/, which is not
committed to git (the generator checkpoint alone is ~900 MB).
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "adithyamurali/GraspGenModels"
DEST_DIR = Path(__file__).resolve().parent.parent / "graspgen_checkpoints"

FILES = [
    "checkpoints/graspgen_franka_panda.yml",
    "checkpoints/graspgen_franka_panda_gen.pth",
    "checkpoints/graspgen_franka_panda_dis.pth",
]


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for repo_path in FILES:
        dest = DEST_DIR / Path(repo_path).name
        if dest.exists():
            print(f"Already present: {dest}")
            continue
        downloaded = hf_hub_download(repo_id=REPO_ID, filename=repo_path)
        dest.write_bytes(Path(downloaded).read_bytes())
        print(f"Downloaded {repo_path} -> {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
