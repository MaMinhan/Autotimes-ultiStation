#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download HuggingFace models to data disk.

Already downloaded models will be skipped automatically.

Models:
- pythia-1b
- Qwen2.5-1.5B

NOTE:
- gpt2 / opt-1.3b are assumed to be already downloaded
- Models will be stored in /root/autodl-tmp/hf_models
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError


# =========================
# CONFIG
# =========================
MODEL_ROOT = Path("/root/autodl-tmp/hf_models")

MODELS = [
    ("pythia-1b", "EleutherAI/pythia-1b"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B"),
]


def human_size(path: Path) -> str:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024:
            return f"{total:.1f}{unit}"
        total /= 1024
    return f"{total:.1f}PB"


def main():
    print(f"📦 Model root: {MODEL_ROOT}")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    if token:
        print("✅ HF token detected")
    else:
        print("⚠️  HF token not found (Qwen may be slower or fail)")

    for name, repo in MODELS:
        print("\n" + "=" * 80)
        print(f"▶ Processing: {name}")

        out_dir = MODEL_ROOT / name

        # Skip if already exists
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"✅ Already exists, skip: {out_dir}")
            print(f"   Size: {human_size(out_dir)}")
            continue

        try:
            print(f"⬇️  Downloading {repo} ...")
            snapshot_download(
                repo_id=repo,
                local_dir=str(out_dir),
                token=token,
                resume_download=True,
                max_workers=1,
            )
            print(f"✅ Done: {name}")
            print(f"   Size: {human_size(out_dir)}")

        except HfHubHTTPError as e:
            print(f"❌ HTTP error while downloading {repo}")
            print(e)
            return 2

        except Exception as e:
            print(f"❌ Failed downloading {repo}")
            print(e)
            return 3

    print("\n🎉 All models processed successfully!")
    print("📂 Models location:")
    print(f"   {MODEL_ROOT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
