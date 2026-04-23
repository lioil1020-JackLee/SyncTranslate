from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "flateon/Belle-whisper-large-v3-turbo-zh-ct2"
DEFAULT_LOCAL_DIR = Path("runtimes/models/belle-zh-ct2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(args.repo_id).strip(),
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Belle model ready: {local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
