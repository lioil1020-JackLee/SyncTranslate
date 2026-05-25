from __future__ import annotations

import argparse


DEFAULT_REPO_ID = "Systran/faster-whisper-large-v3-turbo"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a faster-whisper CTranslate2 model snapshot.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", default=r"runtimes\models\asr\large-v3-turbo")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface-hub is required to download ASR models. Run: uv sync --group build") from exc
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"ASR model downloaded: {args.repo_id} -> {args.local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

