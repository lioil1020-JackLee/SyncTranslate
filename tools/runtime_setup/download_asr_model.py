from __future__ import annotations

import argparse

from app.bootstrap.runtime_assets import DEFAULT_ASR_MODEL_REPO, asr_model_repo_candidates

DEFAULT_REPO_ID = DEFAULT_ASR_MODEL_REPO


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
    failures: list[str] = []
    for repo_id in asr_model_repo_candidates(args.repo_id):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=args.local_dir,
                local_dir_use_symlinks=False,
            )
            print(f"ASR model downloaded: {repo_id} -> {args.local_dir}")
            return 0
        except Exception as exc:
            failures.append(f"{repo_id}: {type(exc).__name__}: {exc}")
            print(f"ASR model download failed from {repo_id}: {exc}")
    details = "\n".join(f"  - {failure}" for failure in failures)
    raise RuntimeError(
        "Unable to download faster-whisper ASR model. "
        "Check network access, HF_TOKEN for private/gated repos, or pass -AsrModelRepo to "
        "tools/runtime_setup/prepare_external_runtimes.ps1.\n"
        f"Tried:\n{details}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
