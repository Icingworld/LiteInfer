#!/usr/bin/env python3
"""Compare the LiteInfer Qwen3 tokenizer with the local Hugging Face tokenizer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPOSITORY_ROOT / "models" / "Qwen3-0.6B-Base"
DEFAULT_BUILD_DIR = REPOSITORY_ROOT / "build"
DEFAULT_CASES = (
    "hello",
    "Hello world!",
    "你好，世界",
    "can't stop",
    "🙂",
    "<|im_start|>",
    "abc<|im_start|>def",
    "line1\nline2",
    "",
    " ",
    "  ",
    "hello,world",
    "Hello's",
    "I'm",
    "we'll",
    "foo_bar",
    "12345",
    "!!!",
    " hello",
    "\tfoo",
    "😀👍🏽",
    "<|endoftext|>",
    "<tool_call>",
    "中文 mixed 123",
)


def resolve_repository_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPOSITORY_ROOT / path).resolve()


def find_tokenizer_executable(build_dir: Path) -> Path | None:
    names = ("liteinfer_qwen3_tokenizer", "liteinfer_qwen3_tokenizer.exe")
    candidates = []
    for name in names:
        candidates.extend(
            (
                build_dir / "examples" / name,
                build_dir / "examples" / "Debug" / name,
                build_dir / "Debug" / "examples" / name,
                build_dir / name,
            )
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LiteInfer Qwen3 token IDs with Hugging Face."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Local Qwen3 model directory (default: models/Qwen3-0.6B-Base)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help="CMake build directory (default: build)",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        help="Path to liteinfer_qwen3_tokenizer",
    )
    parser.add_argument(
        "--text",
        action="append",
        help="Compare one custom text; repeat this option for multiple cases",
    )
    return parser.parse_args()


def parse_cpp_ids(stdout: str) -> list[int]:
    for line in stdout.splitlines():
        if line.startswith("ids:"):
            fields = line[len("ids:") :].split()
            return [int(field) for field in fields]
    raise RuntimeError(f"C++ tokenizer output does not contain an ids line:\n{stdout}")


def compare_case(executable: Path, model_dir: Path, text: str, expected: list[int]) -> None:
    completed = subprocess.run(
        [str(executable), str(model_dir), text],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"C++ tokenizer failed for {text!r}:\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )

    actual = parse_cpp_ids(completed.stdout)
    if actual != expected:
        raise RuntimeError(
            "token ID mismatch for "
            f"{text!r}:\n"
            f"expected={json.dumps(expected)}\n"
            f"actual={json.dumps(actual)}\n"
            f"stdout={completed.stdout}"
        )

    print(f"PASS {text!r}: {actual}")


def main() -> int:
    arguments = parse_arguments()
    model_dir = resolve_repository_path(arguments.model_dir)
    build_dir = resolve_repository_path(arguments.build_dir)
    executable = (
        resolve_repository_path(arguments.executable)
        if arguments.executable is not None
        else find_tokenizer_executable(build_dir)
    )

    if executable is None or not executable.is_file():
        print("Cannot find liteinfer_qwen3_tokenizer. Build the project first.", file=sys.stderr)
        return 2
    if not model_dir.is_dir():
        print(f"Model directory does not exist: {model_dir}", file=sys.stderr)
        return 2

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        print(
            "Transformers is unavailable. Install the project requirements first.",
            file=sys.stderr,
        )
        print(f"Import error: {error}", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    cases = tuple(arguments.text) if arguments.text else DEFAULT_CASES

    for text in cases:
        expected = tokenizer.encode(text, add_special_tokens=False)
        compare_case(executable, model_dir, text, expected)

    print(f"All {len(cases)} tokenizer cases match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
