#!/usr/bin/env python3
"""Compare LiteInfer Qwen3 logits with a PyTorch reference."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPOSITORY_ROOT / "models" / "TinyQwen3"
DEFAULT_BUILD_DIR = REPOSITORY_ROOT / "build"
DEFAULT_OUTPUT_PATH = DEFAULT_BUILD_DIR / "qwen3_cpp_logits.f32"
DEFAULT_TOKEN_IDS = (1, 5, 42, 7)


def resolve_repository_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPOSITORY_ROOT / path).resolve()


def find_forward_executable(build_dir: Path) -> Path | None:
    names = ("liteinfer_qwen3_forward", "liteinfer_qwen3_forward.exe")
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
        description="Compare C++ Qwen3 logits with a PyTorch reference."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Qwen3 model directory (default: models/TinyQwen3)",
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
        help="Path to liteinfer_qwen3_forward",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="C++ raw logits output path",
    )
    parser.add_argument(
        "--token-ids",
        nargs="+",
        type=int,
        default=list(DEFAULT_TOKEN_IDS),
        help="Input token IDs (default: 1 5 42 7)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-4,
        help="Relative tolerance for allclose (default: 1e-4)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-5,
        help="Absolute tolerance for allclose (default: 1e-5)",
    )
    parser.add_argument(
        "--skip-cpp-run",
        action="store_true",
        help="Compare an existing C++ output instead of running the example",
    )
    return parser.parse_args()


def load_reference_logits(model_dir: Path, token_ids: list[int]):
    try:
        import torch
        from transformers import Qwen3ForCausalLM
    except ImportError as error:
        print(
            "PyTorch/Transformers is unavailable. "
            "Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Import error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    model = Qwen3ForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=False)

    return output.logits.detach().cpu().to(torch.float32).contiguous(), torch


def read_shape(shape_path: Path) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in shape_path.read_text().split())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read C++ shape file {shape_path}: {error}") from error

    if not values or any(value < 0 for value in values):
        raise RuntimeError(f"invalid C++ output shape in {shape_path}")
    return values


def read_cpp_logits(output_path: Path, expected_shape: tuple[int, ...], torch):
    shape_path = Path(f"{output_path}.shape")
    actual_shape = read_shape(shape_path)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"shape mismatch: C++={actual_shape}, PyTorch={expected_shape}"
        )

    try:
        raw = output_path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"cannot read C++ logits {output_path}: {error}") from error

    if len(raw) % 4 != 0:
        raise RuntimeError(f"C++ logits size is not a multiple of 4: {len(raw)} bytes")

    cpp_flat = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
    expected_numel = 1
    for dimension in expected_shape:
        expected_numel *= dimension
    if cpp_flat.numel() != expected_numel:
        raise RuntimeError(
            f"logit element count mismatch: C++={cpp_flat.numel()}, "
            f"PyTorch={expected_numel}"
        )

    return cpp_flat.reshape(expected_shape)


def print_worst_difference(reference, cpp, difference) -> None:
    flat_index = int(difference.reshape(-1).argmax())
    indices = []
    for dimension in reversed(tuple(reference.shape)):
        indices.append(flat_index % int(dimension))
        flat_index //= int(dimension)
    index = tuple(reversed(indices))

    print(f"worst index: {index}")
    print(f"PyTorch value: {reference[index].item():.8e}")
    print(f"C++ value:     {cpp[index].item():.8e}")


def main() -> int:
    arguments = parse_arguments()

    if any(token_id < 0 for token_id in arguments.token_ids):
        print("token IDs must be non-negative", file=sys.stderr)
        return 2

    model_dir = resolve_repository_path(arguments.model_dir)
    build_dir = resolve_repository_path(arguments.build_dir)
    output_path = resolve_repository_path(arguments.output)

    if not (model_dir / "config.json").is_file():
        print(f"missing model config: {model_dir / 'config.json'}", file=sys.stderr)
        return 2
    if not (model_dir / "model.safetensors").is_file():
        print(f"missing model weights: {model_dir / 'model.safetensors'}", file=sys.stderr)
        return 2

    try:
        reference, torch = load_reference_logits(model_dir, arguments.token_ids)

        if not arguments.skip_cpp_run:
            executable = (
                resolve_repository_path(arguments.executable)
                if arguments.executable is not None
                else find_forward_executable(build_dir)
            )
            if executable is None or not executable.is_file():
                print(
                    "cannot find liteinfer_qwen3_forward; build it with "
                    "'cmake --build build --parallel'",
                    file=sys.stderr,
                )
                return 2

            output_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(executable),
                str(model_dir),
                str(output_path),
                *(str(token_id) for token_id in arguments.token_ids),
            ]
            print("$ " + shlex.join(command))
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)

        cpp = read_cpp_logits(output_path, tuple(reference.shape), torch)
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"comparison setup failed: {error}", file=sys.stderr)
        return 1

    difference = (cpp - reference).abs()
    max_absolute_error = difference.max().item()
    mean_absolute_error = difference.mean().item()
    max_relative_error = (
        difference / reference.abs().clamp_min(1.0e-12)
    ).max().item()

    reference_top1 = reference.argmax(dim=-1)
    cpp_top1 = cpp.argmax(dim=-1)
    top1_match = bool(torch.equal(reference_top1, cpp_top1))
    allclose = bool(
        torch.allclose(
            cpp,
            reference,
            rtol=arguments.rtol,
            atol=arguments.atol,
        )
    )

    print(f"shape: {tuple(reference.shape)}")
    print(f"max absolute error: {max_absolute_error:.8e}")
    print(f"mean absolute error: {mean_absolute_error:.8e}")
    print(f"max relative error: {max_relative_error:.8e}")
    print(f"top-1 match: {top1_match}")
    print(f"allclose: {allclose}")

    if not allclose:
        print_worst_difference(reference, cpp, difference)
        print(
            f"alignment FAILED (rtol={arguments.rtol}, atol={arguments.atol})",
            file=sys.stderr,
        )
        return 1

    print("alignment PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
