# LiteInfer

基于现代 C++23 的轻量级 Qwen3 CPU 推理框架，面向大模型推理架构、底层计算和性能优化学习。

当前版本已完成从模型文件读取、Tensor 计算、Transformer Decoder 到文本生成的端到端闭环，可加载真实的 Qwen3-0.6B 权重并执行 greedy generation。

> 当前项目处于 M1 CPU 功能基线阶段，重点是模块边界和计算正确性；性能优化将逐步引入。

## 特性

- C++23 实现的连续、row-major CPU Tensor。
- Linear、RMSNorm、RoPE、GQA causal attention 和 SwiGLU。
- Qwen3 Decoder Layer、模型配置解析和权重加载。
- 只读 Safetensors 解析，以及 Windows/POSIX 文件系统抽象。
- 基于 CTest 的模块测试和 PyTorch 数值对齐脚本。
- 支持真实 Qwen3-0.6B BF16 权重，并在当前加载流程中转换为 Float32。

## 项目状态

已支持：

- 真实 Qwen3 模型的 full-sequence forward。
- 单次文本输入和 greedy 文本生成。
- 15 个 CTest 测试目标。

当前限制：

- CPU 标量计算内核，尚未接入优化 GEMM、SIMD 或线程并行。
- 文本生成尚未使用 KV Cache。
- 当前权重加载流程面向单个 `model.safetensors` 文件。
- 暂不包含量化、sampling 和 chat template。

## 构建

要求：CMake 4.2.3+、支持 C++23 的编译器，以及可选的 Python 3 环境。

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Debug 构建：

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

## 运行

准备一个本地 Qwen3 模型目录，例如：

```text
models/Qwen3-0.6B-Base/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Windows PowerShell 示例：

```powershell
.\build\release\examples\liteinfer_qwen3_text_generation.exe models\Qwen3-0.6B-Base 8 "Hello"
```

运行结果为模型生成的新文本。也可以使用 `liteinfer_qwen3_forward` 输出完整 logits：

```powershell
.\build\release\examples\liteinfer_qwen3_forward.exe models\Qwen3-0.6B-Base build\qwen3_logits.f32 1 5 42 7
```

## 数值对齐

生成 TinyQwen3 测试模型并与 PyTorch 结果对比：

```bash
python -m pip install -r requirements.txt
python scripts/generate_tiny_qwen3.py
python scripts/compare_qwen3.py --model-dir models/TinyQwen3 --build-dir build/release
```

## 后续计划

1. 增加 `prefill/decode` 接口和 KV Cache。
2. 引入 workspace 复用、无拷贝 Tensor view 和 RoPE 缓存。
3. 优化 GEMM，加入 tiling、SIMD 和线程并行。
4. 支持权重分片、BF16/量化计算和 streaming generation。

## 目录结构

```text
src/core/       核心 Tensor、算子、Layer、模型和运行时
examples/       Qwen3 前向与文本生成示例
tests/          CTest 测试
scripts/        TinyQwen3 生成和 PyTorch 对齐工具
```
