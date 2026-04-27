# OpenMythos Edge

Edge-optimized Recurrent-Depth Transformer for ARM64/Jetson devices. Pure PyTorch, no Triton kernels, no CUDA dependencies.

## Features
- **1B and 3B variants** designed for Jetson Orin (8GB VRAM)
- **Pure PyTorch** — no custom kernels, works on ARM64
- **Memory estimation** — `config.estimate_memory()` before deployment
- **GQA attention** — standard Grouped Query Attention (MLA optional)
- **Adaptive compute** — ACT halting with configurable threshold
- **MoE sparse experts** — 16 routed experts, top-K routing

## Usage
```python
from open_mythos_edge import OpenMythosEdge, mythos_1b_edge
config = mythos_1b_edge()  # ~1.6 GB
model = OpenMythosEdge(config)
```
