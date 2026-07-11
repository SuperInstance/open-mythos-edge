# OpenMythos Edge

Edge-optimized Recurrent-Depth Transformer for ARM64/Jetson devices. Pure PyTorch, no Triton kernels, no CUDA dependencies.

## Features
- **1B and 3B variants** designed for Jetson Orin (8GB VRAM)
- **Pure PyTorch** — no custom kernels, works on ARM64
- **Memory estimation** — `config.estimate_memory()` before deployment
- **GQA attention** — standard Grouped Query Attention (MLA optional)
- **Adaptive compute** — ACT halting with configurable threshold
- **MoE sparse experts** — 16 routed experts, top-K routing

## Installation

Published on PyPI:

```bash
pip install open-mythos-edge
```

The only runtime dependency is `torch>=2.0.0` (see `pyproject.toml`).

## Usage
```python
from open_mythos_edge import OpenMythosEdge, mythos_1b_edge
config = mythos_1b_edge()  # ~1.6 GB
model = OpenMythosEdge(config)
```

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

`tests/test_model.py` exercises config construction, memory estimation, and a
small forward pass (using a deliberately tiny config as an architecture smoke
test, not a quality benchmark). CI runs the same command on push/PR.

## Related repos

Part of the SuperInstance / Cocapn edge stack. Conceptually related siblings:

- **[marine-gpu-edge](https://github.com/SuperInstance/marine-gpu-edge)** —
  Jetson GPU compute target; this is exactly the model class it would run on
  Orin hardware.
- **[nexus-edge-runtime](https://github.com/SuperInstance/nexus-edge-runtime)**
  — edge bytecode runtime that could host inference workloads like this one.
- **[Edge-Native](https://github.com/SuperInstance/Edge-Native)** — the
  Jetson-side bytecode/firmware layer for the same device family.
