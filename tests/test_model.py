"""Real tests for open_mythos_edge.

These give the CI job something to actually pass or fail on (previously the
test step used `... || true`, which swallowed every failure and reported green
unconditionally).

Coverage:
  * `mythos_1b_edge()` / `mythos_3b_edge()` build valid `MythosConfig` objects.
  * `config.estimate_memory()` returns a plausible positive GB figure.
  * An `OpenMythosEdge` model actually assembles and runs an end-to-end forward
    pass with a sane output shape.

The forward-pass test uses a deliberately tiny custom `MythosConfig` (few
layers, small hidden dim, few experts/loops). This is an architecture smoke
test proving the components wire together and run — it is NOT a performance or
quality benchmark. The full 1B/3B configs are too heavy to instantiate in CI
quickly, so only their *construction* (config objects) is exercised there.
"""

import torch

from open_mythos_edge import MythosConfig, OpenMythosEdge, mythos_1b_edge, mythos_3b_edge


def test_mythos_1b_edge_constructs_valid_config():
    cfg = mythos_1b_edge()
    assert isinstance(cfg, MythosConfig)
    assert cfg.dim == 2048
    assert cfg.n_heads == 16
    assert cfg.n_kv_heads == 4
    assert cfg.dim % cfg.n_heads == 0  # head_dim must be an integer
    assert cfg.n_heads % cfg.n_kv_heads == 0  # GQA groups must divide evenly
    assert cfg.n_experts >= cfg.n_experts_per_tok


def test_mythos_3b_edge_constructs_valid_config():
    cfg = mythos_3b_edge()
    assert isinstance(cfg, MythosConfig)
    assert cfg.dim == 3072
    assert cfg.n_heads == 24
    assert cfg.n_kv_heads == 6
    assert cfg.dim % cfg.n_heads == 0
    assert cfg.n_heads % cfg.n_kv_heads == 0
    assert cfg.n_experts >= cfg.n_experts_per_tok
    assert cfg.dim > mythos_1b_edge().dim  # 3B must be larger than 1B


def test_estimate_memory_1b_is_plausible():
    cfg = mythos_1b_edge()
    mem_gb = cfg.estimate_memory()
    assert isinstance(mem_gb, float)
    assert mem_gb > 0.0
    # README claims ~1.6 GB for the 1B variant; sanity-check a loose band rather
    # than an exact number (the estimate covers weights + KV cache at fp32).
    assert 0.5 < mem_gb < 5.0, f"1B memory estimate {mem_gb} GB outside plausible band"


def test_estimate_memory_counts_tied_embedding_once():
    # Regression test: OpenMythosEdge ties the LM head to the embedding
    # (`self.head.weight = self.embed.weight`), so estimate_memory() must count
    # the embedding table exactly ONCE, not twice.
    #
    # With batch_size=0 the KV-cache term vanishes, so the estimate is purely
    # weights. Holding every other field fixed and varying only vocab_size, the
    # delta must equal (vocab_delta * dim * bytes_per_param) — counted once.
    common = dict(
        dim=128,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=16,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=64,
        lora_rank=4,
    )
    small = MythosConfig(vocab_size=1000, **common)
    big = MythosConfig(vocab_size=2000, **common)
    delta_gb = big.estimate_memory(batch_size=0) - small.estimate_memory(batch_size=0)
    # Tied head: embedding counted once -> delta is one table, not two.
    once_bytes = (big.vocab_size - small.vocab_size) * small.dim * 4
    assert abs(delta_gb - once_bytes / (1024**3)) < 1e-12
    assert delta_gb * (1024**3) != 2 * once_bytes  # would be true if double-counted


def test_estimate_memory_3b_is_plausible_and_larger():
    cfg_1b = mythos_1b_edge()
    cfg_3b = mythos_3b_edge()
    mem_1b = cfg_1b.estimate_memory()
    mem_3b = cfg_3b.estimate_memory()
    assert isinstance(mem_3b, float)
    assert mem_3b > 0.0
    assert 1.0 < mem_3b < 10.0, f"3B memory estimate {mem_3b} GB outside plausible band"
    assert mem_3b > mem_1b  # bigger model must estimate more memory


def _tiny_config() -> MythosConfig:
    # dim must be divisible by 16: dim // 8 (loop_dim) must be even for the loop
    # index embedding, and dim // n_heads (head_dim) must be even for RoPE.
    return MythosConfig(
        vocab_size=64,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=2,
        prelude_layers=1,
        coda_layers=1,
        n_experts=2,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=32,
        lora_rank=4,
        dropout=0.0,
    )


def test_model_forward_pass_shape():
    # The single most valuable test: proves the architecture actually assembles
    # and runs end-to-end, not just that the classes exist.
    cfg = _tiny_config()
    model = OpenMythosEdge(cfg)
    model.eval()

    batch, seq = 1, 8
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq))
    with torch.no_grad():
        out = model(input_ids)

    assert out.shape == (batch, seq, cfg.vocab_size)
    assert torch.isfinite(out).all(), "forward pass produced non-finite values"


def test_model_forward_pass_respects_n_loops():
    # ACT halting respects a user-supplied n_loops override (fewer loops still
    # produces a valid, finite output of the correct shape).
    cfg = _tiny_config()
    model = OpenMythosEdge(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        out = model(input_ids, n_loops=1)

    assert out.shape == (1, 6, cfg.vocab_size)
    assert torch.isfinite(out).all()


def test_model_is_torch_module_and_has_expected_components():
    cfg = _tiny_config()
    model = OpenMythosEdge(cfg)
    assert isinstance(model, torch.nn.Module)
    # Core structural claims from the README: GQA attention, MoE experts,
    # recurrent-depth block, tied embeddings.
    assert hasattr(model, "embed")
    assert hasattr(model, "prelude")
    assert hasattr(model, "recurrent")
    assert hasattr(model, "coda")
    assert hasattr(model, "head")
    # Tied embeddings: head shares the embedding weight tensor.
    assert model.head.weight is model.embed.weight
