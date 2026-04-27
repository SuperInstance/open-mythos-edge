from open_mythos_edge.config import MythosConfig


def mythos_1b_edge() -> MythosConfig:
    """1B edge config. dim=2048, 16 experts, 8 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=2048,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=4096,
        max_loop_iters=8,
        prelude_layers=2,
        coda_layers=2,
        n_experts=16,
        n_shared_experts=2,
        n_experts_per_tok=2,
        expert_dim=1024,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_3b_edge() -> MythosConfig:
    """3B edge config. dim=3072, 16 experts, 8 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=3072,
        n_heads=24,
        n_kv_heads=6,
        max_seq_len=4096,
        max_loop_iters=8,
        prelude_layers=2,
        coda_layers=2,
        n_experts=16,
        n_shared_experts=2,
        n_experts_per_tok=2,
        expert_dim=1536,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )
