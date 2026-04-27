from dataclasses import dataclass


@dataclass
class MythosConfig:
    """Edge-optimized configuration for OpenMythos (1B/3B scale)."""

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 4096
    max_loop_iters: int = 8
    prelude_layers: int = 2
    coda_layers: int = 2
    n_experts: int = 16
    n_shared_experts: int = 2
    n_experts_per_tok: int = 2
    expert_dim: int = 1024
    act_threshold: float = 0.99
    rope_theta: float = 500000.0
    lora_rank: int = 8
    dropout: float = 0.0

    def estimate_memory(self, batch_size: int = 1, seq_len: int | None = None) -> float:
        """Estimate peak activation memory in GB (weights + KV cache).

        Args:
            batch_size: inference batch size.
            seq_len: sequence length; defaults to max_seq_len.
        """
        seq_len = seq_len or self.max_seq_len
        d = self.dim
        h = self.n_heads
        hk = self.n_kv_heads
        hd = d // h
        e = self.n_experts
        es = self.n_shared_experts
        ek = self.n_experts_per_tok
        ed = self.expert_dim
        T = self.max_loop_iters
        bytes_per_param = 4  # fp32

        # Embedding + head (tied)
        embed = self.vocab_size * d

        # Attention per block: Wq + Wk + Wv + Wo
        attn = d * (h * hd) + d * (hk * hd) * 2 + d * (h * hd)

        # Dense FFN per block (prelude/coda): 3 * dim * hidden
        ffn_dense = 3 * d * (d * 4 // 3)

        # MoE FFN (recurrent): routed + shared
        ffn_moe = e * (3 * d * ed) + es * (3 * d * ed * ek)

        # LoRA adapter
        lora = d * self.lora_rank + self.lora_rank * d + T * self.lora_rank

        # LTI injection + ACT halting
        misc = d * 3 + d  # A, B params + halt linear

        block_dense = attn + ffn_dense + misc * 0  # misc mostly in recurrent
        block_recurrent = attn + ffn_moe + lora + misc

        total_params = (
            embed
            + self.prelude_layers * block_dense
            + block_recurrent
            + self.coda_layers * block_dense
            + embed  # head shares embed weight, but counted once above
        )

        # KV cache: K and V per layer per token
        n_layers = self.prelude_layers + self.coda_layers + T
        kv_cache = batch_size * seq_len * hk * hd * 2 * n_layers

        total_bytes = total_params * bytes_per_param + kv_cache * bytes_per_param
        return total_bytes / (1024**3)
