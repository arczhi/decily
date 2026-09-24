"""Minimal MLX inference for the decision model (Apple Silicon, no torch).

Loads the safetensors bundle produced by scripts/export_safetensors.py and runs
the cross-encoder decision forward: [state; question; candidate] -> causal
Qwen3 backbone -> attention pooling over the full sequence -> scalar head ->
softmax over candidates.

Usage:
    python mlx/decision_mlx.py --model-dir ~/models/v5_mlx \
        --state "..." --question "Which department should handle this ticket?" \
        --options "returns,billing,shipping,technical support" --temperature 0.45
"""

from __future__ import annotations

import argparse
import json
import os

import mlx.core as mx

NEG = -1e9


def _rms(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """RMSNorm with fp32 internals (matches HF Qwen3RMSNorm numerics)."""
    dt = x.dtype
    x32 = x.astype(mx.float32)
    var = (x32 * x32).mean(axis=-1, keepdims=True)
    y = x32 * mx.rsqrt(var + eps)
    return (y * w.astype(mx.float32)).astype(dt)


def _rope(x: mx.array, theta: float, offset: int = 0) -> mx.array:
    """HF rotate_half RoPE (mx.fast.rope's conventions differ from HF)."""
    L, H, D = x.shape
    half = D // 2
    pos = mx.arange(L, dtype=mx.float32) + offset
    inv = 1.0 / (theta ** (mx.arange(0, D, 2, dtype=mx.float32) / D))
    freqs = mx.outer(pos, inv)
    cos = mx.concatenate([mx.cos(freqs), mx.cos(freqs)], axis=-1)[:, None, :]
    sin = mx.concatenate([mx.sin(freqs), mx.sin(freqs)], axis=-1)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    rot = mx.concatenate([-x2, x1], axis=-1)
    return x * cos.astype(x.dtype) + rot * sin.astype(x.dtype)


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / mx.sqrt(mx.array(2.0))))


class DecisionModelMLX:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        self.w = mx.load(os.path.join(model_dir, "model.safetensors"))
        self.n_layers = self.cfg["num_hidden_layers"]
        self.n_heads = self.cfg["num_attention_heads"]
        self.n_kv = self.cfg["num_key_value_heads"]
        self.head_dim = self.cfg["head_dim"]
        self.eps = self.cfg["rms_norm_eps"]
        self.rope_theta = float(self.cfg["rope_theta"])
        self.scale = self.head_dim**-0.5

    # ---- backbone ----
    def _attn(self, x: mx.array, p: str) -> mx.array:
        L = x.shape[0]
        q = x @ self.w[p + "self_attn.q_proj.weight"].T
        k = x @ self.w[p + "self_attn.k_proj.weight"].T
        v = x @ self.w[p + "self_attn.v_proj.weight"].T
        q = q.reshape(L, self.n_heads, self.head_dim)
        k = k.reshape(L, self.n_kv, self.head_dim)
        v = v.reshape(L, self.n_kv, self.head_dim)
        q = _rms(q, self.w[p + "self_attn.q_norm.weight"], self.eps)
        k = _rms(k, self.w[p + "self_attn.k_norm.weight"], self.eps)
        q = _rope(q, self.rope_theta)
        k = _rope(k, self.rope_theta)
        reps = self.n_heads // self.n_kv
        if reps > 1:
            k = mx.repeat(k, reps, axis=1)
            v = mx.repeat(v, reps, axis=1)
        q = q.transpose(1, 0, 2)
        k = k.transpose(1, 0, 2)
        v = v.transpose(1, 0, 2)
        causal = mx.triu(mx.full((L, L), NEG, dtype=q.dtype), k=1)[None, None]
        o = mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=self.scale, mask=causal
        )[0]
        o = o.transpose(1, 0, 2).reshape(L, -1)
        return o @ self.w[p + "self_attn.o_proj.weight"].T

    def _mlp(self, x: mx.array, p: str) -> mx.array:
        g = x @ self.w[p + "mlp.gate_proj.weight"].T
        gate = g * mx.sigmoid(g)
        up = x @ self.w[p + "mlp.up_proj.weight"].T
        return (gate * up) @ self.w[p + "mlp.down_proj.weight"].T

    def backbone(self, ids: list[int]) -> mx.array:
        h = self.w["encoder.model.embed_tokens.weight"][mx.array(ids)]
        for i in range(self.n_layers):
            p = f"encoder.model.layers.{i}."
            r = _rms(h, self.w[p + "input_layernorm.weight"], self.eps)
            h = h + self._attn(r, p)
            r = _rms(h, self.w[p + "post_attention_layernorm.weight"], self.eps)
            h = h + self._mlp(r, p)
        return _rms(h, self.w["encoder.model.norm.weight"], self.eps)

    # ---- decision head ----
    def score(self, ids: list[int]) -> float:
        h = self.backbone(ids).astype(mx.float32)
        scores = (h @ self.w["pool.proj.weight"].T).squeeze(-1) + self.w["pool.proj.bias"]
        w = mx.softmax(scores, axis=-1)
        pooled = (h * w[:, None]).sum(axis=0)
        ln_w, ln_b = self.w["head.0.weight"], self.w["head.0.bias"]
        mu = pooled.mean()
        var = ((pooled - mu) ** 2).mean()
        x = (pooled - mu) / mx.sqrt(var + 1e-5) * ln_w + ln_b
        x = _gelu(x @ self.w["head.1.weight"].T + self.w["head.1.bias"])
        return float((x @ self.w["head.4.weight"].T + self.w["head.4.bias"]).item())

    def decide(
        self,
        state: str,
        question: str,
        options: list[str],
        tokenizer,
        temperature: float = 1.0,
        max_state_tokens: int = 256,
        max_question_tokens: int = 96,
        max_option_tokens: int = 64,
    ) -> list[tuple[str, float]]:
        s_ids = tokenizer(state, truncation=True, max_length=max_state_tokens)["input_ids"]
        q_ids = tokenizer(question, truncation=True, max_length=max_question_tokens)["input_ids"]
        logits = []
        for opt in options:
            c_ids = tokenizer(opt, truncation=True, max_length=max_option_tokens)["input_ids"]
            logits.append(self.score(s_ids + q_ids + c_ids))
        z = mx.array(logits) / temperature
        p = mx.softmax(z, axis=-1)
        mx.eval(p)
        return sorted(zip(options, p.tolist()), key=lambda t: -t[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--options", required=True, help="comma-separated")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = DecisionModelMLX(args.model_dir)
    results = model.decide(
        args.state, args.question, [o.strip() for o in args.options.split(",")], tok,
        temperature=args.temperature,
    )
    print(json.dumps({opt: round(p, 4) for opt, p in results}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
