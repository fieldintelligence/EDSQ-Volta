"""Draft kimi_k25 family descriptor for Colibri's family_registry.

Pattern mirrors the installed kimi_k3 descriptor (v1.11.0): a config-key
reader + PlannerGeometry. Pure function, no colibri imports, so it is
testable standalone before the upstream PR.

Kimi K2.5 Tower = DeepSeek-V3-MLA-family geometry (61 layers, 384 experts,
hidden 7168) WITHOUT the linear-attention block that kimi_k3 carries.
Values below marked TODO-VERIFY against the checkpoint config.json before
any plan is trusted (docs/engine-colibri.md contribution plan step 1).
"""


def kimi_k25_geometry(config: dict, context: int) -> dict:
    """PlannerGeometry equivalent for kimi_k25 (standalone draft)."""
    def req(key):
        if key not in config:
            raise ValueError(f"kimi_k25: missing planning key '{key}'")
        return int(config[key])

    layers = req("num_hidden_layers")            # 61 (1 dense + 60 MoE)
    experts = req("num_experts")                 # 384, 8 active per token
    hidden = req("hidden_size")                  # 7168
    kv_lora = req("kv_lora_rank")                # TODO-VERIFY: 512 (DSv3-MLA)
    q_lora = req("q_lora_rank")                  # TODO-VERIFY: 1536
    qk_rope = req("qk_rope_head_dim")            # TODO-VERIFY: 64

    # MLA compressed KV per token: kv_lora + rope (shared across heads)
    kv_bytes_per_token = context * (kv_lora + qk_rope) * 2 * 4   # fp16->fp32 staging?
    # fixed state: dense layer + shared-expert + norms (TODO-VERIFY vs config)
    fixed = (hidden * hidden * 2) * 2            # placeholder until verified
    return {
        "context_state_bytes": kv_bytes_per_token,
        "fixed_state_bytes": fixed,
        "workspace_bytes": q_lora * 2 * 1024,    # placeholder
        "configured_experts": experts,
        "layers": layers,
        "moe_layers": layers - 1,
    }


if __name__ == "__main__":
    import json
    demo = {"num_hidden_layers": 61, "num_experts": 384, "hidden_size": 7168,
            "kv_lora_rank": 512, "q_lora_rank": 1536, "qk_rope_head_dim": 64}
    print(json.dumps(kimi_k25_geometry(demo, 32768), indent=1))
