# Upstream PR draft: add kimi_k25 family (JustVugg/colibri)

## Checklist before opening the PR
- [ ] Verify every TODO-VERIFY value against the K2.5 Tower `config.json`
      (hidden 7168 / 61 layers / 384 experts confirmed; MLA ranks not yet)
- [ ] Run `coli plan --auto-tier` with the descriptor against the native
      INT4 checkpoint on eds1; compare predicted tiers with
      `vllm_volta.moe.predict_decode_seconds` (docs/design-k25-scaling.md)
- [ ] Serve smoke test: `coli serve --model-id kimi-k2.5-tower --no-think`
      + one greedy completion
- [ ] Cite test hardware (node2: 2×V100-SXM2 NVLink, 512G DDR4, 4×512G
      Optane PMem DAX) and share the tier-plan diff in the PR body
- [ ] License note: descriptor is config data, weights stay Moonshot's

## PR body (draft)
Title: "family: add kimi_k25 (MLA, 61L, 384 experts, native INT4 tier plan)"
Body: descriptor + geometry fn (pattern follows kimi_k3), verification
numbers from a 2×V100-SXM2 + Optane PMem box, cross-check vs our open-source
tier model. References: 1CatAI/1Cat-vLLM#987 (our SM70/V100 context).
