# JEV SystemOne validated as verifier lane — with an honest negative result (2026-10-09)

First end-to-end SystemOne integration from the EDSQ side. Toddler's JEV
server (`jevserver`, toddler repo) is **not a chat model but a logprob
classifier**: `POST /v1/systemone` takes a state + typed questions (noul =
p(yes), choice = label probabilities, score = ordinal) and answers with a
single-token probability read from the underlying llama.cpp model
(qwen38-27b on :8011) — 1 output token per question, TTL-cached on
quantised state.

## Operational notes (needed to make it work)

- Auth: keys are sha256-only in `~/.config/toddler-jev/keys.json`; mint via
  `python3 -m jevserver keygen --label x` (stored: `edsq-verifier.key`, 600).
- Serve needs `WorkingDirectory=toddler-repo` (module path), port 8092.
- Labels must be **single tokens** (A/B/C, yes/no): matching is exact-token
  or unambiguous-prefix only — "8.2 percent" as a label yields
  `model put no probability on any allowed answer`.
- Backend timeout is a hard-coded 30 s on the shared qwen38 server: cold
  prompts can exceed it — retry once; `cache_prompt` makes the retry fast.

## Calibration (the portfolio question, correct = 8.2 %)

| probe | result |
|---|---|
| choice A=8.2 / B=8.9 / C=9.2 | **B (0.424)** — the wrong answer; A=0.312, C=0.264 |
| noul "draft 8.9 % correct?" | p(yes)=0.529 — uninformative |
| noul "draft 8.2 % correct?" | p(yes)=0.526 — uninformative |
| (control) noul factual market-hours question | 0.990 — sharp |
| usage per call | ~85–196 in / **1 out** token — ~free vs a generation |

## Conclusions

1. **Protocol validated end-to-end** (auth, noul, choice, cache, usage).
2. **Honest negative result: JEV inherits its base model's blind spot.**
   qwen38-27b falls for the covariance trap (votes 8.9 like its own draft);
   a verifier that shares the proposer's error adds no ensemble diversity.
   The judges that DO catch it (GLM-5.3/DSv4) win by re-deriving, not by
   pattern-matching — cheap logprob verification cannot replace them on
   computation-bearing questions.
3. **Where JEV IS the right tool**: factual/perception gating (0.990 on the
   control), routing, scrape-error triage — any yes/no or small-label task
   where 1 output token at ~0 cost beats a 300-token generation. That is
   the toddler agent loop it was built for.
4. **No OpenAI adapter needed** for this role: SystemOne is directly
   callable. A suite integration would be a specialist "perception lane",
   not a chat battery row — roadmap item 8 updated accordingly.
