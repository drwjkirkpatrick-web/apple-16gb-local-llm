# Project State: apple-16gb-local-llm

> **Last updated:** 2026-10-09
> **Current phase:** maintenance / verification
> **Overall health:** yellow — logic verified against a stubbed server; the 16 GB Mac run is still pending

---

## 1. Goal

Document and script the only configuration that runs Hermes Agent locally on a
16 GB Apple Silicon Mac: Qwen3 8B at 64K context with an 8-bit (q8_0) KV cache.

## 2. Current Status

### Done
- [x] Audited the original project: every doc and script targeted 32K, which Hermes rejects outright (`MINIMUM_CONTEXT_LENGTH = 64_000`)
- [x] `scripts/launch-llm-64k.sh` — new launcher: floor guard, GGUF header read, memory budget, health poll, served-context verification, Hermes handoff
- [x] `scripts/setup-ollama.sh` — rebuilt for 64K + server-side q8_0 KV cache (`OLLAMA_KV_CACHE_TYPE`, `launchctl setenv`), verifies live context via `/api/ps`
- [x] `scripts/build-llamacpp.sh` — checks `--cache-type-k` support, fetches model, hands off to the launcher
- [x] Both Hermes config snippets repointed at 65536 with compression at 0.45
- [x] README rewritten around 64K; all 32K references removed
- [x] `docs/04-context-length.md` rewritten (64K + q8_0 vs q4_0 arithmetic)
- [x] `docs/01`, `docs/02`, `docs/03` updated for 64K
- [x] Stub-harness test: full launch flow, floor guard (`CTX=32768` → exit 1), q4_0 squeeze variant, missing-model path
- [x] Two real bugs found by that test and fixed: KV K+V double-counted (9.56 → 4.78 GiB), wrong q4_0/q5_0 block denominators

### In Progress
- [ ] End-to-end run on real 16 GB Apple Silicon hardware (not available in this environment — 8 GB Jetson only)

### Not Started
- [ ] Commit + push to `origin main`
- [ ] Record measured tok/s and resident memory from the real Mac in the README performance tables
- [ ] Confirm YaRN behavior against the shipped Qwen3 8B GGUF (`YARN=auto` vs `off`)

## 3. Architecture & Key Decisions

| Decision | Rationale | Date |
|---|---|---|
| Target 65,536 tokens, not 65,000 | Clean power of two above the 64,000 Hermes floor | 2026-10-09 |
| q8_0 KV cache as the default | 9.00 GiB → 4.78 GiB at 64K; that difference is the entire project | 2026-10-09 |
| Keep Qwen3 8B rather than shrinking | Qwen3 4B has the identical 36×8×128 KV shape, so it saves weights only | 2026-10-09 |
| Launcher computes the budget from the real GGUF | Avoids publishing assumed numbers; script reads layers/KV heads/head_dim from the header | 2026-10-09 |
| Launcher refuses CTX < 64000 | 32K configurations are not "smaller", they are non-functional under Hermes | 2026-10-09 |
| Bounded portable `--help` probe | Piping a server binary into grep can hang; macOS has no GNU `timeout` | 2026-10-09 |

## 4. Blockers & Risks

- **Blocker:** no 16 GB Apple Silicon Mac in this environment → cannot measure real tok/s, resident memory, or swap behavior. Mitigation: stub-server harness covers control flow; hardware pass scheduled separately.
- **Risk:** `-fa on` value syntax requires a 2025+ llama.cpp (post tri-state FA change). Mitigation: script fails loudly on builds without `--cache-type-k`.
- **Risk:** YaRN may double-apply if a GGUF already bakes in scaling metadata. Mitigation: `YARN=off` escape hatch, documented in README and Chapter 2.
- **Risk:** 5.53 GiB free for macOS + Hermes is thin; a browser can push it into swap. Mitigation: documented q4_0 squeeze path (2.53 GiB cache).

## 5. Next Step

> **Next:** Run `bash scripts/launch-llm-64k.sh` on the 16 GB Apple Silicon Mac and record the verified `n_ctx`, resident memory, and tok/s into the README tables.

## 6. Environment & Tooling Notes

- Target: Apple Silicon macOS 13+, 16 GB, Metal
- Llama.cpp: 2025+ build required (KV cache quantization)
- Model: ~/models/Qwen3-8B-Q4_K_M.gguf (Qwen/Qwen3-8B-GGUF, ~5 GB)
- Hermes: `model.context_length: 65536` (hard floor 64,000)
- Skills used: `hermes-agent`, `apple-silicon-llm-sizing`, `project-state-management`
- Deployment target: local machine only; no service, no CI

## 7. Recent Session Log

- 2026-10-09: Full 64K conversion — new launcher, Ollama KV-cache fix, README/docs rewritten, stub-harness verified, two math bugs caught and fixed.

## 8. References

- Hermes minimum context: `agent/model_metadata.py` → `MINIMUM_CONTEXT_LENGTH = 64_000`
- Hermes local models: https://hermes-agent.nousresearch.com/docs/user-guide/local-models
- llama.cpp KV cache flags: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Ollama context/KV env vars: https://docs.ollama.com/faq
- Qwen3 8B architecture: `Qwen/Qwen3-8B` config.json — 36 layers, 8 KV heads, head_dim 128
