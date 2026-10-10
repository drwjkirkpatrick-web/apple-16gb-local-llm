# Local LLM for Hermes Agent on a 16 GB Apple Silicon Mac

Run a fully local, private Hermes Agent on a 16 GB Apple Silicon iMac or Mac mini
using **Qwen3 8B at 64K context with an 8-bit KV cache**.

No API keys. No cloud. No data leaving the machine.

---

## Why this exists

Hermes Agent **refuses to start** against a model with less than 64,000 tokens of
context. It is a hard constant, not a preference:

```python
# hermes-agent/agent/model_metadata.py
MINIMUM_CONTEXT_LENGTH = 64_000   # sessions, model switches and cron jobs reject anything smaller
```

Most "run a local LLM on 16 GB" guides recommend a small context window because
that is what fits by default. Those setups cannot run Hermes at all — the agent
errors out on startup. So this repo solves the real problem: **how to hold 64K of
context on 16 GB of unified memory.**

The answer is an 8-bit KV cache. At 64K tokens the attention memory for Qwen3 8B is:

- **9.00 GiB** at the default f16 — more than half the machine, unusable
- **4.78 GiB** at q8_0 — fits, with 5.5 GiB left for macOS and Hermes

Everything here exists to make that configuration work, verified end to end.

---

## The memory budget (measured, not guessed)

Qwen3 8B is 36 layers with 8 KV heads of 128 dimensions, so the cache costs
`2 × 36 × 8 × 128 = 76.5 KiB per token` at q8_0. At 65,536 tokens:

| Component | Size |
|-----------|------|
| Model weights (Q4_K_M, file on disk) | 4.68 GiB |
| **KV cache q8_0 @ 65,536 tokens** | **4.78 GiB** |
| Metal compute buffers | ~1.00 GiB |
| **llama-server total** | **~10.47 GiB** |
| Left for macOS + Hermes | ~5.53 GiB of 16 GB |

The launcher script prints this table for *your* actual model file and context
size before it starts anything, and refuses to run below the Hermes floor.

---

## What you get

- A **private AI agent** running entirely on your Mac
- **Hermes Agent** with tool calling (terminal, file access, web search, skills, memory)
- **64K context** — above the floor Hermes enforces, with a q8_0 KV cache
- **~20–30 tokens/second** generation (Apple Silicon Metal acceleration)
- No API costs, no rate limits, no data sent to third parties

---

## Two paths to the same destination

| Path | Difficulty | Speed | Best for |
|------|-----------|-------|----------|
| **Chapter 1: llama.cpp** | Intermediate (compile once) | Fastest | Recommended — the launcher verifies 64K for you |
| **Chapter 2: Ollama** | Easy (no compiling) | Good | Fastest to set up, fewer knobs |

Both end with Hermes talking to a local Qwen3 8B at 64K. Pick one.

---

## Quick start — llama.cpp (recommended)

```bash
# 1. Build llama.cpp with Metal support (needs a 2025+ build for quantized KV cache)
brew install cmake
bash scripts/build-llamacpp.sh

# 2. Get Qwen3 8B Q4_K_M (~5 GB) if you don't have it
hf download Qwen/Qwen3-8B-GGUF --include '*Q4_K_M*' --local-dir ~/models

# 3. Launch at 64K with the q8_0 KV cache
bash scripts/launch-llm-64k.sh
```

The launcher checks the binary supports `-ctk`, prints the memory budget, starts
the server, waits for health, and **confirms the server really reports 65,536
tokens** before telling you the Hermes settings. Under the hood it runs:

```bash
llama-server -m ~/models/Qwen3-8B-Q4_K_M.gguf   -c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1   --host 127.0.0.1 --port 8080 --jinja
```

- `-c 65536` — the context window (the point of the whole exercise)
- `-fa on` — flash attention, **required** by llama.cpp for a quantized V cache
- `-ctk q8_0 -ctv q8_0` — 8-bit K and V cache; this is what makes 64K fit
- `-np 1` — one slot. Every extra slot allocates its own 4.78 GiB cache

Then point Hermes at it:

```yaml
# ~/.hermes/config.yaml  (or: hermes model → Custom endpoint)
model:
  provider: custom
  base_url: "http://127.0.0.1:8080/v1"
  api_key: ""
  default: "qwen3-8b"
  context_length: 65536        # must be >= 64000
```

```bash
hermes chat -q "What tools do you have available?"
```

---

## Quick start — Ollama

```bash
bash scripts/setup-ollama.sh
```

That script does the two things an Ollama 64K setup needs — and the second one is
the step virtually every guide misses:

1. **Per model**: creates `qwen3-8b-64k` with `PARAMETER num_ctx 65536`
2. **Server-wide**: exports `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`

The KV cache type in Ollama is a *server* setting, not a modelfile parameter. Skip
it and 64K needs 9 GiB of cache instead of 4.78 GiB, and macOS starts swapping.
The script also runs `launchctl setenv` so `brew services` (which never reads your
shell profile) picks the variables up, then verifies the live context in
`ollama ps` and warns you if it came up short.

Then configure Hermes the same way, with `default: "qwen3-8b-64k"`.

---

## Repository contents

```
.
├── README.md                          ← You are here
├── Project_state.md                   ← Current status / next step
├── docs/
│   ├── 01-ollama-setup.md              ← Chapter 1: Ollama (easy path)
│   ├── 02-llamacpp-setup.md            ← Chapter 2: llama.cpp (fast path)
│   ├── 03-model-selection.md           ← Why Qwen3 8B
│   └── 04-context-length.md            ← Why 64K + an 8-bit KV cache
├── scripts/
│   ├── launch-llm-64k.sh               ← Launch llama-server at 64K, verified
│   ├── setup-ollama.sh                 ← Ollama + q8_0 KV cache + 64K variant
│   ├── build-llamacpp.sh               ← Build llama.cpp, check KV support
│   ├── hermes-llamacpp-config.yaml      ← Hermes config for the llama.cpp path
│   └── hermes-ollama-config.yaml        ← Hermes config for the Ollama path
└── LICENSE
```

---

## System requirements

- **Mac**: Apple Silicon (M1, M2, M3, M4) with **16 GB RAM**
- **macOS**: 13.0 (Ventura) or newer
- **Disk**: ~6 GB for the model, ~2 GB for tooling
- **Xcode Command Line Tools** (llama.cpp path): `xcode-select --install`
- **llama.cpp**: a 2025-or-later build — KV cache quantization (`-ctk`/`-ctv`) is mandatory here

---

## Verification: prove you have 64K

Do not trust the config, ask the server. Both paths have a one-liner:

```bash
# llama.cpp — the launcher does this automatically
curl -s http://127.0.0.1:8080/props | python3 -c 'import json,sys; print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])'
# -> 65536

# Ollama
ollama ps          # the CONTEXT column must read 65536
```

If that number is below 64000, Hermes will refuse to start. Fix the server before
you debug Hermes.

---

## What this setup is good at

- ✅ Chat and Q&A
- ✅ Tool calls (terminal commands, file reads/writes, search)
- ✅ Web search and content extraction
- ✅ Multi-step agent workflows that need a real context window
- ✅ Private, offline, no-API-key AI assistance

## What this setup struggles with

- ⚠️ Complex reasoning chains (an 8B model loses the plot on long chains, regardless of context)
- ⚠️ Running heavy apps alongside it — the LLM holds ~10.5 GiB of your 16 GB
- ⚠️ Very large code generation (works; a bigger model is noticeably better)
- ⚠️ Long single documents: attention quality degrades at extreme lengths even when the window allows them

---

## Tuning knobs

`launch-llm-64k.sh` reads environment variables, so nothing needs editing:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CTX` | `65536` | Context window. Refuses to go below the Hermes floor |
| `CACHE_K` / `CACHE_V` | `q8_0` | KV cache precision — drop to `q4_0` to free ~2.3 GiB |
| `PORT` / `HOST` | `8080` / `127.0.0.1` | Where the API listens |
| `MODEL` | `~/models/Qwen3-8B-Q4_K_M.gguf` | Any Qwen3-8B Q4_K_M GGUF |
| `NGL` | `99` | Layers on the GPU (all of them) |
| `PARALLEL` | `1` | Server slots — each one costs a full 4.78 GiB cache |
| `YARN` | `auto` | RoPE scaling when you ask for more than the trained window |
| `JINJA` | `1` | Use the GGUF's own chat template (Qwen3 tool calls) |

**If your Mac starts swapping**, in order: close browsers and Docker, then
`CACHE_K=q4_0 CACHE_V=q4_0 bash scripts/launch-llm-64k.sh` (KV drops from 4.78 GiB
to 2.53 GiB, and 64K still works — quality loss shows up in long-context recall).

**A smaller model is not the easy fix.** Qwen3 4B has the same 36 layers × 8 KV
heads × 128 head_dim shape, so its cache costs the identical 4.78 GiB at 64K. It
saves ~2.2 GB of weights and nothing else, while making the agent dumber. Adjust
the KV precision instead.

---

## Context smoothing in Hermes

64K is the floor you must clear, not the working size you should aim to fill. The
config snippets set compression to trigger at 45% (~29K tokens), which keeps the
cache comfortably inside the window:

```yaml
compression:
  enabled: true
  threshold: 0.45
  target_ratio: 0.20
```

---

## Upgrading later

The model is the only thing that changes — Hermes, the launcher and your configs
stay as they are.

| Mac RAM | Model | Quantization | Context |
|---------|-------|--------------|---------|
| 16 GB | Qwen3 8B | Q4_K_M | 64K, q8_0 KV |
| 24 GB | Qwen3 14B | Q4_K_M | 64K+ |
| 48 GB | Qwen3 14B | Q8_0 | 128K |
| 64 GB+ | Llama 3.3 70B | Q4_K_M | 128K |

---

## Troubleshooting

**`V cache quantization requires flash_attn` at startup**
Add `-fa on`. llama.cpp will not use a quantized V cache without flash attention;
the launcher passes it for you.

**Hermes exits immediately, complaining the context is too small**
The server is serving less than 64,000 tokens. Check `curl .../props` (llama.cpp)
or `ollama ps` (Ollama), not the config file.

**The Mac swaps and stutters**
The KV cache is the culprit. Drop to `CACHE_K=q4_0 CACHE_V=q4_0`, or close
everything else. A 16 GB Mac running Hermes locally is near its ceiling by design.

**Server reports a lower context than you asked for**
Ollama silently clamps `num_ctx` when the KV cache will not fit — that usually
means `OLLAMA_KV_CACHE_TYPE` never reached the server. Run it yourself in a shell
where the variables are exported: `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`.

**Responses degrade on long conversations**
Expected. Attention quality falls off at extreme lengths even inside the window,
and Hermes compresses at 45% to keep you away from the edge.

**`llama-server not found`**
Build it (`scripts/build-llamacpp.sh`) or pass `LLAMA_SERVER=/path/to/llama-server`.

---

## References

- Hermes Agent docs — [Local models](https://hermes-agent.nousresearch.com/docs/user-guide/local-models)
- Hermes Agent — [Quickstart & minimum context](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart/)
- llama.cpp — [KV cache quantization flags](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
- Ollama — [FAQ: context length and KV cache type](https://docs.ollama.com/faq)

---

## Credits

Built by [Walker Kirkpatrick](https://github.com/drwjkirkpatrick-web) with assistance from [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research.

Model: [Qwen3 8B](https://qwenlm.github.io/) by Alibaba Qwen Team.

## License

MIT — see [LICENSE](LICENSE).
