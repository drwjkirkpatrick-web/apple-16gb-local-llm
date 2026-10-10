# Chapter 1: Ollama Setup (Easy Path)

Get a local LLM serving **64K context** on a 16 GB Apple Silicon Mac without
compiling anything. Five steps; the fourth is the one that matters.

Hermes Agent requires at least 64,000 tokens of context, so the window is fixed
up front. What varies is whether that window fits in memory — and that comes down
to the KV cache type.

---

## Why Ollama

Ollama wraps model files, Metal acceleration and an OpenAI-compatible API behind
one CLI. For a 16 GB Mac:

- **Zero build step** — pre-compiled with Metal support
- **Model variants** — a one-liner creates a 64K context derivate
- **OpenAI-compatible API** — Hermes talks to it directly
- **One catch** — KV cache precision is a *server* setting, not a model setting

That last point is where most 64K-on-16-GB attempts fail. Read Step 2 carefully.

---

## Step 1: Install and start Ollama

```bash
brew install ollama
brew services start ollama

# verify
curl -s http://localhost:11434/api/version
```

---

## Step 2: Enable an 8-bit KV cache (the critical step)

At 64K context, Qwen3 8B's KV cache costs 9.00 GiB at Ollama's default `f16`.
That does not fit alongside 4.7 GiB of weights and macOS. Quantized to q8_0 it
costs 4.78 GiB, which does.

```bash
export OLLAMA_FLASH_ATTENTION=1     # required for a quantized cache
export OLLAMA_KV_CACHE_TYPE=q8_0    # f16 -> q8_0: 9.00 GiB -> 4.78 GiB
export OLLAMA_CONTEXT_LENGTH=65536  # default context for any model loaded
```

Persist these, because `brew services` runs Ollama through launchd and **launchd
does not read your shell profile**:

```bash
# for GUI/launchd-started processes
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
launchctl setenv OLLAMA_CONTEXT_LENGTH 65536

# for a manually started server
echo 'export OLLAMA_FLASH_ATTENTION=1'  >> ~/.zshrc
echo 'export OLLAMA_KV_CACHE_TYPE=q8_0' >> ~/.zshrc
echo 'export OLLAMA_CONTEXT_LENGTH=65536' >> ~/.zshrc

brew services restart ollama
```

`scripts/setup-ollama.sh` does all of the above for you, idempotently, and then
verifies the result. Use it if you would rather not hand-edit anything.

---

## Step 3: Pull Qwen3 8B and create the 64K variant

```bash
ollama pull qwen3:8b

ollama create qwen3-8b-64k -f - <<'EOF'
FROM qwen3:8b
PARAMETER num_ctx 65536
PARAMETER keep_alive 30m
EOF
```

`num_ctx` is what Ollama actually allocates — the model's advertised maximum is
irrelevant. `keep_alive 30m` stops Ollama unloading the 5 GB model between
requests.

---

## Step 4: Verify the running window, not the config

```bash
ollama run qwen3-8b-64k "say ok"      # force a load, then Ctrl+D

ollama ps
#   NAME              SIZE     PROCESSOR   CONTEXT
#   qwen3-8b-64k      5.9 GB   100% GPU    65536      <- this must be >= 64000
```

If `CONTEXT` is smaller than 64,000, the KV cache variables did not reach the
server. Confirm by hand with them exported in front:

```bash
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=65536 ollama serve
```

---

## Step 5: Install Hermes and point it at Ollama

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

hermes model    # → Custom endpoint (self-hosted / vLLM / llama.cpp)
#   Base URL:        http://127.0.0.1:11434/v1
#   API key:         (leave blank)
#   Model:           qwen3-8b-64k
#   Context length:  65536        <- must be >= 64000
```

Or edit `~/.hermes/config.yaml` (`hermes config edit`):

```yaml
model:
  provider: custom
  base_url: "http://127.0.0.1:11434/v1"
  api_key: ""
  default: "qwen3-8b-64k"
  context_length: 65536
```

A ready-to-paste version lives in `scripts/hermes-ollama-config.yaml`, including
a compression threshold of 0.45 so the cache never actually fills to 64K.

---

## Test it

```bash
hermes chat -q "What tools do you have available?"
hermes doctor
```

If Hermes starts and answers, you are done — it validated the 64K window on the
way in.

---

## Troubleshooting

**Hermes refuses to start: context too small**
`ollama ps` is showing less than 64,000. Fix the server, not Hermes. Usually
`OLLAMA_KV_CACHE_TYPE` never reached the process — see Step 2.

**The Mac swaps**
The cache did not quantize (check `ollama ps` size against ~5.9 GB — an f16 cache
at 64K inflates it several GB), or other apps are eating memory. Close browsers
and Docker.

**Out of memory with the cache applied**
Drop the request's cache to 4-bit: `OLLAMA_KV_CACHE_TYPE=q4_0` (2.53 GiB at 64K).
Quality on long context suffers, but it fits.

**Model unloads and reloads constantly**
Increase `keep_alive` in the modelfile, or set `OLLAMA_KEEP_ALIVE=30m`.

**Context silently clamped below what you asked for**
Ollama reduces `num_ctx` when the cache will not fit. That is a signal the q8_0
cache type is not active.

---

## What's next

- [Chapter 2](02-llamacpp-setup.md) — the same 64K setup on llama.cpp, ~15–20% faster
- [Why Qwen3 8B](03-model-selection.md) — the model choice
- [Why 64K and q8_0](04-context-length.md) — the arithmetic behind all of this
