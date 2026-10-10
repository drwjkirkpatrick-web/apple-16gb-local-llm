# Why 64K Context and an 8-bit KV Cache

Hermes sets the target and the hardware sets the ceiling. Here is the arithmetic
that lands on exactly one configuration.

---

## The floor: 64,000 tokens

Hermes Agent will not run on less:

```python
# agent/model_metadata.py
MINIMUM_CONTEXT_LENGTH = 64_000
```

A session, a model switch or a cron job against a smaller window is rejected —
the framework decides the model cannot hold enough working memory for multi-step
tool calling. This is enforced at agent init, so a model served with a smaller
window does not "work but feel cramped"; Hermes never starts.

For a local setup, that makes the context window a requirement, not a preference.
Any guide that optimizes for the smallest workable window is optimizing for
something Hermes cannot use.

---

## The ceiling: what 16 GB can hold

An Apple Silicon Mac has unified memory, so the model, its cache, macOS and Hermes
all draw from the same 16 GB.

| Component | Typical |
|-----------|---------|
| macOS + WindowServer + background apps | 3–4 GiB |
| Hermes (Python, tools, session store) | 1–2 GiB |
| Browser/editor/Docker, if you keep them open | 2–6 GiB |
| **Available for the LLM** | **~10–11 GiB** |

So the design problem is fitting weights + KV cache + compute buffers into roughly
10.5 GiB, without leaving macOS so starved that it swaps to the SSD.

---

## The KV cache: the number that actually decides this

The cache is the model's working memory — one key and one value vector per token,
per KV head, per layer. For Qwen3 8B (36 layers, 8 KV heads, head_dim 128):

```
per token = 2 (K and V) × 36 layers × 8 heads × 128 dims × bytes per element
          = 73,728 elements per token
```

Block quantization adds a little overhead per 32 elements (a q8_0 block is 34
bytes: a 2-byte scale plus 32 signed bytes):

| KV cache type | Bytes/element | Per token | @ 65,536 tokens |
|---------------|---------------|-----------|-----------------|
| f32 | 4.0 | 288 KiB | 18.0 GiB |
| f16 / bf16 (llama.cpp default) | 2.0 | 144 KiB | 9.00 GiB |
| **q8_0** | **1.0625** | **76.5 KiB** | **4.78 GiB** |
| q4_0 | 0.5625 | 40.5 KiB | 2.53 GiB |

The entire problem in one line: **f16 needs 9.00 GiB for the cache alone, which
plus 4.68 GiB of weights leaves macOS with nothing.** q8_0 brings the cache under
5 GiB and the whole server to ~10.5 GiB, which fits — tightly, but genuinely.

The launcher computes these numbers from your actual model file and GGUF header
rather than assuming them.

---

## Why q8_0 and not q4_0

q4_0 saves another 2.25 GiB, which is real headroom on this machine. It also
quantizes the values the model attends over to 4 bits, and the loss shows up as
degraded recall over long contexts — exactly the thing you paid 64K to get.

Use q8_0 as the default. Drop to q4_0 only when you are swapping and cannot close
anything else. The launcher accepts `CACHE_K=q4_0 CACHE_V=q4_0` for that case.

---

## Required settings, and why each one

### llama.cpp

```bash
llama-server -m ~/models/Qwen3-8B-Q4_K_M.gguf   -c 65536 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -np 1
```

| Flag | Why it is there |
|------|-----------------|
| `-c 65536` | The window Hermes requires. Hermes needs ≥64,000; 65,536 is the clean power-of-two equivalent |
| `-ctk q8_0 -ctv q8_0` | Quantize the K and V caches. Without this, 64K does not fit |
| `-fa on` | Flash attention. llama.cpp requires it for a quantized V cache and will abort without it |
| `-ngl 99` | All layers on the Metal GPU (unified memory, so this is about compute, not VRAM) |
| `-np 1` | One server slot. Slots do not share a cache: two slots means two 4.78 GiB caches |

KV cache quantization needs a 2025-or-later llama.cpp build. `-ctk`/`-ctv` simply
do not exist in older binaries, which is why `build-llamacpp.sh` checks for them
and fails loudly instead of letting you discover it at 64K.

### Ollama

Ollama splits the same settings across two layers, which is why Ollama users
routinely end up with a 9 GiB cache they did not ask for:

```bash
# 1. per model — the window
ollama create qwen3-8b-64k -f - <<'EOF'
FROM qwen3:8b
PARAMETER num_ctx 65536
EOF

# 2. server-wide — the cache type. NOT a modelfile parameter.
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
```

`OLLAMA_KV_CACHE_TYPE` defaults to `f16` and is only honored with flash attention
enabled. Environment variables must reach the *server* process — `brew services`
starts Ollama through launchd, which does not read your shell profile, so
`setup-ollama.sh` also calls `launchctl setenv`.

---

## The trained window, and RoPE scaling

Qwen3 8B's GGUF declares a 40,960-token trained window. Asking for 65,536 without
adjusting the position encoding stretches the model past what it was trained on,
and attention quality drops.

The launcher enables YaRN scaling automatically when `CTX` exceeds that declared
window (`--rope-scaling yarn --rope-scale 1.6000 --yarn-orig-ctx 40960`, computed
from your GGUF's own metadata), and leaves it alone otherwise. If your GGUF already
carries baked-in scaling, pass `YARN=off`.

---

## Do not fill the window

Clearing the 64K floor is not a reason to run at 64K of live context. Attention
quality falls off at extreme lengths — the "lost in the middle" effect — and a
full cache is also when memory pressure gets dangerous.

The config snippets set Hermes to compress at 45% (~29K tokens):

```yaml
compression:
  enabled: true
  threshold: 0.45
  target_ratio: 0.20
```

You get the large window for the moments that need it — pasted logs, a long file,
a deep tool chain — while normal conversation stays in the range where an 8B model
is sharp.

---

## Summary

| | f16 KV @ 64K | **q8_0 KV @ 64K** | q4_0 KV @ 64K |
|---|---|---|---|
| KV cache | 9.00 GiB | **4.78 GiB** | 2.53 GiB |
| Total with weights | 14.68 GiB | **10.47 GiB** | 8.22 GiB |
| Left for macOS + Hermes | 1.32 GiB | **5.53 GiB** | 7.78 GiB |
| Long-context recall | best | **~indistinguishable** | noticeably worse |
| Survives a browser open? | no | **mostly** | yes |
| Verdict | unusable | **use this** | emergency squeeze |

**64K at q8_0 is the configuration: the context window Hermes requires, in the
memory a 16 GB Mac has.**
