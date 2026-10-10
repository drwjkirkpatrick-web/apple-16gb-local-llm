# Chapter 2: llama.cpp Setup (Fast Path)

Compiling llama.cpp from source gives ~15–20% faster generation than Ollama on the
same Mac, and — more importantly here — direct control of the KV cache that makes
**64K context** fit in 16 GB.

Hermes will not run below 64,000 tokens of context, so the flag list below is not
a tuning suggestion. It is the only combination that works.

---

## Why compile from source

- **Built for your chip** — flags tuned for M1/M2/M3/M4 Metal generations
- **Explicit KV cache control** — `-ctk`/`-ctv` and flash attention, which is the whole ballgame at 64K
- **No hidden defaults** — nothing silently downgrades your context window
- **Faster** — typically 30–38 tok/s vs 25–30 on the same machine

---

## Prerequisites

```bash
xcode-select --install
brew install cmake
```

**Build currency matters.** KV cache quantization (`-ctk`/`-ctv`) is a 2025-or-later
feature. `scripts/build-llamacpp.sh` checks for it and stops with a clear message
instead of letting you find out at 64K.

---

## Step 1: Build llama.cpp with Metal

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir -p build && cd build

cmake .. \
  -DGGML_METAL=ON \
  -DGGML_METAL_USE_BMETAL=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release -j "$(sysctl -n hw.ncpu)"
```

| Flag | Purpose |
|------|---------|
| `-DGGML_METAL=ON` | Metal GPU acceleration — the important one |
| `-DGGML_METAL_USE_BMETAL=ON` | Newer Metal shader compiler |
| `-DGGML_NATIVE=ON` | Optimize for this CPU's instruction set |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized, no debug overhead |
| `-j $(sysctl -n hw.ncpu)` | Parallel compile |

Verify KV quantization support before going further:

```bash
./bin/llama-server --help | grep -E 'cache-type|flash-attn'
#   -ctk, --cache-type-k TYPE
#   -ctv, --cache-type-v TYPE
#   -fa,  --flash-attn [on|off|auto]
```

No `--cache-type-k` line means the build cannot hold 64K on this machine.

---

## Step 2: Get the model

```bash
mkdir -p ~/models
hf download Qwen/Qwen3-8B-GGUF --include '*Q4_K_M*' --local-dir ~/models
# ~5 GB. Any Qwen3 8B Q4_K_M GGUF works.
```

Any Qwen3 8B Q4_K_M GGUF works; the launcher reads the architecture straight out
of the file's header, so it does not assume a filename.

---

## Step 3: Launch at 64K with an 8-bit KV cache

Use the launcher — it prints the memory budget, starts the server, waits for
health, and verifies the served context before telling you the Hermes settings:

```bash
bash scripts/launch-llm-64k.sh
```

Equivalent by hand:

```bash
cd ~/llama.cpp/build
./bin/llama-server \
  -m ~/models/Qwen3-8B-Q4_K_M.gguf \
  -c 65536 \
  -ngl 99 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  -np 1 \
  --host 127.0.0.1 \
  --port 8080 \
  --jinja
```

| Flag | Why |
|------|-----|
| `-c 65536` | Context window. Hermes needs ≥64,000 |
| `-ctk q8_0 -ctv q8_0` | 8-bit K/V cache: 9.00 GiB → 4.78 GiB at 64K |
| `-fa on` | Flash attention — **required** by llama.cpp for a quantized V cache |
| `-ngl 99` | All layers on the Metal GPU |
| `-np 1` | One slot. Each additional slot allocates its own 4.78 GiB cache |
| `--jinja` | Use the GGUF's chat template (Qwen3 tool calling) |

The launcher additionally adds YaRN RoPE scaling when the requested context
exceeds the window declared in the GGUF — Qwen3 8B declares 40,960 — so attention
does not degrade past that mark. Disable with `YARN=off` if your GGUF already
bakes in its own scaling.

---

## Step 4: Verify the server, then Hermes

```bash
# health
curl -s http://127.0.0.1:8080/health

# the number that matters
curl -s http://127.0.0.1:8080/props \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])'
# -> 65536

# a real completion
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
```

Then point Hermes at it:

```yaml
model:
  provider: custom
  base_url: "http://127.0.0.1:8080/v1"
  api_key: ""
  default: "qwen3-8b"
  context_length: 65536      # must be >= 64000
```

```bash
hermes chat -q "What tools do you have available?"
```

`scripts/hermes-llamacpp-config.yaml` has the whole block, compression settings
included.

---

## Performance: Ollama vs llama.cpp

On an M2 Mac mini (16 GB), 64K context, q8_0 KV cache:

| Metric | Ollama | llama.cpp (compiled) |
|--------|--------|----------------------|
| Tokens/second | ~25–30 | ~30–38 |
| Time to first token | ~1.5 s | ~1.2 s |
| Resident memory | ~6.5 GB | ~6.4 GB |
| KV flags available | server env only | `-ctk`/`-ctv` direct |
| Setup time | 5 minutes | 20 minutes |

Both fit. llama.cpp is faster and gives you the flags directly.

---

## Troubleshooting

**`V cache quantization requires flash_attn`**
Add `-fa on`. llama.cpp refuses a quantized V cache without flash attention.

**Server starts, but `/props` reports less than 65536**
Usually an old build clamping context, or `-c` overridden by a stray config.
`launch-llm-64k.sh` checks this for you and exits non-zero rather than leaving you
with a server Hermes will reject.

**Metal not engaged / very slow generation**
```bash
sudo powermetrics --samplers gpu_power -n 1
```
Near-zero GPU power during generation means Metal is not in use — rebuild with
`-DGGML_METAL=ON`.

**Swapping under load**
Close browsers/Docker, or relaunch with a 4-bit cache:
```bash
CACHE_K=q4_0 CACHE_V=q4_0 bash scripts/launch-llm-64k.sh
```

**Two slots, double the memory**
Do not raise `PARALLEL` on 16 GB. Slots do not share a KV cache.

**Updating llama.cpp**

```bash
cd ~/llama.cpp && git pull --ff-only
cd build && cmake --build . --config Release -j "$(sysctl -n hw.ncpu)"
bash scripts/launch-llm-64k.sh      # re-verifies context after the rebuild
```

---

## What's next

- [Why Qwen3 8B](03-model-selection.md)
- [Why 64K and q8_0](04-context-length.md)
