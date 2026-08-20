# Chapter 2: llama.cpp Setup (Fast Path)

Compiling llama.cpp from source gives you ~15–20% faster inference than Ollama on the same hardware. This is the path for tinkerers who want maximum performance from a 16 GB Apple Silicon Mac.

---

## Why compile from source

Ollama is convenient but ships a general-purpose binary. When you build llama.cpp yourself on your Mac, the compiler can:

- **Optimize for your exact chip** (M1 vs M2 vs M3 vs M4 — different GPU architectures)
- **Enable Metal-specific optimizations** not in the generic build
- **Tune batch sizes and thread counts** to your core count
- **Skip features you don't need** (smaller binary, less overhead)

The real-world difference is ~15–20% faster token generation. On a 16 GB Mac where every token matters, that's worth the 20 minutes of setup.

---

## Prerequisites

### Xcode Command Line Tools

```bash
xcode-select --install
```

If a dialog appears, click "Install" and wait for it to finish. If it says "already installed", you're good.

### CMake

```bash
brew install cmake
```

Verify:
```bash
cmake --version
# Should be 3.16+ 
```

---

## Step 1: Clone llama.cpp

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

---

## Step 2: Build with Metal support

```bash
mkdir build
cd build

cmake .. \
  -DGGML_METAL=ON \
  -DGGML_METAL_USE_BMETAL=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release -j $(sysctl -n hw.ncpu)
```

### What these flags do

| Flag | Purpose |
|------|---------|
| `-DGGML_METAL=ON` | Enable Apple Metal GPU acceleration (the big one) |
| `-DGGML_METAL_USE_BMETAL=ON` | Use the newer Metal shader compiler (faster kernels) |
| `-DGGML_NATIVE=ON` | Auto-detect and optimize for your CPU's instruction set |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build, no debug overhead |
| `-j $(sysctl -n hw.ncpu)` | Use all CPU cores for compiling (faster build) |

### Verify the build

```bash
./bin/llama-cli --version
```

If you see a version string, the build succeeded.

### Verify Metal is enabled

```bash
./bin/llama-cli -m /dev/null -p "test" 2>&1 | grep -i metal
```

You should see something like `Metal device: ...` in the output. If you don't see Metal mentioned, the GPU acceleration isn't working and inference will be CPU-only (very slow).

---

## Step 3: Download the model

We need the GGUF format of Qwen3 8B at Q4_K_M quantization.

### Option A: Download from HuggingFace (recommended)

```bash
# Create a models directory
mkdir -p ~/models
cd ~/models

# Download Qwen3 8B Q4_K_M (about 5 GB)
# Visit https://huggingface.co/Qwen/Qwen3-8B-Instruct-GGUF
# Download the file named something like:
# qwen3-8b-instruct-q4_k_m.gguf

# Or use curl if you know the exact URL:
curl -L -o qwen3-8b-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen3-8B-Instruct-GGUF/resolve/main/qwen3-8b-instruct-q4_k_m.gguf"
```

### Option B: Convert from Ollama

If you already pulled the model via Ollama, you can locate the blob and use it directly, but this is more complex. The HuggingFace download is simpler.

### Verify the model file

```bash
ls -lh ~/models/qwen3-8b-q4_k_m.gguf
# Should be about 4.9 GB
```

---

## Step 4: Start the server

```bash
cd ~/llama.cpp/build

./bin/llama-server \
  -m ~/models/qwen3-8b-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  -c 32768 \
  --ctx-size 32768
```

### What these flags mean

| Flag | Purpose |
|------|---------|
| `-m ~/models/...gguf` | Path to the model file |
| `--host 0.0.0.0` | Listen on all interfaces (so other devices can reach it if needed) |
| `--port 8080` | API server port |
| `-ngl 99` | Offload all layers to Metal GPU (99 = all of them for an 8B model) |
| `-c 32768` | Context window size: 32K tokens |
| `--ctx-size 32768` | Same as -c, explicit context size |

The server exposes an OpenAI-compatible API at `http://localhost:8080/v1`.

### Verify the server is running

```bash
curl http://localhost:8080/v1/models
# Should return JSON with the model name
```

Test a completion:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

---

## Step 5: Create a launch script

So you don't have to remember all those flags:

```bash
cat > ~/start-llm.sh << 'EOF'
#!/bin/bash
# Start local LLM server with Qwen3 8B on Metal GPU
cd ~/llama.cpp/build
./bin/llama-server \
  -m ~/models/qwen3-8b-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -ngl 99 \
  -c 32768
EOF

chmod +x ~/start-llm.sh
```

Now you just run:
```bash
~/start-llm.sh
```

---

## Step 6: Install Hermes Agent

If you haven't already:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

---

## Step 7: Point Hermes at llama.cpp

### Interactive setup

```bash
hermes model
```

Select:
1. **Custom endpoint (self-hosted / VLLM / etc.)**
2. **Base URL**: `http://localhost:8080/v1`
3. **API key**: leave blank
4. **Model name**: `qwen3-8b` (or whatever the server reports — check `curl http://localhost:8080/v1/models`)

### Or edit config.yaml directly

```bash
hermes config edit
```

```yaml
model:
  provider: custom
  base_url: "http://localhost:8080/v1"
  api_key: ""
  default: "qwen3-8b"
  context_length: 32768
```

---

## Step 8: Test the setup

```bash
hermes chat -q "What tools do you have available?"
```

If Hermes responds and can call tools, you're running.

---

## Performance comparison: Ollama vs llama.cpp

On an M2 Mac Mini (16 GB), typical results:

| Metric | Ollama | llama.cpp (compiled) |
|--------|--------|----------------------|
| Tokens/second | ~25–30 | ~30–38 |
| First token latency | ~1.5s | ~1.2s |
| Memory usage | ~7.5 GB | ~6.8 GB |
| Setup time | 5 minutes | 20 minutes |
| Build updates | Automatic (`brew upgrade ollama`) | Manual (`git pull && rebuild`) |

The speed difference is real but not dramatic. Choose Ollama if you want simplicity; choose llama.cpp if you want every last token/second.

---

## Troubleshooting

### Build fails with Metal errors

Make sure Xcode Command Line Tools are installed:
```bash
xcode-select --install
```

If Metal is still not found, check your GPU:
```bash
system_profiler SPDisplaysDataType | grep "Chipset"
```

### "Metal device not found"

This means llama.cpp can't see your GPU. On Apple Silicon this should never happen, but try:
```bash
# Force Metal framework
cmake .. -DGGML_METAL=ON -DGGML_METAL_USE_BMETAL=ON -DGGML_NATIVE=ON
```

### Server starts but responses are very slow

Check if Metal is actually being used:
```bash
# While the server is running, look for GPU activity
sudo powermetrics --samplers gpu_power -n 1
```

If GPU usage is near zero during generation, Metal isn't engaged. Rebuild with `-DGGML_METAL=ON`.

### Out of memory

Same guidance as the Ollama chapter — close memory-heavy apps. You can also reduce context:
```bash
./bin/llama-server -m ~/models/qwen3-8b-q4_k_m.gguf -ngl 99 -c 16384
```

And update Hermes config: `context_length: 16384`.

### Want to update llama.cpp

```bash
cd ~/llama.cpp
git pull
cd build
cmake --build . --config Release -j $(sysctl -n hw.ncpu)
```

---

## What's next

- Read [Why Qwen3 8B](03-model-selection.md) to understand the model choice
- Read [Why 32K context](04-context-length.md) to understand the context window decision
- Bookmark the [llama.cpp docs](https://github.com/ggml-org/llama.cpp/tree/master/docs) for advanced tuning