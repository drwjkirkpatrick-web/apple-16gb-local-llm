# Chapter 1: Ollama Setup (Easy Path)

The fastest way to get a local LLM running on a 16 GB Apple Silicon Mac. Three commands, five minutes, no compiling.

---

## Why Ollama

Ollama is a background service that manages local model files, handles Metal GPU acceleration automatically, and exposes an OpenAI-compatible API. It does for local LLMs what Docker did for containers — wraps the complexity in a clean CLI.

For a 16 GB Mac, Ollama is the right default choice because:
- **Zero build step** — pre-compiled binary, Metal support built in
- **Automatic memory management** — loads/unloads models as needed
- **OpenAI-compatible API** — Hermes can talk to it directly, no adapter needed
- **Model versioning** — creates custom model variants (like our 32K context config) with a one-liner

---

## Step 1: Install Ollama

### Option A: Homebrew (recommended)

```bash
brew install ollama
```

### Option B: Direct download

Download the installer from [ollama.com](https://ollama.com/download/mac) and run it.

### Verify installation

```bash
ollama --version
# Should print something like: ollama version 0.x.x
```

---

## Step 2: Start the Ollama service

```bash
# Start as background service (persists across reboots)
brew services start ollama

# Or run manually in a terminal (stops when you close the window)
ollama serve
```

Ollama listens on `http://localhost:11434` by default.

### Verify the service is running

```bash
curl http://localhost:11434/api/version
# Should return: {"version":"0.x.x"}
```

---

## Step 3: Pull Qwen3 8B

```bash
ollama pull qwen3:8b
```

This downloads the Q4_K_M quantized version (~5 GB). On a decent connection it takes 5–10 minutes.

The model file is stored at `~/.ollama/models/`. You can see all downloaded models with:

```bash
ollama list
```

---

## Step 4: Create a 32K context variant

The default Qwen3 8B in Ollama ships with 4096 token context. That's far too small for Hermes — the system prompt alone is 2–4K tokens. We need 32K.

Ollama lets you create a custom model variant with different parameters:

```bash
ollama create qwen3-8b-32k -f - <<'EOF'
FROM qwen3:8b
PARAMETER num_ctx 32768
EOF
```

This creates a new model entry called `qwen3-8b-32k` that's identical to `qwen3:8b` but with 32K context. It doesn't duplicate the 5 GB model file — it just stores the parameter override.

### Verify it works

```bash
ollama run qwen3-8b-32k "Hello, what model are you?"
```

You should see a response from Qwen3. Press Ctrl+D to exit the chat.

---

## Step 5: Install Hermes Agent

If you haven't already:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Verify:

```bash
hermes --version
```

---

## Step 6: Point Hermes at Ollama

### Interactive setup

```bash
hermes model
```

Select these options:
1. **Custom endpoint (self-hosted / VLLM / etc.)**
2. **Base URL**: `http://localhost:11434/v1`
3. **API key**: leave blank (Ollama doesn't need one)
4. **Model name**: `qwen3-8b-32k`

### Or edit config.yaml directly

```bash
hermes config edit
```

Find the `model:` section and set:

```yaml
model:
  provider: custom
  base_url: "http://localhost:11434/v1"
  api_key: ""
  default: "qwen3-8b-32k"
  context_length: 32768
```

---

## Step 7: Test the setup

```bash
# Quick single-query test
hermes chat -q "What tools do you have available?"

# Start interactive chat
hermes chat
```

If Hermes responds and can call tools (like reading a file or running a command), you're done.

---

## Step 8: Set context length in Hermes config

Even though Ollama is configured for 32K, Hermes also needs to know:

```bash
hermes config edit
```

Ensure this is set:

```yaml
model:
  context_length: 32768
```

This tells Hermes to start compressing context at ~16K tokens (50% of 32K), which keeps memory usage safe.

---

## Troubleshooting

### "Connection refused" from Hermes

Ollama isn't running. Start it:
```bash
ollama serve
# or
brew services start ollama
```

### Model loads slowly / Mac feels sluggish

The 5 GB model is loading into memory. First request after idle is always slow. Ollama keeps the model loaded for 5 minutes by default after last use. To keep it loaded longer:

```bash
ollama create qwen3-8b-32k -f - <<'EOF'
FROM qwen3:8b
PARAMETER num_ctx 32768
PARAMETER keep_alive 30m
EOF
```

### Out of memory errors

If macOS is swapping heavily or Ollama crashes:
1. Close memory-heavy apps (browsers with many tabs, Docker, etc.)
2. Reduce context to 16K:
   ```bash
   ollama create qwen3-8b-16k -f - <<'EOF'
   FROM qwen3:8b
   PARAMETER num_ctx 16384
   EOF
   ```
3. Update Hermes config: `context_length: 16384`

### Hermes says model not found

Make sure you're using the exact model name:
```bash
ollama list
# Should show: qwen3-8b-32k
```

If the name doesn't match what you entered in Hermes config, update the config.

---

## What's next

- Read [Chapter 2](02-llamacpp-setup.md) if you want ~15–20% faster inference by compiling llama.cpp from source
- Read [Why Qwen3 8B](03-model-selection.md) to understand the model choice
- Read [Why 32K context](04-context-length.md) to understand the context window decision