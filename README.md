# Local LLM for Hermes Agent on a 16 GB Apple Silicon Mac

A practical guide to running a fully local, private AI agent on a 16 GB Apple Silicon iMac or Mac Mini using **Qwen3 8B** with **Hermes Agent**.

No API keys. No cloud. No data leaving the machine.

---

## Why this exists

Most "run a local LLM" guides assume you have 32+ GB of RAM and recommend models that won't fit on a 16 GB Mac. This guide is specifically for the tight 16 GB budget — the minimum viable configuration for a real Hermes Agent setup that's actually usable day-to-day.

---

## What you get

- A **private AI agent** running entirely on your Mac
- **Hermes Agent** with tool calling (terminal, file access, web search, skills, memory)
- **25–35 tokens/second** generation speed (Apple Silicon Metal acceleration)
- **32K context window** — enough for real agent workflows without swapping
- No API costs, no rate limits, no data sent to third parties

---

## The setup

This repo provides two paths to the same destination:

| Path | Difficulty | Speed | Best for |
|------|-----------|-------|----------|
| **Chapter 1: Ollama** | Easy (3 commands) | Good | Most users — fastest setup |
| **Chapter 2: llama.cpp** | Intermediate (compile from source) | ~15–20% faster | Tinkerers who want max performance |

Both paths end with Hermes Agent talking to a local Qwen3 8B model. Pick one.

---

## Quick start (Ollama path — 5 minutes)

```bash
# 1. Install Ollama
brew install ollama
ollama serve &

# 2. Pull the model and create a 32K context variant
ollama pull qwen3:8b
ollama create qwen3-8b-32k -f - <<'EOF'
FROM qwen3:8b
PARAMETER num_ctx 32768
EOF

# 3. Install Hermes Agent
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# 4. Point Hermes at Ollama
hermes model
# Select "Custom endpoint"
# URL: http://localhost:11434/v1
# API key: (leave blank)
# Model: qwen3-8b-32k

# 5. Test it
hermes chat -q "What tools do you have available?"
```

Done. You have a local AI agent.

---

## Repository contents

```
.
├── README.md                          ← You are here
├── docs/
│   ├── 01-ollama-setup.md              ← Chapter 1: Ollama (easy path)
│   ├── 02-llamacpp-setup.md            ← Chapter 2: llama.cpp (fast path)
│   ├── 03-model-selection.md           ← Why Qwen3 8B
│   └── 04-context-length.md            ← Why 32K context
├── scripts/
│   ├── setup-ollama.sh                 ← Automated Ollama setup
│   ├── build-llamacpp.sh               ← Automated llama.cpp build
│   ├── hermes-ollama-config.yaml       ← Hermes config snippet for Ollama
│   └── hermes-llamacpp-config.yaml     ← Hermes config snippet for llama.cpp
└── LICENSE
```

---

## System requirements

- **Mac**: Any Apple Silicon Mac (M1, M2, M3, M4) with **16 GB RAM**
- **macOS**: 13.0 (Ventura) or newer
- **Disk space**: ~6 GB for the model + ~2 GB for tools
- **Xcode Command Line Tools** (for llama.cpp path): `xcode-select --install`

---

## What this setup is good at

- ✅ Chat and Q&A
- ✅ Simple tool calls (terminal commands, file reads/writes)
- ✅ Web search and content extraction
- ✅ Learning how Hermes Agent works
- ✅ Private, offline, no-API-key AI assistance

## What this setup struggles with

- ⚠️ Complex multi-step agent workflows (the 8B model loses the plot on long reasoning chains)
- ⚠️ Very long conversations (context compression kicks in frequently)
- ⚠️ Heavy code generation (works, but a 32B+ model is noticeably better)
- ⚠️ Running other memory-intensive apps simultaneously (browser with many tabs, etc.)

---

## Upgrading later

If you outgrow the 8B model, the upgrade path is straightforward:

| Mac RAM | Recommended model | Quantization | Context |
|---------|-------------------|--------------|---------|
| 16 GB | Qwen3 8B | Q4_K_M | 32K |
| 32 GB | Qwen3 32B | Q4_K_M | 128K |
| 48 GB | Qwen3 32B | Q8_0 | 64K |
| 64 GB+ | Llama 3.3 70B | Q4_K_M | 128K |

Swap the model file in Ollama (`ollama pull qwen3:32b`) or llama.cpp, update the Hermes model name, and you're done. Hermes itself doesn't change.

---

## Credits

Built by [Walker Kirkpatrick](https://github.com/drwjkirkpatrick-web) with assistance from [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research.

Model: [Qwen3 8B](https://qwenlm.github.io/) by Alibaba Qwen Team.

## License

MIT — see [LICENSE](LICENSE).