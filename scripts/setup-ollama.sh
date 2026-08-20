#!/bin/bash
# setup-ollama.sh — Automated Ollama + Qwen3 8B setup for 16 GB Apple Silicon Mac
# Usage: bash setup-ollama.sh
set -euo pipefail

echo "═══════════════════════════════════════════════"
echo "  Ollama + Qwen3 8B Setup for 16 GB Apple Silicon"
echo "═══════════════════════════════════════════════"
echo ""

# ---- Step 1: Check if Ollama is installed ----
if command -v ollama &>/dev/null; then
    echo "[✓] Ollama already installed: $(ollama --version)"
else
    echo "[…] Installing Ollama via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "[!] Homebrew not found. Install from https://brew.sh first."
        exit 1
    fi
    brew install ollama
    echo "[✓] Ollama installed: $(ollama --version)"
fi

# ---- Step 2: Start Ollama service ----
echo ""
echo "[…] Starting Ollama service..."
# Check if already running
if curl -s http://localhost:11434/api/version &>/dev/null; then
    echo "[✓] Ollama service already running"
else
    brew services start ollama 2>/dev/null || ollama serve &
    sleep 3
    if curl -s http://localhost:11434/api/version &>/dev/null; then
        echo "[✓] Ollama service started"
    else
        echo "[!] Ollama service failed to start. Try: ollama serve"
        exit 1
    fi
fi

# ---- Step 3: Pull Qwen3 8B ----
echo ""
echo "[…] Pulling Qwen3 8B (Q4_K_M, ~5 GB)..."
echo "    This will take 5–10 minutes depending on your connection."
ollama pull qwen3:8b
echo "[✓] Qwen3 8B downloaded"

# ---- Step 4: Create 32K context variant ----
echo ""
echo "[…] Creating qwen3-8b-32k variant (32K context)..."
ollama create qwen3-8b-32k -f - <<'EOF'
FROM qwen3:8b
PARAMETER num_ctx 32768
PARAMETER keep_alive 10m
EOF
echo "[✓] qwen3-8b-32k created"

# ---- Step 5: Verify ----
echo ""
echo "[…] Verifying setup..."
OLLAMA_MODELS=$(ollama list 2>/dev/null | grep qwen3)
if echo "$OLLAMA_MODELS" | grep -q "qwen3-8b-32k"; then
    echo "[✓] Model variant confirmed in Ollama"
else
    echo "[!] Warning: qwen3-8b-32k not found in model list"
fi

# ---- Step 6: Print Hermes config instructions ----
echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ Ollama setup complete!"
echo "═══════════════════════════════════════════════"
echo ""
echo "Next: configure Hermes Agent to use this model."
echo ""
echo "Option A — interactive:"
echo "  hermes model"
echo "  → Custom endpoint"
echo "  → URL: http://localhost:11434/v1"
echo "  → API key: (leave blank)"
echo "  → Model: qwen3-8b-32k"
echo ""
echo "Option B — edit config.yaml directly:"
echo "  hermes config edit"
echo ""
echo "  model:"
echo "    provider: custom"
echo "    base_url: \"http://localhost:11434/v1\""
echo "    api_key: \"\""
echo "    default: \"qwen3-8b-32k\""
echo "    context_length: 32768"
echo ""
echo "Then test:"
echo "  hermes chat -q \"What tools do you have available?\""
echo ""
echo "See docs/01-ollama-setup.md for full instructions."
echo "═══════════════════════════════════════════════"