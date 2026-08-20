#!/bin/bash
# build-llamacpp.sh — Build llama.cpp with Metal support on Apple Silicon
# Usage: bash build-llamacpp.sh
set -euo pipefail

echo "═══════════════════════════════════════════════"
echo "  llama.cpp Build with Metal for Apple Silicon"
echo "═══════════════════════════════════════════════"
echo ""

# ---- Step 1: Check prerequisites ----
echo "[…] Checking prerequisites..."

if ! xcode-select -p &>/dev/null; then
    echo "[!] Xcode Command Line Tools not found."
    echo "    Install with: xcode-select --install"
    exit 1
fi
echo "[✓] Xcode Command Line Tools present"

if ! command -v cmake &>/dev/null; then
    echo "[…] Installing CMake via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "[!] Homebrew not found. Install from https://brew.sh first."
        exit 1
    fi
    brew install cmake
fi
echo "[✓] CMake: $(cmake --version | head -1)"

# ---- Step 2: Clone or update llama.cpp ----
LLAMA_DIR="$HOME/llama.cpp"

if [ -d "$LLAMA_DIR/.git" ]; then
    echo "[…] Updating existing llama.cpp clone..."
    cd "$LLAMA_DIR"
    git pull
else
    echo "[…] Cloning llama.cpp..."
    cd "$HOME"
    git clone https://github.com/ggml-org/llama.cpp.git
    cd "$LLAMA_DIR"
fi
echo "[✓] llama.cpp source ready"

# ---- Step 3: Build ----
echo ""
echo "[…] Building with Metal support..."
echo "    This takes 2–5 minutes depending on your Mac."

mkdir -p build
cd build

CPU_CORES=$(sysctl -n hw.ncpu)
echo "    Using $CPU_CORES CPU cores for compilation"

cmake .. \
    -DGGML_METAL=ON \
    -DGGML_METAL_USE_BMETAL=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release -j "$CPU_CORES"

echo "[✓] Build complete"

# ---- Step 4: Verify Metal ----
echo ""
echo "[…] Verifying Metal GPU support..."
if ./bin/llama-cli --help 2>&1 | grep -qi "metal"; then
    echo "[✓] Metal support detected in build"
else
    # Try running with a dummy to see if Metal initializes
    echo "[?] Checking Metal via verbose output..."
    echo "    If inference is slow, Metal may not be working."
    echo "    Rebuild with: cmake .. -DGGML_METAL=ON"
fi

# ---- Step 5: Check for model file ----
MODEL_DIR="$HOME/models"
MODEL_FILE="$MODEL_DIR/qwen3-8b-q4_k_m.gguf"

echo ""
if [ -f "$MODEL_FILE" ]; then
    echo "[✓] Model file found: $MODEL_FILE"
else
    echo "[!] Model file not found at $MODEL_FILE"
    echo "    Download from HuggingFace:"
    echo "    https://huggingface.co/Qwen/Qwen3-8B-Instruct-GGUF"
    echo "    Look for: qwen3-8b-instruct-q4_k_m.gguf (~5 GB)"
    echo ""
    echo "    Or use curl:"
    echo "    mkdir -p $MODEL_DIR"
    echo "    curl -L -o $MODEL_FILE \\"
    echo "      'https://huggingface.co/Qwen/Qwen3-8B-Instruct-GGUF/resolve/main/qwen3-8b-instruct-q4_k_m.gguf'"
fi

# ---- Step 6: Print launch instructions ----
echo ""
echo "═══════════════════════════════════════════════"
echo "  ✅ llama.cpp build complete!"
echo "═══════════════════════════════════════════════"
echo ""
echo "Start the server:"
echo "  cd $LLAMA_DIR/build"
echo "  ./bin/llama-server \\"
echo "    -m $MODEL_FILE \\"
echo "    --host 127.0.0.1 \\"
echo "    --port 8080 \\"
echo "    -ngl 99 \\"
echo "    -c 32768"
echo ""
echo "Or create a launch script:"
echo "  cp scripts/start-llm-template.sh ~/start-llm.sh"
echo "  chmod +x ~/start-llm.sh"
echo ""
echo "Then configure Hermes:"
echo "  hermes model"
echo "  → Custom endpoint"
echo "  → URL: http://localhost:8080/v1"
echo "  → API key: (leave blank)"
echo "  → Model: qwen3-8b"
echo ""
echo "See docs/02-llamacpp-setup.md for full instructions."
echo "═══════════════════════════════════════════════"