#!/usr/bin/env bash
# =============================================================================
#  setup-ollama.sh — Ollama + Qwen3 8B at 64K context, q8_0 KV cache
# =============================================================================
#  Hermes Agent requires at least 64,000 tokens of context (hard constant:
#  MINIMUM_CONTEXT_LENGTH = 64_000 in agent/model_metadata.py). So 64K is the
#  floor, not a luxury — and 64K only fits on a 16 GB Mac with an 8-bit KV cache:
#
#      weights (Q4_K_M)                 ~5.0 GB
#      KV cache q8_0 @ 65,536 tokens     4.78 GiB      (9.00 GiB at f16)
#      ----------------------------------------------
#      total                            ~9.8 GiB of 16 GB
#
#  Two separate settings are required, and Ollama users usually miss the second:
#
#      1. per-model:  PARAMETER num_ctx 65536
#      2. server-wide: OLLAMA_FLASH_ATTENTION=1 + OLLAMA_KV_CACHE_TYPE=q8_0
#
#  Usage:  bash setup-ollama.sh
#          SKIP_ENV_WRITE=1 bash setup-ollama.sh    # don't touch shell profile
# =============================================================================
set -euo pipefail

CTX=65536
BASE_MODEL="qwen3:8b"
MODEL_NAME="qwen3-8b-64k"
SKIP_ENV_WRITE="${SKIP_ENV_WRITE:-0}"
MARKER_BEGIN="# >>> hermes-local-llm 64K KV cache >>>"
MARKER_END="# <<< hermes-local-llm 64K KV cache <<<"

ok()   { printf '[✓] %s\n' "$*"; }
info() { printf '[…] %s\n' "$*"; }
warn() { printf '[!] %s\n' "$*"; }
die()  { printf '[✗] %s\n' "$*" >&2; exit 1; }

printf '%s\n' "═══════════════════════════════════════════════════════════════"
printf '%s\n' "  Ollama + Qwen3 8B @ ${CTX} context (q8_0 KV cache)"
printf '%s\n' "═══════════════════════════════════════════════════════════════"

# ---- 1. Ollama installed? --------------------------------------------------
if command -v ollama >/dev/null 2>&1; then
    ok "Ollama present: $(ollama --version 2>/dev/null | head -1)"
else
    info "Installing Ollama via Homebrew..."
    command -v brew >/dev/null 2>&1 || die "Homebrew not found. Install from https://brew.sh first."
    brew install ollama
    ok "Ollama installed: $(ollama --version 2>/dev/null | head -1)"
fi

# ---- 2. KV cache quantization env ------------------------------------------
# Ollama's KV cache type is a *server* setting, not a modelfile parameter.
info "Configuring server-side KV cache quantization (q8_0 + flash attention)..."

export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_CONTEXT_LENGTH="$CTX"

if [ "$SKIP_ENV_WRITE" = "1" ]; then
    warn "SKIP_ENV_WRITE=1 — not touching your shell profile."
    warn "You must export these yourself before 'ollama serve':"
    warn "    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=${CTX}"
else
    PROFILE="$HOME/.zshrc"
    [ -f "$HOME/.zshrc" ] || PROFILE="$HOME/.bash_profile"
    if [ -f "$PROFILE" ] && grep -qF "$MARKER_BEGIN" "$PROFILE"; then
        ok "Env block already present in $PROFILE"
    else
        {
            printf '\n%s\n' "$MARKER_BEGIN"
            printf '# Required so Ollama can hold 64K context in 16 GB of RAM.\n'
            printf 'export OLLAMA_FLASH_ATTENTION=1\n'
            printf 'export OLLAMA_KV_CACHE_TYPE=q8_0\n'
            printf 'export OLLAMA_CONTEXT_LENGTH=%s\n' "$CTX"
            printf '%s\n' "$MARKER_END"
        } >>"$PROFILE"
        ok "Appended env block to $PROFILE (open a new shell to pick it up)"
    fi
fi

# launchd does not read shell profiles, so brew services / the Ollama.app
# GUI would otherwise start WITHOUT these variables.
for kv in OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 "OLLAMA_CONTEXT_LENGTH=$CTX"; do
    launchctl setenv "${kv%%=*}" "${kv#*=}" 2>/dev/null && ok "launchctl setenv ${kv%%=*} ${kv#*=}"
done

# ---- 3. Restart the service so the env actually applies --------------------
info "Restarting Ollama service..."
if brew services list 2>/dev/null | grep -q '^ollama'; then
    brew services restart ollama >/dev/null 2>&1 || true
else
    pkill -f '[o]llama serve' 2>/dev/null || true
    sleep 1
    nohup ollama serve >/tmp/ollama.log 2>&1 &
fi

for _ in $(seq 1 20); do
    curl -sf http://localhost:11434/api/version >/dev/null 2>&1 && break
    sleep 1
done
if curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
    ok "Ollama service responding on :11434"
else
    die "Ollama did not come up. Start it manually: OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=${CTX} ollama serve"
fi

# ---- 4. Model + 64K variant ------------------------------------------------
info "Pulling ${BASE_MODEL} (Q4_K_M, ~5 GB) — this takes a few minutes..."
ollama pull "$BASE_MODEL"
ok "${BASE_MODEL} downloaded"

info "Creating ${MODEL_NAME} at ${CTX} context..."
ollama create "$MODEL_NAME" -f - <<EOF
FROM ${BASE_MODEL}
PARAMETER num_ctx ${CTX}
PARAMETER keep_alive 30m
EOF
ok "${MODEL_NAME} created"

# ---- 5. Verify the variant really carries 64K ------------------------------
MODELFILE="$(ollama show "$MODEL_NAME" --modelfile 2>/dev/null || true)"
if printf '%s' "$MODELFILE" | grep -qiE "num_ctx[[:space:]]+${CTX}"; then
    ok "Modelfile confirms num_ctx ${CTX}"
else
    warn "Could not confirm num_ctx ${CTX} in the modelfile. Got:"
    printf '%s\n' "$MODELFILE" | grep -i num_ctx || printf '      (no num_ctx line found)\n'
fi

# ---- 6. Verify the RUNNING context (env applied, memory fits) --------------
info "Loading the model to check the live context window..."
curl -sf http://localhost:11434/api/generate \
    -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_ctx\":${CTX}}}" \
    >/dev/null 2>&1 || true

LIVE="$(curl -sf http://localhost:11434/api/ps 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for m in d.get("models", []):
    ctx = m.get("context_length") or 0
    size = m.get("size") or 0
    print(f"{ctx} {size/1024**3:.2f}")
    break
' 2>/dev/null || true)"

if [ -n "$LIVE" ]; then
    LIVE_CTX="${LIVE%% *}"; LIVE_GB="${LIVE##* }"
    if [ "${LIVE_CTX:-0}" -ge 64000 ]; then
        ok "Live context: ${LIVE_CTX} tokens, ${LIVE_GB} GiB resident — above the Hermes floor"
    else
        warn "Live context is ${LIVE_CTX} tokens, below the Hermes floor of 64000."
        warn "The KV cache env vars probably did not reach the Ollama server."
        warn "Most reliable fix — run the server yourself in a terminal:"
        warn "    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=${CTX} ollama serve"
    fi
else
    warn "Could not read /api/ps. Check by hand:"
    warn "    ollama ps          # CONTEXT column should read ${CTX}"
fi

# ---- 7. Hand off to Hermes -------------------------------------------------
cat <<EOF

═══════════════════════════════════════════════════════════════
  ✅ Ollama side is set up — now point Hermes at it
═══════════════════════════════════════════════════════════════

Option A — interactive:
  hermes model   → Custom endpoint (self-hosted / vLLM / llama.cpp)
      Base URL:       http://127.0.0.1:11434/v1
      API key:        (leave blank)
      Model:          ${MODEL_NAME}
      Context length: ${CTX}     ← must be ≥ 64000

Option B — ~/.hermes/config.yaml:
      model:
        provider: custom
        base_url: "http://127.0.0.1:11434/v1"
        api_key: ""
        default: "${MODEL_NAME}"
        context_length: ${CTX}

Test it:
  hermes chat -q "What tools do you have available?"

Full walkthrough: docs/01-ollama-setup.md
═══════════════════════════════════════════════════════════════
EOF
