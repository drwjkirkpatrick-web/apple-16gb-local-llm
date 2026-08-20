# Why 32K Context

The context window decision for a 16 GB Mac is a tight optimization problem. Here's the full analysis.

---

## What context window means

The context window is how many tokens (roughly word-pieces) the model can "see" at once. This includes:

| Component | Approximate tokens |
|-----------|-------------------|
| System prompt (Hermes instructions) | 2,000–4,000 |
| User profile + memory injection | 500–2,000 |
| One loaded skill | 1,000–5,000 |
| Conversation history | Grows over time |
| Tool results (web pages, file contents, command output) | Variable — can be large |
| Model's response | Being generated |

**At session start**, before you've even said anything, Hermes has already consumed 5–10K tokens of context. Every tool call adds more.

---

## The memory cost of context

The KV cache is the memory the model uses to store its "working memory" — the tokens it's seen so far in the current context. It grows linearly with context length.

For Qwen3 8B (32 layers, GQA with 8 KV heads, 128 head dim):

| Context length | FP16 KV cache | Q8 KV cache |
|----------------|---------------|-------------|
| 4K | ~0.5 GB | ~0.25 GB |
| 8K | ~1 GB | ~0.5 GB |
| 16K | ~2 GB | ~1 GB |
| **32K** | **~4 GB** | **~2 GB** |
| 64K | ~8 GB | ~4 GB |
| 128K | ~16 GB | ~8 GB |

After the ~5 GB model weights, we have ~3–5 GB for KV cache on a 16 GB Mac.

---

## Why 32K, not 64K or 128K

### 128K is impossible

128K FP16 KV cache = ~16 GB. That's the entire machine's RAM. Even at Q8 KV cache (~8 GB), combined with 5 GB weights = 13 GB, leaving 3 GB for macOS — which needs 6–8 GB. **The Mac would swap to SSD constantly, making everything slow.**

### 64K is risky

64K Q8 KV cache = ~4 GB. Plus 5 GB weights = 9 GB. That leaves 7 GB for macOS + Hermes + Python + terminal + any other apps. This is tight enough that:

- Opening a browser with several tabs could push you into swap
- Hermes's own Python process (1–2 GB) eats into the headroom
- If the model processes a large tool result (like a web page extraction), the spike can cause OOM

**64K works on paper, fails in practice on 16 GB.** You'll spend more time managing memory than using the agent.

### 32K is the sweet spot

32K Q8 KV cache = ~2 GB. Plus 5 GB weights = 7 GB. That leaves **~9 GB** for macOS, Hermes, and other apps. Comfortable margin.

- macOS gets its 6–8 GB
- Hermes + Python gets 1–2 GB
- Browser, terminal, etc. fit in the remaining space
- **No swap, no OOM, no crashes**

---

## Is 32K enough for Hermes?

### What fits in 32K tokens

32K tokens is approximately:
- ~24,000 words of text
- ~480 lines of code (at ~50 chars/line)
- ~60 pages of a book
- A medium-length technical article or two

For most agent interactions, this is plenty:

| Scenario | Context used | Fits 32K? |
|----------|-------------|-----------|
| Simple Q&A | 5–8K | ✅ Easily |
| Read a file + discuss it | 8–15K | ✅ Yes |
| Web search + summarize results | 10–20K | ✅ Yes |
| Multi-turn conversation (10–15 exchanges) | 15–25K | ✅ Yes |
| Read a large code file + modify it | 20–28K | ⚠️ Getting tight |
| Analyze an entire codebase | 30K+ | ❌ No |

### Hermes has compression

Hermes includes automatic context compression (configured in `config.yaml`):

```yaml
compression:
  enabled: true
  threshold: 0.50    # Compress when context hits 50% of limit
  target_ratio: 0.20  # Compress to 20% of the threshold
```

At 32K context:
- Compression triggers at ~16K tokens of actual conversation
- It summarizes older messages down to ~20% of their size
- This gives you effectively much more than 32K of "wall clock conversation"

For a 16 GB Mac, compression is your friend — it keeps memory usage bounded.

---

## Why not just use the model's max (128K)?

There's a second reason beyond memory: **small models degrade at long context.**

An 8B model physically can attend to 128K tokens, but research shows that small models lose track of information in long contexts:

- At 4K: full performance
- At 16K: ~95% performance
- At 32K: ~85% performance
- At 64K: ~70% performance
- At 128K: ~50% performance (or worse)

This is sometimes called the "lost in the middle" problem — the model remembers the beginning and end of context but forgets the middle.

**32K is the point where an 8B model is still performing well.** Pushing to 64K or 128K on a small model gives you more context capacity but less effective use of it. You'd have more tokens that the model is worse at using.

---

## How to set 32K

### In Ollama

```bash
ollama create qwen3-8b-32k -f - <<'EOF'
FROM qwen3:8b
PARAMETER num_ctx 32768
EOF
```

### In llama.cpp

```bash
./bin/llama-server -m ~/models/qwen3-8b-q4_k_m.gguf -c 32768 -ngl 99
```

### In Hermes config

```yaml
model:
  context_length: 32768
```

All three must agree. If Ollama says 128K but Hermes says 32K, Hermes will start compressing at 16K (50% of 32K) even though the model could hold more — this is actually fine and conservative. If Hermes says 128K but Ollama says 32K, the model will error when context exceeds 32K.

**Match all three to 32K for consistency.**

---

## When to reduce to 16K

If the 16 GB Mac is also running other memory-intensive apps (Docker, large IDE, many browser tabs), you may need to drop to 16K context:

| Context | Q8 KV cache | Total (weights + KV) | Free for OS |
|---------|-------------|---------------------|-------------|
| 16K | ~1 GB | ~6 GB | ~10 GB |
| 32K | ~2 GB | ~7 GB | ~9 GB |
| 64K | ~4 GB | ~9 GB | ~7 GB |

16K is the safe fallback. You lose some conversation depth but gain stability.

To switch:
- Ollama: `PARAMETER num_ctx 16384`
- llama.cpp: `-c 16384`
- Hermes: `context_length: 16384`

---

## Summary

| | 16K | **32K** | 64K | 128K |
|---|---|---|---|---|
| KV cache (Q8) | ~1 GB | **~2 GB** | ~4 GB | ~8 GB |
| Total with weights | ~6 GB | **~7 GB** | ~9 GB | ~13 GB |
| Free for macOS | ~10 GB | **~9 GB** | ~7 GB | ~3 GB |
| Model quality at this length | Good | **Good** | Degraded | Poor |
| Hermes compression trigger | ~8K | **~16K** | ~32K | ~64K |
| Risk of swap/OOM | None | **None** | Moderate | High |
| **Verdict** | Safe fallback | **Recommended** | Risky | Don't |

**32K is the sweet spot: maximum context that fits comfortably in 16 GB with no performance risk, and the maximum at which an 8B model still performs well.**