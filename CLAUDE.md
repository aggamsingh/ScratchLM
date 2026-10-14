# CLAUDE.md — ScratchLM

> Project guide for Claude Code. Read this fully before doing any work in this repo.

---

## 1. What this project is

**ScratchLM** is a learning-driven project: I (Aggam) am building the full LLM training pipeline myself to understand how language models are actually trained, and to practise implementing each stage correctly.

It has two parts:

1. **Part 1 — Pretraining from scratch.** Write a byte-level BPE tokenizer, a GPT-style transformer and a training loop, then pretrain a ~12M-parameter model on **TinyStories** until it writes short, coherent, *new* stories.
2. **Part 2 — Post-training a real model.** Take **Qwen2.5-0.5B (base)** and improve it on **GSM8K** math word problems with three methods I implement myself: **SFT** (with prompt-loss masking), **LoRA**, and **DPO** (with self-generated preference pairs). Measure every stage on the same test set.

Final deliverables: a results table, parity tests against reference libraries, a novelty check for generated stories, a Gradio demo on Hugging Face Spaces, and a README write-up of what I learned.

**Part 2 is the headline of the project.** Part 1 is the foundation — keep it tight and don't over-polish it.

---

## 2. How Claude should work with me

This is a **learning project**. The goal is that I understand every line of the core components, not just that the code exists.

- **Explain before implementing.** For the core from-scratch components (§4), give a short explanation of the approach and the key math before writing code.
- **When I say "I'll write it" or "guide me"**: give the structure, function signatures, and hints — not the full implementation. Review my code when I paste it.
- **When I ask you to implement**: write clean, readable code with comments on the non-obvious lines (shapes, masking, scaling factors, numerical-stability tricks).
- **Always annotate tensor shapes** in comments for attention, loss and LoRA code, e.g. `# (B, T, n_head, head_dim)`.
- **Never silently swap a from-scratch component for a library call.** If a library would be easier, say so and let me decide.
- **Flag bugs and wrong assumptions directly**, including in my code. Don't soften real problems.
- **Prefer small, testable steps.** Every new component ships with a test.
- **Don't invent results.** Never write numbers into the README or results table that didn't come from an actual run.

---

## 3. Git rules (important)

- **I am the author of every commit:** `Aggam Singh Arora <aggamsingh1@gmail.com>`.
- **Never** use a Claude author identity.
- **Never** add `Co-Authored-By: Claude`, `Claude-Session`, or any similar trailer to commit messages.
- Commit messages: short imperative subject (`Add LoRALinear with zero-init B`), optional body explaining *why*.
- Never commit checkpoints, datasets, tokenized `.bin` files, or API keys. Keep them in `.gitignore`.

---

## 4. What must be implemented from scratch vs. what may use libraries

**From scratch (no library shortcuts):**

| Component | File |
|---|---|
| Byte-level BPE tokenizer (train, encode, decode) | `scratchlm/tokenizer/bpe.py` |
| GPT transformer (attention, MLP, blocks, generation) | `scratchlm/model/gpt.py` |
| Pretraining loop (AdamW setup, LR schedule, AMP, accumulation, checkpointing) | `scratchlm/pretrain/train.py` |
| SFT loop with prompt-loss masking | `scratchlm/posttrain/sft.py` |
| LoRA module and injection | `scratchlm/posttrain/lora.py` |
| DPO loss and training loop | `scratchlm/posttrain/dpo.py` |
| GSM8K answer extraction and evaluation | `scratchlm/eval/gsm8k.py` |
| Novelty (n-gram overlap) check | `scratchlm/eval/novelty.py` |

**Allowed libraries:**
- `torch` (core tensors, `nn.Module`, optimizers, AMP). `torch.optim.AdamW` is fine — the point is the loop, not the optimizer internals.
- `transformers` — **only** to load Qwen2.5-0.5B weights and its tokenizer.
- `datasets` — to download TinyStories and GSM8K.
- `peft`, `trl`, `tokenizers` — **only inside `tests/`** as references for parity checks, and for baselines if explicitly needed. Never in the main code path.
- `gradio` for the demo; `numpy`, `regex`, `tqdm`, `matplotlib`, `pyyaml`, `pytest`.

---

## 5. Repository structure

```
scratchlm/
├── CLAUDE.md
├── README.md                     # results table + learnings write-up (fill with REAL numbers only)
├── requirements.txt              # pin exact versions after first working install
├── .gitignore                    # data/, checkpoints/, *.bin, *.pt, .env, wandb/
├── configs/
│   ├── pretrain_tiny.yaml
│   ├── sft_full.yaml
│   ├── sft_lora.yaml
│   └── dpo_lora.yaml
├── scratchlm/
│   ├── __init__.py
│   ├── utils/
│   │   ├── config.py             # dataclass configs loaded from YAML
│   │   ├── seed.py               # set_seed(): python, numpy, torch, cuda
│   │   ├── logging.py            # CSV logger (+ optional wandb)
│   │   └── checkpoint.py         # save/load model, optimizer, scaler, step, RNG state
│   ├── tokenizer/
│   │   └── bpe.py
│   ├── data/
│   │   ├── tinystories.py        # download, train tokenizer on subset, encode → uint16 memmap
│   │   └── gsm8k.py              # load, train/dev split, prompt formatting
│   ├── model/
│   │   └── gpt.py
│   ├── pretrain/
│   │   ├── train.py
│   │   └── sample.py             # generate stories from a checkpoint
│   ├── posttrain/
│   │   ├── sft.py
│   │   ├── lora.py
│   │   ├── make_dpo_pairs.py
│   │   └── dpo.py
│   └── eval/
│       ├── gsm8k.py
│       ├── novelty.py
│       └── bootstrap.py          # 95% bootstrap confidence intervals
├── scripts/                      # thin CLI entry points (python -m scripts.xxx --config ...)
│   ├── train_tokenizer.py
│   ├── pretrain.py
│   ├── run_baselines.py
│   ├── run_sft.py
│   ├── make_pairs.py
│   ├── run_dpo.py
│   ├── evaluate.py
│   └── plot_results.py
├── notebooks/
│   └── kaggle_runner.ipynb       # clones repo, installs deps, runs scripts on Kaggle GPU
├── tests/
│   ├── test_bpe.py
│   ├── test_gpt.py
│   ├── test_sft_masking.py
│   ├── test_lora.py
│   ├── test_dpo_loss.py
│   └── test_gsm8k_extract.py
├── results/                      # committed: small JSON/CSV metrics, plots, story samples
└── demo/
    ├── app.py                    # Gradio app (Stories tab + Math tab)
    └── requirements.txt
```

---

## 6. Part 1 — Pretraining from scratch

### 6.1 Data: TinyStories
- Hugging Face dataset: `roneneldan/TinyStories` (~2M short stories, simple vocabulary).
- Use the official train/validation splits.
- Encode the full corpus once and store as a flat `uint16` numpy memmap (`train.bin`, `val.bin`), with an end-of-text token between stories. Vocab size 4,096 fits in uint16.
- Encoding in pure Python is slow: use `multiprocessing` and cache the result. Never re-encode unnecessarily.

### 6.2 Tokenizer: byte-level BPE (`bpe.py`)
- Base vocabulary: 256 byte tokens. Learn merges until vocab = **4,096** (including special tokens).
- Pre-tokenize with the GPT-2-style regex (via the `regex` package) so merges don't cross word/space boundaries.
- Special tokens: `<|endoftext|>` at minimum.
- **Train the tokenizer on a subset** (e.g. ~50–100 MB of text), not the full corpus — pure-Python BPE training on everything is too slow.
- API:
  ```python
  class BPETokenizer:
      def train(self, text: str, vocab_size: int, verbose: bool = False) -> None: ...
      def encode(self, text: str) -> list[int]: ...
      def decode(self, ids: list[int]) -> str: ...
      def save(self, path: str) -> None: ...
      @classmethod
      def load(cls, path: str) -> "BPETokenizer": ...
  ```
- Done when: exact round-trip on arbitrary text (including unicode and emoji), merges saved/loaded correctly, compression ratio (bytes per token) reported.

### 6.3 Model: GPT (`gpt.py`)
- Config:
  ```yaml
  vocab_size: 4096
  n_layer: 6
  n_head: 6
  n_embd: 384
  block_size: 256
  dropout: 0.1        # can drop to 0.0 if underfitting
  bias: false
  tie_weights: true   # token embedding shared with LM head
  ```
- ≈12M parameters with tied embeddings. Print the exact count at init.
- Components: token embedding + learned positional embedding → N × [pre-LayerNorm → causal multi-head self-attention → residual; pre-LayerNorm → MLP (4× expansion, GELU) → residual] → final LayerNorm → LM head.
- Attention: implement it **manually** (QKᵀ/√d, causal mask, softmax, ×V). Add a flag `use_sdpa` that switches to `F.scaled_dot_product_attention`, and test that both give the same output (`allclose`, tolerance ~1e-5 in fp32).
- Init: normal(0, 0.02); scale residual-projection weights by `1/sqrt(2 * n_layer)`.
- `generate(idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None)` — autoregressive sampling, cropping context to `block_size`.

### 6.4 Training loop (`train.py`)
- Optimizer: AdamW, betas (0.9, 0.95), weight decay 0.1 on 2D weights only (not biases / LayerNorm / embeddings' norms).
- LR: linear warmup (~500 steps) to peak `6e-4`, cosine decay to 10% of peak.
- Gradient clipping at 1.0.
- Mixed precision: **fp16 + `torch.cuda.amp.GradScaler`** on T4/P100 (T4 has no fast bf16). Use bf16 only if the GPU supports it.
- Micro-batch e.g. 64 × 256 tokens; gradient accumulation to an effective ~128k tokens/step. Tune to the GPU's memory.
- Evaluate val loss every N steps on a fixed set of val batches; log train loss, val loss, LR, tokens/sec, grad norm.
- Save checkpoints regularly (model, optimizer, scaler, step, RNG state) and support **resume** — Kaggle sessions end unexpectedly.
- Save story samples at fixed steps (e.g. 500, 2k, 10k, final) with fixed prompts and seeds → `results/samples/`.

### 6.5 Part 1 outputs
- Loss curves plot (train + val).
- Final val loss and perplexity.
- Samples across training steps showing the progression from noise to coherent stories.
- Samples at temperatures 0.5 / 0.8 / 1.0 / 1.2.

### 6.6 Novelty check (`novelty.py`)
- Goal: show generated stories are new, not memorized.
- For each generated story, extract all token 8-grams; check what fraction appear in the training set.
- Implementation: hash each training 8-gram to uint64, store as a **sorted numpy array**, look up with `np.searchsorted`. (A Python `set` of all training 8-grams will not fit in RAM.) If memory is tight, use a large random sample of the training set and say so.
- Report: % of generated 8-grams that never appear in training data, over e.g. 200 samples.

---

## 7. Part 2 — Post-training Qwen2.5-0.5B on GSM8K

### 7.1 Model and data
- Model: `Qwen/Qwen2.5-0.5B` — the **base** model, not Instruct (so improvements are attributable to my training).
- Dataset: `openai/gsm8k`, config `main`. 7,473 train / 1,319 test. Gold answers end with `#### <number>`.
- **Split train into train (≈6,973) + dev (500)** with a fixed seed. All tuning and DPO checkpoint selection use **dev**. The **test set is used only for final reported numbers.** Never tune on test.
- Note: Qwen2.5 base models already have some math ability, and GSM8K may overlap with pretraining data. Measure the baseline first and report gains honestly, whatever their size.

### 7.2 Prompt format (use everywhere — baselines, SFT, DPO, eval)
```
Question: {question}
Answer: {reasoning ... } #### {final_number}<eos>
```
- During SFT, the target is the gold solution text (reasoning + `#### N`).
- Max sequence length 512 tokens; drop or truncate rare longer examples and log how many.

### 7.3 Evaluation (`eval/gsm8k.py`)
- Greedy decoding, `max_new_tokens` ≈ 320, stop at EOS or after the `#### N` line.
- Batched generation with **left padding**.
- Answer extraction: take the number after the last `####`; fallback to the last number in the output. Normalize commas, `$`, trailing `.`, and compare numerically.
- Metric: exact-match accuracy, with a **95% bootstrap CI** (`bootstrap.py`, 1,000 resamples).
- Save per-example predictions to `results/preds/<run>.jsonl` for error analysis.

### 7.4 Baselines (`run_baselines.py`)
1. Base model, zero-shot.
2. Base model, 8-shot (fixed exemplars drawn from the train split, not dev/test).

### 7.5 SFT (`sft.py`)
- Loss on **answer tokens only**: set labels to `-100` for all prompt tokens and padding.
- Two variants:
  - **Full fine-tune**: all weights trainable. On a 16 GB T4, use gradient checkpointing, micro-batch 1–4 with accumulation, fp16 AMP. If it still OOMs, try 8-bit AdamW (bitsandbytes) or document the OOM honestly — that itself shows why LoRA exists.
  - **LoRA**: base frozen, only adapters trainable (see 7.6).
- Starting hyperparameters (tune on dev): epochs 2–3, LR 1e-5 (full) / 2e-4 (LoRA), cosine schedule with warmup ~3%, effective batch 32 sequences.
- Select the checkpoint by **dev accuracy**.

### 7.6 LoRA (`lora.py`)
- `LoRALinear(base: nn.Linear, r: int, alpha: int, dropout: float)`:
  - Freeze `base.weight` (and bias).
  - `A`: shape `(r, in_features)`, Kaiming-uniform init. `B`: shape `(out_features, r)`, **zero init** → at step 0 the output equals the base layer exactly.
  - Forward: `base(x) + dropout(x) @ A.T @ B.T * (alpha / r)`.
  - `merge()` folds `B @ A * scale` into the base weight for fast inference; `unmerge()` reverses it.
  - A way to **disable adapters** (context manager or flag) — needed for the DPO reference model.
- `inject_lora(model, target_modules=["q_proj","k_proj","v_proj","o_proj"], r=16, alpha=32, dropout=0.05)` — replaces matching `nn.Linear` modules in place; returns the count of trainable params.
- Default targets are the attention projections. Ablation option: also `gate_proj`, `up_proj`, `down_proj`.
- Rank ablation: r ∈ {4, 8, 16, 32}, alpha = 2r, everything else fixed.

### 7.7 DPO preference pairs (`make_dpo_pairs.py`)
- Use the **best SFT (LoRA) model**.
- For each question in the train split, sample **k = 4** answers at temperature 0.8, top_p 0.95.
- Grade each with the GSM8K extractor. Keep questions with at least one correct and one incorrect answer.
- Pair: correct → `chosen`, incorrect → `rejected`. Take one pair per question by default (option for more).
- Save to `data/dpo_pairs.jsonl` with fields `prompt, chosen, rejected, question_id`. Log the number of usable pairs.
- Never use dev or test questions to build pairs.

### 7.8 DPO (`dpo.py`)
- Loss (from the DPO paper), per pair:
  `L = -log σ( β · [ (log πθ(y_w|x) − log πref(y_w|x)) − (log πθ(y_l|x) − log πref(y_l|x)) ] )`
  where log-probs are **summed over response tokens only** (prompt masked).
- β = 0.1 (try 0.05 / 0.3 on dev if time).
- Reference model = the SFT model, frozen.
- Memory trick: **merge the SFT LoRA into the base** → this is both the starting policy and the reference. Add a **fresh LoRA** for DPO; computing reference log-probs = running the same model with the DPO adapter **disabled**. One model in memory instead of two.
- Optional speed-up: precompute reference log-probs for all pairs once and cache them.
- Starting hyperparameters: LR 5e-6 to 5e-5 (LoRA), 1–2 epochs, effective batch 32 pairs.
- Log: loss, reward margin (chosen minus rejected implicit reward), and reward accuracy (% of pairs where chosen > rejected).
- Select by dev accuracy. Watch for regressions: DPO can make answers longer or break the `####` format — check dev outputs, not just loss.

### 7.9 Part 2 outputs — the results table
Fill in from real runs only, all on the 1,319-problem test set:

| Model | GSM8K acc (95% CI) | Trainable params | Peak GPU mem | Train time |
|---|---|---|---|---|
| Base, zero-shot | | 0 | | — |
| Base, 8-shot | | 0 | | — |
| SFT, full fine-tune | | ~494M | | |
| SFT, LoRA r=16 | | | | |
| SFT (LoRA) + DPO | | | | |

Plus:
- LoRA rank ablation plot (r vs dev/test accuracy).
- Error analysis: 30–50 failures categorised (arithmetic slip, wrong setup, format failure, truncated output) — compare base vs SFT vs DPO.

---

## 8. Tests (parity tests are a core feature, not optional)

Run with `pytest -q`. Every test must be deterministic (fixed seeds) and runnable on CPU with tiny shapes.

| Test file | What it verifies |
|---|---|
| `test_bpe.py` | Exact encode→decode round-trip (ASCII, unicode, emoji, whitespace); save/load gives identical encodings; special tokens never split. Compare compression ratio to a HF `tokenizers` BPE trained on the same text — similar, not identical (merge tie-breaking differs). |
| `test_gpt.py` | Output shapes; causal mask (changing future tokens doesn't change earlier logits); manual attention == SDPA path; a tiny model can overfit a single batch (loss → near 0). |
| `test_sft_masking.py` | Prompt and padding positions have label −100; my loss equals the HF model's built-in loss when given the same masked labels. |
| `test_lora.py` | With B = 0, LoRA model output == base output; after copying identical A/B weights into a PEFT LoRA layer, outputs and gradients match (`allclose`); merge() then forward == unmerged forward; only A/B have `requires_grad`. |
| `test_dpo_loss.py` | My loss matches TRL's DPO loss on the same log-probs (pin the TRL version — its API changes between releases); if TRL is unavailable, match an independent formula implementation in the test. Sanity: loss = log 2 when policy == reference; loss decreases as the chosen margin increases. |
| `test_gsm8k_extract.py` | Extraction handles `#### 1,234`, `$5`, `7.`, negatives, decimals, missing `####` (fallback), and garbage output. |

---

## 9. Compute and environment

- **GPU:** Kaggle notebooks (T4 ×2 or P100) or Colab free tier. Check current session and weekly quota limits before long runs.
- Run code from the repo via `notebooks/kaggle_runner.ipynb`: clone repo → `pip install -r requirements.txt` → run `scripts/*`.
- Checkpoint every ~15–30 minutes and always support resume.
- Use fp16 AMP on T4/P100. Use gradient checkpointing for full fine-tuning of the 0.5B model.
- Python 3.10+; pin exact package versions in `requirements.txt` once a working setup is confirmed.
- Store large artifacts (checkpoints, merged models) on Kaggle datasets or the Hugging Face Hub — never in git.

---

## 10. Coding standards

- Type hints on public functions; short docstrings stating inputs, outputs and tensor shapes.
- Configs as dataclasses loaded from YAML in `configs/`; no magic numbers buried in code.
- `set_seed()` at the start of every script; record the seed, config and git commit hash in each run's results JSON.
- Scripts are thin wrappers; logic lives in the `scratchlm/` package.
- Each run writes `results/<run_name>/metrics.json` + `config.yaml`.
- Keep functions small and readable over clever — this code is for understanding.

---

## 11. Demo (`demo/app.py`)

- Gradio app with two tabs:
  - **Stories**: prompt box, temperature and top-k sliders → story from my from-scratch GPT.
  - **Math**: question box → answers from Base, SFT, and SFT+DPO **side by side**, with the extracted final answer highlighted.
- Load merged weights (LoRA folded in) from the Hugging Face Hub.
- Deploy to Hugging Face Spaces (free CPU tier is fine; 0.5B on CPU is slow but works — cap `max_new_tokens`).

---

## 12. Milestones and definitions of done

| Week | Milestone | Done when |
|---|---|---|
| 1 | Tokenizer + GPT | `test_bpe.py` and `test_gpt.py` pass; tokenizer trained and saved; corpus encoded to memmap |
| 2 | Pretraining | Full run finished; loss curves, perplexity and step-wise samples in `results/`; novelty check reported |
| 3 | Baselines + SFT + LoRA | Base zero-/8-shot numbers; full-FT and LoRA SFT trained and evaluated on dev; `test_sft_masking.py` and `test_lora.py` pass |
| 4 | DPO | Pairs generated (count logged); DPO trained and selected on dev; `test_dpo_loss.py` passes; final test-set table filled |
| 5 | Ablation + demo + write-up | Rank ablation plot; error analysis; demo live on HF Spaces; README complete |

Don't start a stage until the previous stage's tests pass.

---

## 13. README write-up (final)

The README must contain:
1. One-paragraph summary: what I built and why (learning how LLM training really works).
2. Architecture and pipeline diagram (Part 1 → Part 2).
3. The results table (§7.9) and the Part 1 results (§6.5), real numbers only.
4. Story samples across training steps + novelty result.
5. Rank ablation plot and error analysis.
6. **What I learned / what broke** — real bugs, surprises, and what I'd do differently.
7. How to reproduce (exact commands).
8. Link to the live demo.

---

## 14. Resume target (fill in X/Y/Z only from real results)

**ScratchLM — LLM Training Pipeline from First Principles** · PyTorch, Hugging Face, Gradio
- Implemented SFT, LoRA and DPO from scratch on Qwen2.5-0.5B, verified against PEFT/TRL with unit tests; raised GSM8K accuracy from X% to Y%.
- Generated DPO preference pairs automatically from the model's own errors, adding Z points over SFT alone.
- Showed LoRA (r=16) reaching N% of full fine-tuning accuracy with ~M× fewer trainable parameters; pretrained a 12M-parameter GPT (own BPE tokenizer, transformer and training loop) on TinyStories.s

---

## 15. Roadmap & human-only tasks

The phased plan lives in `docs/roadmap.md`. Keep it updated: tick finished items, record each phase's completion date, and add new 👤 tasks as they come up. Decisions go in `docs/decisions.md`; learning notes and quizzes in `docs/learning/phase<N>.md`.

**👤 Aggam-only tasks. Claude never does these or fakes their results:**
- Creating or configuring accounts (GitHub, Kaggle, Hugging Face), including Kaggle phone verification and GPU enablement.
- Handling secrets (API keys, HF tokens). Tell Aggam to use Kaggle Secrets or a local `.env`. Never ask for a token in chat; never commit one.
- Running long GPU jobs on Kaggle/Colab, and downloading/uploading checkpoints.
- Pushing to the GitHub remote and deploying the HF Space under Aggam's account.
- Inspecting outputs by eye: story samples, loss curves, GSM8K prediction spot-checks.
- The judgment parts of error analysis: Aggam categorises failures first, Claude checks afterwards.
- Answering end-of-component quiz questions in their own words.
- Writing the README "What I learned / what broke" section and their resume bullets in their own words. Claude may edit for clarity but keeps Aggam's voice.
- Approving any decision that changes this plan (model, dataset, metric, scope).

**How to hand off a 👤 task.** Stop and print a block like this:

```
👤 YOUR TURN
Task: ...
Exact steps: ... (numbered, with commands where relevant)
Send me back: ... (e.g. loss plot, last 20 log lines, metrics JSON)
```

Do not continue with dependent work until Aggam has reported back. If Aggam tries to skip a 👤 task, remind them once why it matters, then follow their call.
