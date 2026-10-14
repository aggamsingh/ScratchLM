# ScratchLM Roadmap

Legend: 🤖 = Claude does it · 👤 = Aggam only (Claude never does it or fakes its result) · ✅ = exit criteria · ⏱ = rough estimate
Current phase: **Phase 0** (Phase 1 coding can proceed in parallel; it is CPU-only and needs no accounts).

| Phase | Name | Status | Completed |
|---|---|---|---|
| 0 | Setup | 🟡 current | — |
| 1 | BPE tokenizer | 🟡 in progress (skeleton + tests done) | — |
| 2 | GPT model | ⬜ | — |
| 3 | Pretraining + novelty | ⬜ | — |
| 4 | GSM8K eval + baselines | ⬜ | — |
| 5 | SFT + LoRA | ⬜ | — |
| 6 | DPO | ⬜ | — |
| 7 | Ablations + error analysis | ⬜ | — |
| 8 | Demo + README + resume | ⬜ | — |

Rules that apply to every phase are in CLAUDE.md §2, §3 and §15. Supporting docs:
`docs/decisions.md` (decision log) and `docs/learning/phase<N>.md` (learning notes + quiz).

---

## Phase 0 — Setup (accounts, repo, environment) 🟡 CURRENT
**Goal:** Everything needed to write, test and later run code exists, and no secret is ever committed.

### 🤖 CLAUDE tasks
- [x] Folder structure from CLAUDE.md §5 (2026-10-13)
- [x] `requirements.txt`, `.gitignore`, placeholder `README.md` (2026-10-13)
- [x] `utils/seed.py`, `utils/config.py` (2026-10-13)
- [x] Git repo initialised, first commit authored by Aggam, no Claude trailers (2026-10-13)
- [x] `docs/roadmap.md`, CLAUDE.md §15 (2026-10-14)
- [x] `docs/decisions.md` (decision log) started with the setup decisions (2026-10-15)
- [x] `.env.example` with empty placeholder variable names (no values) (2026-10-15)
- [x] Verify the local environment (Python version, `pytest`, `regex`, `tokenizers` import) and report gaps (2026-10-15)
- [x] Sanity-check that `.gitignore` covers `.env`, `data/`, `checkpoints/`, `*.bin`, `*.pt` (2026-10-15)

### 👤 AGGAM tasks
- [ ] Create a GitHub repo and add it as `origin` (private is fine until the demo)
- [ ] Create a Kaggle account; complete phone verification; enable GPU (Settings → Accelerator) and check your weekly GPU quota
- [ ] Create a Hugging Face account; create a **write** access token
- [ ] Store secrets safely: Kaggle Secrets (`HF_TOKEN`) for notebooks, and a local `.env` (gitignored) for your machine. **Never paste a token in chat or commit it.**
- [ ] Create a local virtualenv and `pip install -r requirements.txt` (torch install differs for CPU/CUDA; tell me what you get)
- [ ] Push the repo to GitHub when you are happy with the history

### ✅ Exit criteria
- [ ] GitHub remote exists and `git push` works
- [ ] Kaggle GPU runs `nvidia-smi` in a test notebook (reported back by you)
- [ ] HF token is in Kaggle Secrets and `.env`; `git status` shows `.env` is ignored
- [ ] `pytest -q` runs locally (failures from unimplemented code are fine)

⏱ ~1–2 hours (mostly account verification waits)

---

## Phase 1 — BPE tokenizer 🟡 IN PROGRESS
**Goal:** A byte-level BPE tokenizer you wrote yourself, with exact round-trips, trained to a 4,096 vocab on TinyStories.

### 🤖 CLAUDE tasks
- [x] Explain byte-level BPE and the GPT-2 regex (2026-10-13)
- [x] `bpe.py` skeleton with signatures and TODOs (2026-10-13)
- [x] `tests/test_bpe.py` (23 tests, validated against a throwaway reference implementation) (2026-10-13)
- [ ] Review your `bpe.py` when you paste it; flag bugs and slow spots directly
- [ ] `scratchlm/data/tinystories.py`: download TinyStories, take a ~50–100 MB subset for tokenizer training
- [ ] `scripts/train_tokenizer.py` + `TokenizerConfig` wiring
- [ ] Encoding script: multiprocessing, `uint16` memmap `train.bin`/`val.bin`, `<|endoftext|>` between stories, cached (no re-encoding)
- [ ] Report compression ratio (bytes/token) and write it to `results/`
- [ ] Learning notes + quiz for this phase in `docs/learning/phase1.md`; decision-log entries

### 👤 AGGAM tasks
- [ ] **Implement `bpe.py` yourself** in this order: `train` → `_encode_chunk`/`encode` → `decode` → `save`/`load` → `compression_ratio`. Run `pytest tests/test_bpe.py -q` after each step.
- [ ] Paste your implementation for review; fix what I flag
- [ ] Run tokenizer training on the TinyStories subset (locally or Kaggle) and send me the output
- [ ] Run the corpus-encoding job (long; Kaggle or local) and confirm `train.bin`/`val.bin` exist
- [ ] Eyeball the tokenizer: encode a few stories, print the tokens, check they look like sensible word pieces
- [ ] Answer the Phase 1 quiz in your own words

### ✅ Exit criteria
- [ ] `pytest tests/test_bpe.py -q` fully green (including the HF compression comparison)
- [ ] Tokenizer trained to 4,096 and saved; compression ratio recorded in `results/`
- [ ] `train.bin`/`val.bin` encoded, with token counts logged
- [ ] Your quiz answers reviewed by Claude
- [ ] Completed on: ____

⏱ ~1 day for implementation and review; ~1–3 hours of compute for tokenizer training and encoding

---

## Phase 2 — GPT model
**Goal:** A ~12M-parameter GPT with manual and SDPA attention paths that agree numerically, and that can overfit one batch.

### 🤖 CLAUDE tasks
- [ ] Explain attention, the causal mask, pre-LN blocks, weight tying and init scaling before any code
- [ ] `gpt.py` skeleton (or full code if you ask), with tensor-shape comments
- [ ] `tests/test_gpt.py`: shapes, causal mask, manual == SDPA (`allclose` ~1e-5 fp32), overfit-one-batch
- [ ] `GPTConfig` YAML + `configs/pretrain_tiny.yaml`
- [ ] Review your code; print the exact parameter count at init
- [ ] Learning notes + quiz; decision-log entries

### 👤 AGGAM tasks
- [ ] Implement `gpt.py` yourself (attention, MLP, block, model, `generate`)
- [ ] Paste for review; fix flagged issues
- [ ] Confirm the parameter count is ~12M and say whether it matches what you expected
- [ ] Answer the Phase 2 quiz in your own words

### ✅ Exit criteria
- [ ] `pytest tests/test_gpt.py -q` green
- [ ] Parameter count printed and recorded
- [ ] Quiz reviewed
- [ ] Completed on: ____

⏱ ~1–2 days

---

## Phase 3 — Pretraining on TinyStories + novelty check
**Goal:** A trained model that writes short, coherent, new stories, with evidence (curves, samples, novelty %).

### 🤖 CLAUDE tasks
- [ ] Explain the training loop pieces: AdamW param groups, warmup + cosine, grad accumulation, fp16 + GradScaler, clipping, resume
- [ ] Skeletons/reviews for `pretrain/train.py`, `pretrain/sample.py`, `utils/logging.py`, `utils/checkpoint.py`
- [ ] Smoke-test config (tiny, CPU) to prove the loop, resume, and checkpointing work before the long run
- [ ] `eval/novelty.py` (8-gram hashing, sorted uint64 array, `np.searchsorted`) with tests
- [ ] `scripts/plot_results.py` for loss curves; Kaggle runner notebook cells
- [ ] Learning notes + quiz; decision-log entries

### 👤 AGGAM tasks
- [ ] Implement `train.py` (the loop is the point of this phase) and paste for review
- [ ] Run the smoke test locally and report the output
- [ ] **Run the full pretraining job on Kaggle** (resume after session limits) and download the checkpoint/logs yourself
- [ ] **Look at the loss curves**: do train/val go down smoothly, any spikes, any overfitting gap? Tell me what you see.
- [ ] **Read the story samples** across steps and temperatures (0.5/0.8/1.0/1.2); judge coherence yourself
- [ ] Implement or review `novelty.py` and run it (200 samples); report the % novel 8-grams
- [ ] Answer the Phase 3 quiz in your own words

### ✅ Exit criteria
- [ ] Full run finished; final val loss and perplexity recorded in `results/`
- [ ] Loss-curve plot and step-wise + temperature samples committed to `results/`
- [ ] Novelty result recorded (with the training-set sample size if sampled)
- [ ] You have reported your own observations on curves and samples
- [ ] Quiz reviewed
- [ ] Completed on: ____

⏱ ~3–5 days of work; the GPU run is several hours, split over sessions

---

## Phase 4 — GSM8K eval + baselines
**Goal:** A trustworthy evaluator and honest baseline numbers for Qwen2.5-0.5B base on the test set.

### 🤖 CLAUDE tasks
- [ ] `data/gsm8k.py`: load, train(≈6,973)/dev(500) split with a fixed seed, prompt formatting (§7.2)
- [ ] Explain answer extraction, left-padded batched greedy decoding, and bootstrap CIs
- [ ] Skeletons/reviews for `eval/gsm8k.py`, `eval/bootstrap.py`; `tests/test_gsm8k_extract.py`
- [ ] `scripts/run_baselines.py` (zero-shot and 8-shot; fixed exemplars from train only)
- [ ] Learning notes + quiz; decision-log entries

### 👤 AGGAM tasks
- [ ] Implement the answer extractor and evaluator; paste for review
- [ ] Run the baselines on Kaggle (zero-shot and 8-shot, test set) and send me the metrics JSON
- [ ] **Spot-check 15–20 predictions** in `results/preds/*.jsonl` by eye: is the extractor reading the right number? Report any mismatches.
- [ ] Answer the Phase 4 quiz in your own words

### ✅ Exit criteria
- [ ] `pytest tests/test_gsm8k_extract.py -q` green
- [ ] Zero-shot and 8-shot accuracies with 95% CIs recorded (real runs only)
- [ ] Dev/test discipline confirmed: the test set is touched only for final numbers
- [ ] Your spot-check findings reported
- [ ] Quiz reviewed
- [ ] Completed on: ____

⏱ ~2–3 days

---

## Phase 5 — SFT (full fine-tune) + LoRA
**Goal:** SFT with prompt-loss masking and a from-scratch LoRA, both evaluated on dev, with parity tests against HF/PEFT.

### 🤖 CLAUDE tasks
- [ ] Explain prompt-loss masking, `-100` labels, LoRA math (`A`, `B`, `alpha/r`, merge), gradient checkpointing, fp16 AMP
- [ ] Skeletons/reviews for `posttrain/sft.py`, `posttrain/lora.py`, configs `sft_full.yaml`/`sft_lora.yaml`
- [ ] `tests/test_sft_masking.py` and `tests/test_lora.py` (PEFT parity, merge, zero-init, requires_grad)
- [ ] Memory guidance for the 16 GB T4 (and what to document if full FT OOMs)
- [ ] Learning notes + quiz; decision-log entries

### 👤 AGGAM tasks
- [ ] Implement `sft.py` and `lora.py` yourself; paste for review
- [ ] **Run full fine-tune and LoRA SFT on Kaggle**; download checkpoints; send me logs and peak GPU memory/time
- [ ] **Watch training loss/dev accuracy curves** and tell me whether they look sane
- [ ] Select the best checkpoint by dev accuracy; **approve any hyperparameter change that departs from CLAUDE.md**
- [ ] If full FT OOMs, decide: 8-bit AdamW or document honestly (a plan change needs your approval)
- [ ] Answer the Phase 5 quiz in your own words

### ✅ Exit criteria
- [ ] `test_sft_masking.py` and `test_lora.py` green
- [ ] Full-FT and LoRA dev accuracies, params, memory and time recorded
- [ ] Best checkpoints uploaded by you (Kaggle dataset or HF Hub)
- [ ] Quiz reviewed
- [ ] Completed on: ____

⏱ ~1 week (most of it compute and debugging)

---

## Phase 6 — DPO pairs + DPO training
**Goal:** Self-generated preference pairs and a DPO loop that matches TRL's loss and improves (or honestly fails to improve) dev accuracy.

### 🤖 CLAUDE tasks
- [ ] Explain the DPO loss, implicit rewards, the merged-LoRA + disabled-adapter reference trick, and failure modes (length growth, broken `####`)
- [ ] Skeletons/reviews for `posttrain/make_dpo_pairs.py` and `posttrain/dpo.py`; `configs/dpo_lora.yaml`
- [ ] `tests/test_dpo_loss.py` (TRL parity with pinned version, log 2 at policy == reference, monotonicity)
- [ ] Guard rails: pairs built from the train split only, never dev/test
- [ ] Learning notes + quiz; decision-log entries

### 👤 AGGAM tasks
- [ ] Implement `dpo.py` (loss + loop) yourself; paste for review
- [ ] **Run pair generation on Kaggle**; send me the number of usable pairs
- [ ] **Run DPO training**; send me loss, reward margin and reward accuracy curves
- [ ] **Read dev outputs by eye** for length growth and format breaks, not just accuracy
- [ ] Pick the DPO checkpoint by dev accuracy; approve any β or scope change
- [ ] Answer the Phase 6 quiz in your own words

### ✅ Exit criteria
- [ ] `test_dpo_loss.py` green
- [ ] Pair count logged; DPO dev accuracy recorded against SFT
- [ ] Final test-set results table filled from real runs only
- [ ] Your observations on DPO outputs reported
- [ ] Quiz reviewed
- [ ] Completed on: ____

⏱ ~1 week

---

## Phase 7 — Ablations + error analysis
**Goal:** Evidence for how LoRA rank matters and why each model fails.

### 🤖 CLAUDE tasks
- [ ] Rank-ablation configs (r ∈ {4, 8, 16, 32}, alpha = 2r) and the plotting script (r vs dev/test accuracy)
- [ ] Error-analysis tooling: sample 30–50 failures per model from `results/preds/`, blank category column for you
- [ ] Check your categories afterwards and flag disagreements with reasons
- [ ] Learning notes + quiz; decision-log entries

### 👤 AGGAM tasks
- [ ] **Run the rank ablation on Kaggle** and send me the metrics
- [ ] **Categorise 30–50 failures yourself** for base, SFT and DPO (arithmetic slip, wrong setup, format failure, truncated output). I review only after you finish.
- [ ] Look at the ablation plot and say what you conclude
- [ ] Answer the Phase 7 quiz in your own words

### ✅ Exit criteria
- [ ] Rank-ablation plot committed
- [ ] Error analysis table (your categories, my review notes) committed
- [ ] Quiz reviewed
- [ ] Completed on: ____

⏱ ~3–5 days

---

## Phase 8 — Demo deployment + README + resume bullets
**Goal:** A live demo and an honest write-up in your own voice.

### 🤖 CLAUDE tasks
- [ ] `demo/app.py` (Stories + Math tabs, merged weights from the HF Hub, capped `max_new_tokens`) and `demo/requirements.txt`
- [ ] README structure per CLAUDE.md §13, tables and plots filled from real result files only
- [ ] Reproduce-section commands, and a check that they work from a clean clone
- [ ] Clarity edits to your README and resume text (your voice stays)

### 👤 AGGAM tasks
- [ ] Upload merged model weights to your HF Hub account (with your token, never in chat)
- [ ] **Create and deploy the HF Space** under your account; add secrets in Space settings
- [ ] Test the live demo yourself on several prompts
- [ ] **Write "What I learned / what broke"** in your own words
- [ ] **Write your resume bullets** from real numbers (X/Y/Z/N/M)
- [ ] Push the final repo to GitHub
- [ ] Approve final scope/claims in the README

### ✅ Exit criteria
- [ ] Live Space link works and is in the README
- [ ] Every README number traces to a file in `results/`
- [ ] Your learning section and resume bullets are written by you
- [ ] Completed on: ____

⏱ ~3–5 days

---

## Running log of new 👤 tasks
Add anything that comes up mid-project here, with a date.

- _(none yet)_
