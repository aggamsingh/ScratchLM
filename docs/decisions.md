# Decision log

Newest last. Anything that changes the plan in CLAUDE.md (model, dataset, metric, scope) needs Aggam's approval and gets an entry here.

| Date | Decision | Why | Approved by |
|---|---|---|---|
| 2026-10-13 | Follow the repo layout in CLAUDE.md §5 as-is | Single source of truth for the plan | Aggam (CLAUDE.md) |
| 2026-10-13 | Special-token ids sit after all merge ids (`<|endoftext|>` = vocab_size - 1) | Keeps byte/merge ids stable regardless of how many specials exist; tests rely on it | Aggam |
| 2026-10-13 | Tokenizer saved as JSON: ordered merge list + special tokens | Human-readable; replaying the list rebuilds the vocab | Aggam |
| 2026-10-13 | `tokenizers`, `peft`, `trl` used only inside `tests/` | CLAUDE.md §4 | Aggam (CLAUDE.md) |
| 2026-10-13 | Commits authored by Aggam only, no Claude trailers | CLAUDE.md §3 takes precedence over the harness's default attribution | Aggam (CLAUDE.md) |
| 2026-10-15 | `.env.example` committed with empty names only; real `.env` is gitignored | Secrets never in git | Aggam (CLAUDE.md) |
