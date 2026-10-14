# Implementation Plan: "Ask Your Market" — RAG Assistant

## Overview

This is a **2-day MVP** with 6 phases, each building on the previous one. The critical path is: data → tools → LLM tool-calling → Gradio → eval → deploy.

---

## PHASE 0 — Environment & Decisions (Day 1, ~1 hour)

### Goal
Resolve the open questions, set up the project skeleton, and make the stack concrete.

### Decisions to lock

| Decision | Options | Recommendation |
|---|---|---|
| Embedding model | `multilingual-e5-small`, `bge-m3`, others | `multilingual-e5-small` (fast, local, decent multilingual) — verify it runs on available hardware |
| LLM via OpenRouter | Any OpenAI-compatible model with tool calling | Pick a free/cheap model that reliably supports tool calls; test it early with a minimal call |
| Description source | Re-scrape vs. add column to existing scraper | For the MVP demo, generate descriptions synthetically (or hand-write short ones) so the demo is self-contained; document that the real path is "add one column to the n8n scraper" |
| Chroma persistence | in-memory vs. persisted dir | Persisted `chroma_db/` dir so the index isn't rebuilt on every run |

### Tasks
1. Create the repo structure (see file layout below).
2. Write `.env.example` with: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `EMBEDDING_MODEL`, any other knobs.
3. Create a Python venv, pin dependencies in `requirements.txt`.
4. Quick sanity check: can we (a) load the embedding model locally, (b) call OpenRouter with tool calling? Do this **before** writing any business logic — a blocker here reshapes the whole plan.
5. Decide the synthetic data approach and the book sandbox URL.

### Output
- Working Python env, `.env.example`, confirmed embedding + LLM tool-calling path.

---

## PHASE 1 — Data layer (Day 1, ~2-3 hours)

### Goal
Get a CSV of snapshot history onto disk and a Python data module that can query it reliably. This is the foundation everything else queries.

### Sub-phase 1a — Synthetic history generator

The sandbox (`books.toscrape.com`) has static prices, so we generate a believable 30-day history.

**What the script does:**
- Starts from a small product catalog (20-100 products) with: `url, title, description, base_price, currency, in_stock, promo, rating, reviews_count`.
- For each of ~30 days, produces a snapshot row per product with small random walks on price (down weeks, promo spikes), occasional stock flips, promo appearing/disappearing, rating drift.
- Writes one combined CSV: `snapshots.csv` with columns `scraped_at, url, title, price, currency, in_stock, promo, rating, reviews_count, description`.
- Also writes a catalog file (the "latest" view) for convenience.

**Design points:**
- Make the random seed fixed for reproducibility of the eval set.
- Ensure there are real, checkable events: e.g., product X drops price on day 12; product Y goes out of stock on day 20. These become gold questions later.
- Keep price changes within realistic ranges (no 90% drops).
- `description` should be varied enough that semantic search is meaningful — not identical boilerplate.

### Sub-phase 1b — Data module (`data.py` or `src/data.py`)

A small pandas-based module with clean functions the tools will call:

- `load_snapshots(path) -> DataFrame` — cached, parsed dates.
- `latest_snapshot(df) -> DataFrame` — most recent `scraped_at` per product.
- `price_change_over_period(df, days) -> DataFrame` — products whose price changed in the last N days, with old/new price and delta.
- `product_history(df, url) -> DataFrame` — time series for one product.
- `out_of_stock(df) -> DataFrame` — currently OOS products.
- `most_expensive(df, n)`, `least_expensive(df, n)`.
- `find_by_url(df, url) -> row`.
- A small "available data" descriptor the LLM can reference (column list, date range, product count) so it knows what it *can't* answer.

### Design points
- All functions return DataFrames but the tool layer will convert to small, LLM-friendly structures (lists of dicts, not 100-row tables).
- Cap result sizes (e.g., top 20) to keep prompts small and token costs down.
- Keep the module pure (no LLM, no vector DB) — easy to test against the CSV.

### Output
- `scripts/generate_history.py` producing `data/snapshots.csv`.
- `src/data.py` with the query functions.
- A quick smoke test: load the CSV, run each query function, print results.

---

## PHASE 2 — Vector index for descriptions (Day 1, ~2 hours)

### Goal
Semantic search over product descriptions, persisted locally.

### Tasks

1. **Embedding function** (`src/embed.py` or similar):
   - Load the chosen local model once (singleton), expose `embed(text) -> vector` and `embed_batch(texts) -> matrix`.
   - Handle the model's expected input format (some want a prefix like `"query: "` / `"passage: "`).

2. **Index builder** (`src/index.py`):
   - Read the CSV, dedupe by URL (one record per product, latest title/price as metadata).
   - Embed each description.
   - Store in Chroma with metadata: `url, title, price, currency, in_stock, promo, scraped_at`.
   - Persist to `chroma_db/`.
   - Provide `build_index(csv_path, persist_dir)` and `get_collection(persist_dir)`.

3. **Semantic search function**:
   - `search_descriptions(query, top_k) -> [ {url, title, price, ... , score} ]`.

### Design points
- One record per product, not one per snapshot — we want "find products like X", not "find snapshots like X".
- Metadata must carry enough for sources (URL, title, price) without a hit to the CSV.
- Rebuild the index only when the CSV changes (script, not on every chat run). The chat app loads the existing collection.
- If hardware is tight, embed in batches and consider quantized model; measure embedding time for the catalog size.

### Output
- `chroma_db/` populated.
- `src/index.py` with build + search.
- Smoke test: search for a product's own description and for a related term; verify the product ranks first.

---

## PHASE 3 — Tools + LLM orchestrator (Day 1-2, ~3-4 hours)

### Goal
An LLM that can call two tools, get results, and produce a sourced answer or a refusal. This is the core of the assistant.

### Sub-phase 3a — Tool definitions

Two tools, matching the concept:

**Tool 1 — `query_snapshots`** (table queries):
- Accepts a structured intent, not free text. In practice, the LLM calls it with parameters like:
  - `question_type` ∈ { `price_changes`, `out_of_stock`, `product_history`, `most_expensive`, `least_expensive`, `current_product`, `available_data` }
  - `params` per type: e.g. `days` for price_changes; `url` for product_history; `n` for most/least expensive.
- Implementation dispatches to `src/data.py` functions, returns a compact result.

**Tool 2 — `search_descriptions`** (semantic search):
- `query` (string), `top_k` (int, default ~5-10).
- Returns matched products with metadata + score.

**A third "identity" tool (optional but useful):**
- `get_data_summary` — returns column list, date range, product count. Helps the LLM decide what's in scope and what isn't.

Keep tool outputs small and typed. The LLM must never see raw 100-row tables.

### Sub-phase 3b — LLM client with tool calling

- Wrap OpenRouter as an OpenAI-compatible client.
- Implement the tool-call loop:
  1. Send messages + tool definitions.
  2. If the model requests tool calls, execute them (one round; the concept wants at most one tool, or both for mixed — decide the loop policy; start with "call all requested tools once, then answer").
  3. Append tool results to the conversation.
  4. Get the final answer.
- System prompt is critical. It should encode:
  - Answer **only** from tool results.
  - Always cite sources: product title, snapshot date, URL, and the value (price/rating/etc.) from that snapshot.
  - If tools return nothing relevant → explicit refusal: "This is not in the data." with a short reason.
  - If the question asks for something you don't track (sales volume, revenue, future prices) → refuse, don't hedge.
  - Prefer `query_snapshots` for exact numeric questions; prefer `search_descriptions` for "similar to" / meaning questions; use both for mixed.
  - Keep figures exact (copy from the data), don't round inventively.

### Sub-phase 3c — Source attribution

- Build a small `format_sources(results)` helper that turns tool results into the "Sources" block the UI will show.
- The answer text and the sources block should be produced together so they stay consistent. Consider returning a structured object `{answer: str, sources: [...]}` from the orchestrator rather than parsing the LLM's free text for sources. This is more reliable for eval and for the UI.

### Design points
- Tool calling quality depends heavily on the model. Test with the chosen free model early; if it's unreliable, the fallback is a cheaper paid model for the demo — that's a real, honest finding for the README.
- Add a timeout and a cheap retry for LLM calls.
- Log the conversation + tool calls during eval so failures are debuggable.

### Output
- `src/tools.py` (tool definitions + dispatch).
- `src/llm.py` (OpenRouter client + tool-call loop).
- `src/orchestrator.py` (system prompt + flow + structured return).
- Smoke test: exercise each tool type manually and via the orchestrator with a few prompts.

---

## PHASE 4 — Gradio chat interface (Day 2, ~2 hours)

### Goal
A usable chat UI with a Sources block under each answer, deployable to Hugging Face Spaces.

### Tasks

1. **Build the chat** (`app.py` or `demo.py`):
   - Gradio `ChatInterface` or a custom `Blocks` chat with message history.
   - On each user message: run the orchestrator, render the answer + a "Sources" component beneath it.
   - Sources block: list of cards/rows with title, price/value, snapshot date, URL (clickable).
   - Show a small status while the LLM is thinking (the free model can be slow).

2. **Loading / startup**:
   - On startup: load the CSV via `src/data`, load the Chroma collection via `src/index`. Fail loudly with a clear message if either is missing (helps debugging on Spaces).
   - Keep the embedding model and Chroma client as singletons.

3. **Config from env**:
   - Model, API key, paths — all from env / `.env`, so the same code runs locally and on Spaces.

4. **Polish**:
   - Example prompts (the concept's example questions) as presets/quick prompts.
   - Nice formatting for prices (currency), stock status, promo.
   - Empty-state: if the user asks and the data is missing, show the refusal clearly.

### Design points
- The Sources block must be a first-class UI element, not buried in the answer text — that's the whole "verifiable answer" selling point.
- Keep the UI minimal; this is a portfolio demo, not a product. But it must look intentional.

### Output
- `app.py` runnable with `python app.py` locally and as a Hugging Face Space.
- Local smoke: open the chat, ask one exact, one semantic, one mixed, one out-of-scope question; verify sources appear.

---

## PHASE 5 — Evaluation set + metrics (Day 2, ~2-3 hours)

### Goal
A credible, measurable quality story for the README — the part that sells the project.

### Sub-phase 5a — Build the 25-question set (`eval/questions.json`)

| Category | Count | Reference |
|---|---|---|
| Exact | 10 | Computed automatically from the CSV (so it's objective) |
| Semantic | 8 | Which products should be in the top results |
| Mixed | 4 | Both tools expected |
| Out of scope | 3 | Refusal expected |

**Exact questions (auto-referenced):** derive answers directly from the data — e.g., "which products got cheaper in the last 7 days" → compute from CSV; "what's the price of product X" → look up latest; "how did product X's price change" → history diff. This makes pass/fail objective, not subjective.

**Semantic questions:** pick queries where you know the intended products (e.g., a product's own description, or a category term), and record the expected top-K set.

**Out-of-scope:** questions about sales volume, revenue, forecasts, competitor internal info — things the data genuinely doesn't have.

Keep questions in a structured file: `{id, type, question, expected}`. For exact, `expected` is computable; for semantic, `expected` is a set of URLs/titles; for out-of-scope, `expected` is "refusal".

### Sub-phase 5b — Run the eval (`eval/run.py`)

For each question:
1. Send it through the orchestrator.
2. Capture: answer text, sources, which tools were called, refusal or not, latency, token usage if available.
3. Score:
   - **Exact:** does the answer contain/get the right figures and products? Auto-compare against the computed reference where possible (e.g., set of URLs, price values). For textual answers, a light manual pass is fine for the demo, but the exact-number ones should be automated.
   - **Semantic:** recall@5 (did the expected products appear in the top 5?). Also compare embedding search vs. a BM25 keyword baseline — the concept explicitly asks for this comparison.
   - **Mixed:** did it use both tools and get the right products + the right price/change info?
   - **Out of scope:** was it a refusal (correct) vs. a made-up answer (fail)?

4. Write `eval/results.json` and a human-readable summary.

### Sub-phase 5c — BM25 baseline

- Implement a tiny BM25 over the descriptions (e.g., `rank_bm25` package or a small custom implementation) against the same catalog.
- Run the 8 semantic queries through BM25 and through embedding search.
- Record recall@5 for both. This gives a concrete "why vectors" datapoint for the README.

### Sub-phase 5d — Business-effect timing

- Pick 5 representative questions.
- Time answering by hand in the spreadsheet (stopwatch, do it once or twice, state the assumption).
- Time the assistant (including LLM latency).
- Record tokens/cost per answer from the OpenRouter usage if available.
- Put both numbers in the README as a comparison, with the calculation basis stated (no unearned "saves X hours" claims).

### Output
- `eval/questions.json`
- `eval/run.py`
- `eval/results.json` + a summary table
- BM25 comparison numbers
- Timing numbers for 5 questions

---

## PHASE 6 — README, portfolio package, and deploy (Day 2, ~2 hours)

### Goal
Everything that makes this a sellable portfolio piece, plus a live demo.

### README contents

1. **Headline + one-paragraph pitch** (the Upwork copy from the concept).
2. **Business problem** (the three bullets from the concept).
3. **What it does** — example dialogue with 2-3 turns showing: an exact answer with figures, a semantic answer, a sources block, and a refusal.
4. **Architecture diagram** (data → index → tools → LLM → UI). A simple ASCII or an image generated from the code is fine.
5. **Stack** (Python, pandas, Chroma, local embeddings, OpenRouter, Gradio, HF Spaces).
6. **How to run locally** (`.env` from `.env.example`, install, generate data, build index, run app).
7. **Demo data caveat** — explicit note that the history is synthetic, why, and that on a real client site it builds from the n8n scraper.
8. **Quality results** — the metrics table from eval (exact accuracy, semantic recall@5 for embedding vs. BM25, source-affectation rate, refusal accuracy), plus the 25-question set reference.
9. **Business effect** — the 5-question timing comparison + cost per answer + the calculated weekly saving with assumptions stated.
10. **Limitations and ethics** (the concept's bullets).
11. **Roadmap** (the concept's "not in MVP" list).
12. **Link to the previous project** — the data-collection → accessibility chain.

### Portfolio artifacts
- Screenshots: a chat turn with an answer + the Sources block; a refusal turn.
- A 60-90s Loom: ask an exact question (figures), ask a semantic question (similar products), ask an out-of-scope question (refusal). Show the Sources block.
- Repo contents: code, `data/snapshots.csv`, `scripts/generate_history.py`, `eval/` with questions + results, `.env.example`, `README.md`.

### Deployment
- Push to a GitHub repo.
- Create a Hugging Face Space (Gradio), point it at the repo, set the env vars (OpenRouter key).
- Verify the Space loads the CSV from the repo and answers the four question types live.
- Note: the Space must not require Google Sheets — it runs off the repo's CSV, exactly as the concept specifies.

### Output
- README, screenshots, Loom, live HF Space, public repo.

---

## File layout (proposed)

```
ask-your-market/
├── .env.example
├── requirements.txt
├── README.md
├── app.py                  # Gradio entrypoint
├── src/
│   ├── __init__.py
│   ├── data.py             # pandas queries over snapshots.csv
│   ├── embed.py            # local embedding model
│   ├── index.py            # Chroma build + semantic search
│   ├── tools.py            # tool definitions + dispatch
│   ├── llm.py              # OpenRouter client + tool-call loop
│   └── orchestrator.py     # system prompt + flow + structured answer
├── scripts/
│   └── generate_history.py # synthetic 30-day history → data/snapshots.csv
├── data/
│   └── snapshots.csv       # demo data (git-tracked)
├── chroma_db/              # persisted vector index (git-ignored)
├── eval/
│   ├── questions.json
│   ├── run.py
│   └── results.json
└── .gitignore
```

---

## Cross-cutting concerns to handle throughout

**Reliability of the free LLM.** Free/open models vary in tool-call adherence. Mitigation: pick the model with a real test in Phase 0; design the orchestrator to validate tool calls (ignore malformed ones, re-prompt once); if the free model is flaky, use a cheap paid model for the demo and say so honestly in the README. This is a real finding, not a failure.

**Determinism for eval.** Fixed random seed in the history generator so the eval references stay valid. Log model + config used for each eval run.

**Token cost discipline.** Cap tool result sizes, cap `top_k`, don't feed whole tables to the LLM. Track tokens for the cost-per-answer metric.

**Source correctness.** Sources come from the tool results, not from the LLM's memory. The orchestrator returns structured sources; the UI renders them. This is what makes "verifiable" real rather than a claim.

**Out-of-scope handling.** The data-summary tool + system prompt should make refusals the default for questions about sales/revenue/forecasts. Eval quantifies how often that works.

**Local-first, deployable.** Everything runs locally with no API except the LLM (and even embeddings are local). HF Spaces deployment only needs the OpenRouter key as a secret.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Chosen free model doesn't do tool calling reliably | Test in Phase 0; have a paid fallback; document the choice |
| Embeddings too slow on available hardware | Benchmark in Phase 0; consider quantized model or a smaller model; the catalog is only 20-100 products, so one-time indexing cost is acceptable |
| Semantic search quality weak with short descriptions | Write varied, informative descriptions in the synthetic data; if quality is poor, that's an honest eval result, not a blocker |
| Eval "exact" scoring is subjective | Make as many exact references as possible auto-computed from the CSV; only the free-text phrasing needs a light manual pass |
| HF Space won't load the index or model in time | Keep index build separate from app startup; the Space loads a prebuilt index; document build steps |

---

## Phase 0 status — DONE (2026-10-13)

**Environment:** ✅ Python 3.13.5 venv created; `requirements.txt` installed successfully.

**Files created:** `.env.example`, `requirements.txt`, `.gitignore`, `scripts/phase0_openrouter_smoke.py` (tool-calling smoke test — skips cleanly when no API key is set).

**Directories:** `src/`, `scripts/`, `data/`, `eval/`, `chroma_db/` all present.

**Embedding model (confirmed):** `intfloat/multilingual-e5-small` loads locally via `sentence-transformers` (~49s cold first load; fast thereafter). 384-dim vectors. Tested with `query:`/`passage:` prefixes: related pair ~0.87, unrelated ~0.80. Decision: **use multilingual-e5-small with the e5 `query:`/`passage:` prefix convention** in `src/embed.py`. No fallback needed — hardware is fine for a 20-100 product catalog.

**OpenRouter tool-calling (CONFIRMED):** a dedicated test script `scripts/phase0_toolcall_test.py` runs a two-turn scenario against the live OpenRouter API: (1) ask the model to call `get_data_summary`, (2) inject a fake summary result, (3) ask for a concise answer from that data. **Result:** `nvidia/nemotron-3.5-lightning:free` calls the tool correctly (finish_reason=tool_calls, valid JSON args `{"verbose":false}`), and the second turn produces a grounded answer from the injected result. Cost was $0 (free tier). Latency ~2s per turn. The test also includes a retry loop (up to 2 re-prompts) in case the model returns text instead of a tool call on a given turn. **Decision locked: `OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free`.**

### Decisions locked after Phase 0

| Decision | Choice | Notes |
|---|---|---|
| Embedding model | `intfloat/multilingual-e5-small` | Local, 384-dim, fast; use `query:`/`passage:` prefixes |
| LLM via OpenRouter | `nvidia/nemotron-3.5-lightning:free` (confirmed) | Free tier, $0, tool-calling verified live; set in `OPENROUTER_MODEL`. Swap in `.env` for a stronger model later. |
| Description source | synthetic, self-contained | No Google Sheets for the demo; real path = add one column to the n8n scraper |
| Chroma persistence | persisted `chroma_db/` | Rebuild only when CSV changes |
| Data approach | synthetic 30-day history generator | Fixed seed; embeds real checkable events for eval |

### Blockers / next actions

None — Phase 0 fully complete.

1. Proceed to **Phase 1** (synthetic history generator + `src/data.py`).

### Notes

- `chromadb` resolved to 1.5.9 (requirements said `>=0.6`). That's fine — the concept just wants a local Chroma; the API we use is stable.
- No Hugging Face token is set; model downloads work unauthenticated but with lower rate limits. Not a blocker for the demo.
- Symlink warning on the HF cache (Windows/Dev Mode). Not a blocker; embeddings still work.
