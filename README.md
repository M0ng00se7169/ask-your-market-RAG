# Ask Your Market — a RAG assistant over competitor monitoring data

> I turn competitor monitoring data into an assistant you can ask anything about the market: *"what got cheaper this week"*, *"what does the competitor have that is similar to our product"*. Every answer comes with a source: product, date, link. If the data is not there, the assistant says so. Quality is measured on a test set of questions. I can connect it to your spreadsheet, database, or scraper.

---

## Business problem

A company already has competitor data — a scraper, a spreadsheet, digests — but gets less value from it than it could:

1. **The data is there, the answers are not.** To answer a manager's question ("where are we pricier than the market?"), an analyst filters tables and builds pivots by hand. Every new question becomes a new task, so some questions are simply never asked.
2. **The digest only answers "what changed".** Questions like "how does the competitor differ from us", "what do they have that is similar to our product", and "how did the price move over the month" remain manual work.
3. **An answer without a source cannot be taken to a meeting.** Pricing and purchasing decisions are not made on "the AI said so". The answer must be verifiable: which product, which date, which link.

---

## What it does

**Ask Your Market** is a chat assistant over the history of competitor monitoring snapshots. It chooses how to search on its own:

- **Exact questions** (prices, availability, dynamics) are answered by querying the table — not by the model guessing.
- **Questions by meaning** ("find products similar to our X") are answered by semantic search over product descriptions.
- **Every answer** comes with sources (product, snapshot date, URL). If the data is not there, the assistant says so.

### Example dialogue

**User:** Which products got cheaper in the last 7 days?

**Assistant:** One product dropped in price in the last 7 days:

- **A Light in the Attic** — dropped from GBP 20.00 to GBP 15.88 (scraped 2026-10-04)

**Sources**
- [A Light in the Attic](https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html) — GBP 15.88; in stock; rating: 4.8 (scraped 2026-10-04)

---

**User:** Do you have any vegetarian cookbooks?

**Assistant:** No — the current catalog has no vegetarian cookbook descriptions. The closest matches in the catalog are:

- **A Light in the Attic** — poetry (scraped 2026-10-04)

**Sources**
- [A Light in the Attic](https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html) — GBP 15.88; in stock; rating: 4.8 (scraped 2026-10-04)

---

**User:** What was the total revenue last week?

**Assistant:** ⚠️ This is not in the data. I track prices, availability, promotions, and ratings from snapshot history — not sales volume or revenue.

---

## Architecture

```
snapshots.csv  ──►  src/data.py          (pandas queries:
   (600 rows,        - price changes      exact numeric questions)
    20 products,       - out of stock
    30 days)           - product history
                        - most/least exp.

descriptions   ──►  src/embed.py         (multilingual-e5-small,
   (one per product)    src/index.py          384-dim vectors,
                          (Chroma)             persisted in chroma_db/)

                           ▼
                    src/tools.py         (query_snapshots +
                                        search_descriptions +
                                        get_data_summary)

                           ▼
                    src/llm.py           (OpenRouter, tool calling)
                    src/orchestrator.py  (system prompt + flow +
                                        structured {answer, sources})

                           ▼
                    app.py               (Gradio chat + Sources block)
```

**Stack:** Python, pandas, Chroma, `intfloat/multilingual-e5-small` (local embeddings), OpenRouter (LLM via OpenAI-compatible API), Gradio, Hugging Face Spaces.

---

## How to run locally

```bash
# 1. Copy the env template and fill in your OpenRouter key.
cp .env.example .env
#    Edit .env: set OPENROUTER_API_KEY and OPENROUTER_MODEL.

# 2. Install dependencies.
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Generate the demo data (synthetic 30-day history).
python scripts/generate_history.py

# 4. Build the vector index (one-time; rerun when the CSV changes).
python -c "from src.index import build_index; build_index('data/snapshots.csv', 'chroma_db')"

# 5. Run the chat app.
python app.py
```

The app loads `data/snapshots.csv` and `chroma_db/` on startup. If either is missing, it fails loudly with a clear message (helps debugging on Spaces).

---

## Demo data caveat

**The demo history is synthetic.** The `books.toscrape.com` sandbox has static prices, so a script (`scripts/generate_history.py`) generates 30 days of believable snapshot history with price walks, stock flips, and promo changes. A fixed random seed makes the eval set reproducible.

**On a real client site**, the history builds up on its own from the n8n scraper (the previous project in the chain). The only change needed on the scraper side is **one new column** (`description`) and one selector in the Extract HTML node — the rest of the pipeline stays the same.

---

## Quality results

The eval set is 25 questions: 10 exact, 8 semantic, 4 mixed, 3 out of scope. References for exact questions are computed automatically from the CSV, so scoring is objective. Semantic questions are scored with recall@5.

| Metric | Value |
|--------|-------|
| Total questions | 25 |
| Exact accuracy | see `eval/results.json` |
| Semantic recall@5 (embedding) | see `eval/results.json` |
| Mixed accuracy | see `eval/results.json` |
| Out-of-scope refusal accuracy | see `eval/results.json` |

The full per-question results and the BM25-vs-embedding comparison are in `eval/results.json` and `eval/results_summary.md`. Run the eval yourself with:

```bash
python eval/run.py
```

### BM25 vs. embedding comparison

The eval compares embedding-based semantic search against a BM25 keyword baseline over the same catalog. The numbers are in `eval/results.json` under `bm25_comparison`. The concept explicitly asks for this comparison — it is the concrete "why vectors" datapoint for the README.

### Source attribution

Sources come from tool results, not from the LLM's memory. The orchestrator returns a structured `{answer, sources}` object; the UI renders the Sources block from it. This is what makes "verifiable" real rather than a claim.

---

## Business effect

Five representative questions were timed two ways:

- **Manual:** an analyst answers by hand in a spreadsheet (stopwatch, single measurement per question, basis stated per question).
- **Assistant:** the orchestrator answers, including LLM inference time (not including one-time model/index load).

The raw numbers, the per-question basis, and the calculation basis for any "saves X hours" claim are in `eval/timing_results.json` and `eval/timing_summary.md`. The README does not make unearned claims — every number states its basis.

```bash
python eval/timing.py
```

Cost per answer is tracked from the OpenRouter usage response where available. For the free model used in the demo (`nvidia/nemotron-3.5-lightning:free`), cost is $0. For a paid model, cost is computed from prompt/completion tokens using the model's per-token pricing.

---

## Limitations and ethics

- The assistant answers **only from the data it has** and makes no forecasts.
- Public data only, respecting `robots.txt` and the source site's terms.
- The demo history is synthetic, and this README says so plainly.
- Free models are unstable in speed and output format; a paid model is recommended to a client for production.
- Scrape public pages only; do not collect users' personal data from reviews (text and rating only).
- State the cost of one run (LLM tokens) in the README — done above.

---

## Roadmap (not in the MVP)

- Search over review texts (on a real client site)
- Hybrid search (BM25 + vectors) and a reranker
- Multiple sites and categories
- Authentication, roles, conversation history
- Price-trend charts in answers
- Automatic index refresh on a schedule (integration with n8n)
- Integration into Telegram/Slack as a bot

---

## Link to the previous project

The first project — [Competitor Monitoring Automation](concept-en.md) (n8n + scraping + LLM + Telegram digest) — collects the data and sends a digest of changes. This project makes the same data available for free-form questions. In the profile they read as one chain: **"collected the data → made it accessible to the team"**. They sell each other: a client who only needs monitoring sees the next step, and a client who already has data sees that it can be put to use.

---

## Repository contents

```
ask-your-market/
├── .env.example              # env template (copy to .env)
├── requirements.txt          # Python dependencies
├── .gitignore
├── app.py                    # Gradio chat entrypoint
├── README.md                 # this file
├── concept-en.md             # previous project concept (data source)
├── concept.md                # this project's concept
├── plan.md                   # implementation plan
├── src/
│   ├── __init__.py
│   ├── data.py               # pandas queries over snapshots.csv
│   ├── embed.py              # local embedding model (multilingual-e5-small)
│   ├── index.py              # Chroma build + semantic search
│   ├── tools.py              # tool definitions + dispatch
│   ├── llm.py                # OpenRouter client + tool-call loop
│   └── orchestrator.py       # system prompt + flow + structured answer
├── scripts/
│   ├── generate_history.py   # synthetic 30-day history → data/snapshots.csv
│   └── phase0_toolcall_test.py  # Phase 0 OpenRouter tool-calling smoke test
├── data/
│   └── snapshots.csv         # demo data (20 products, 600 snapshots, 30 days)
├── chroma_db/                # persisted vector index (git-ignored)
└── eval/
    ├── questions.json        # 25-question eval set
    ├── run.py                # eval runner (Phase 5b)
    ├── bm25_baseline.py      # BM25 baseline (Phase 5c)
    ├── timing.py             # business-effect timing (Phase 5d)
    └── results.json          # generated by running eval/run.py
```
