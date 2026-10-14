# Concept: Ask Your Market (a RAG assistant on top of competitor monitoring data)

Related project: [Competitor Monitoring Automation](concept-en.md) (the data source).

## Goal

Build a working MVP in one to two days of an assistant that answers questions about competitor monitoring data, with figures, dates, and links to the source. A portfolio project for Upwork, following on from the n8n project.

## Business problem

A company already has competitor data (a scraper, a spreadsheet, digests), but gets less value from it than it could:

1. **The data is there, the answers are not.** To answer a manager's question ("where are we pricier than the market?"), an analyst filters tables and builds pivots by hand. Every new question becomes a new task, so some questions are simply never asked.
2. **The digest only answers "what changed".** Questions like "how does the competitor differ from us", "what do they have that is similar to our product", and "how did the price move over the month" remain manual work.
3. **An answer without a source cannot be taken to a meeting.** Pricing and purchasing decisions are not made on "the AI said so". The answer must be verifiable: which product, which date, which link.

## Solution

A chat assistant over the history of monitoring snapshots. It chooses how to search on its own:
- **exact questions** (prices, availability, dynamics) are answered by querying the table, not by the model "guessing";
- **questions by meaning** ("find products similar to our X") are answered by semantic search over descriptions;
- **every answer** comes with sources (product, snapshot date, URL). If the data is not there, the assistant says so.

**Portfolio headline:** "I turn a competitor monitoring spreadsheet into an assistant you can ask anything about the market, and get a verifiable answer."

## Target client

Online store owners, category managers, marketplace sellers, and agencies that run e-commerce for their clients. They already have data or a scraper, but lack quick access to answers.

## Example questions

| Type | Question | How the system answers |
|---|---|---|
| Exact | Which products got cheaper in the last 7 days? | Query over snapshot history |
| Exact | What is out of stock right now? | Query over the latest snapshot |
| Exact | How did the price of product X change? | History by URL |
| Semantic | Find products with a description similar to Y | Semantic search |
| Mixed | Which products similar to Y got cheaper? | Semantic search + table query |
| Out of scope | How many sales does the competitor make per month? | Refusal: there is no such data |

## How it works

1. **Data:** the snapshot history from Google Sheets (the `snapshots` tab) is exported to CSV. For the demo the file lives in the repository, so no Google connection is needed.
2. **Index:** product descriptions are turned into vectors and stored in a local vector database (one record per product; metadata: URL, title, price).
3. **Assistant tools:**
   - `query_snapshots`: ready-made functions over the table (price changes over a period, availability, product history, most and least expensive);
   - `search_descriptions`: semantic search over descriptions.
4. **Answer:** the LLM picks a tool (or both), receives the result, and writes the answer strictly from it, with sources. If the tools return nothing, it answers "this is not in the data."
5. **Interface:** a Gradio chat with a "Sources" block under each answer.

## Data

- Base: the `snapshots` table from the monitoring project (`scraped_at, url, title, price, currency, in_stock, promo, rating, reviews_count`).
- **To add to the scraper:** a `description` field (the product description text). One new column and one selector in the Extract HTML node.
- **Important caveat about the demo.** In the `books.toscrape.com` sandbox prices do not change, so the demo history is **synthetic**: a small script generates several snapshots over 30 days with price and availability changes. The README states this explicitly. On a real client site the history builds up on its own.

## Stack

- **Python**, pandas for queries over the history
- **Vector database:** Chroma (local, no infrastructure)
- **Embeddings:** a local model (for example `multilingual-e5-small` or `bge-m3`), free and API-less
- **LLM:** via OpenRouter (OpenAI-compatible API) with tool calling; the model is set in the config
- **Interface and deployment:** Gradio + Hugging Face Spaces

## In the MVP

- 1 data source, 20-100 products, ~30 days of history
- Two tools: table queries and semantic search over descriptions
- Answers with sources and a correct refusal when data is missing
- A test set of 25 questions and a metrics table in the README
- A simple Gradio chat

## Not in the MVP (Roadmap)

- Search over review texts (on a real client site)
- Hybrid search (BM25 + vectors) and a reranker
- Multiple sites and categories
- Authentication, roles, conversation history
- Price-trend charts in answers
- Automatic index refresh on a schedule (integration with n8n)
- Integration into Telegram/Slack as a bot

## How to measure the value (for the README)

**Quality.** A set of 25 questions with reference answers:
- 10 exact (the reference is computed automatically from the table)
- 8 semantic (the reference: which products should appear in the top results)
- 4 mixed
- 3 out of scope (a refusal is expected)

Metrics: share of correct answers to exact questions, recall@5 for semantic search, share of answers with a correct source, share of correct refusals. For semantic search, compare two variants: keyword search (BM25) and embedding search.

**Business effect.** Pick 5 typical questions and time with a stopwatch how long an answer takes by hand in the spreadsheet versus through the assistant. Put the response time and the cost per answer (tokens) in the README. Use a phrase like "saves X hours a week" only as a calculation based on these measurements, with the assumptions stated.

## Limitations and ethics

- The assistant answers only from the data it has and makes no forecasts.
- Public data only, respecting `robots.txt` and the source site's terms.
- The demo history is synthetic, and the README says so plainly.
- Free models are unstable in speed and output format; a paid model is recommended to a client for production.

## Definition of done

- [ ] The assistant answers questions of all four types
- [ ] Every answer contains sources (product, date, URL)
- [ ] For out-of-scope questions the system refuses instead of making things up
- [ ] The 25-question set has been run and the metrics recorded in the README
- [ ] Manual versus assistant response time has been measured
- [ ] A live demo (Hugging Face Spaces) works on the CSV from the repository

## What goes into the portfolio package

- README: business problem, example dialogue, diagram, metrics table, timing comparison, limitations, Roadmap
- Screenshots of the chat with answers and the sources block
- A 60-90 second Loom: a question, an answer with figures, an answer with sources, a refusal on an out-of-scope question
- Repository: code, a CSV with demo data, the history-generation script, `eval/` with questions and results, `.env.example`

## Upwork copy (draft)

> I turn competitor monitoring data into an assistant you can ask anything about the market: "what got cheaper this week", "what does the competitor have that is similar to our product". Every answer comes with a source: product, date, link. If the data is not there, the assistant says so. Quality is measured on a test set of questions. I can connect it to your spreadsheet, database, or scraper.

## Link to the previous project

The first project collects data and sends a digest of changes. This project makes the same data available for free-form questions. In the profile they read as a chain, "collected the data → made it accessible to the team", and they sell each other: a client who only needs monitoring sees the next step, and a client who already has data sees that it can be put to use.

## Open questions

- Which local embedding model to use (depends on hardware and the language of descriptions)?
- Which LLM via OpenRouter (tool calling support is required; check this when choosing a free model)?
- Add the product description to the already collected data, or re-scrape from scratch?