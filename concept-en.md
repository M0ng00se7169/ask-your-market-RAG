# Concept: Competitor Monitoring Automation (n8n + Scraping + LLM)

## Goal

Build a working MVP of an automation in one day for an Upwork portfolio that demonstrates scraping, LLM data processing with structured output, change detection, and notifications. The data from this project is reused in the follow-up RAG project (semantic search over products and reviews).

## Problem

E-commerce store owners check competitors' prices and reviews by hand. It takes several hours a week, is done irregularly, and important changes (a price drop, a promotion, a wave of negative reviews) are noticed too late.

## Solution

An n8n workflow that collects competitors' product data on a schedule, extracts it into a single JSON schema with an LLM, compares it with the previous run, and sends a digest of changes to Telegram.

**Portfolio headline:** "Competitor monitoring: from 3 hours of manual work a week to an automatic digest."

## Target client

Small and mid-sized online stores, marketplace sellers, and agencies that run e-commerce for their clients.

## How it works

1. **Trigger:** a schedule (once a day) plus a manual run for demos.
2. **Scraping:** n8n takes a list of product URLs, loads the pages, and extracts the raw text/HTML of the product block.
3. **Extraction (LLM):** the model turns the raw text into JSON following a fixed schema. An invalid response is discarded and retried once.
4. **Comparison:** the new data is matched against the previous snapshot in storage, and changes are detected.
5. **Notification:** if there are changes, a digest goes to Telegram. If scraping or the LLM fails, a separate alert is sent.
6. **Storage:** the snapshot is written to Google Sheets (or Airtable).

## Data schema (JSON)

```json
{
  "url": "string",
  "title": "string",
  "price": 0.0,
  "currency": "string",
  "in_stock": true,
  "promo": "string | null",
  "rating": 0.0,
  "reviews_count": 0,
  "review_sentiment": "positive | mixed | negative",
  "review_highlights": ["string"],
  "scraped_at": "ISO datetime"
}
```

## What counts as a change (MVP)

- Any price change
- Product came back in stock / went out of stock
- A promotion appeared or disappeared
- A noticeable rating drop (for example, 0.3 or more)

## Stack

- **n8n** (self-hosted in Docker or n8n Cloud trial)
- **Scraping:** n8n HTTP Request + HTML Extract (Firecrawl as a fallback for dynamic pages)
- **LLM:** OpenAI or Claude via n8n, structured output following a JSON schema
- **Storage:** Google Sheets
- **Notifications:** Telegram Bot

## In the MVP

- 1 niche, 1 site (or 2-3 sites), 10-20 products
- One workflow with the main flow and one with error handling
- LLM extraction with schema validation
- Comparison with the previous snapshot and a Telegram digest
- Failure alert
- Deduplication (one product = one row per run)

## Not in the MVP (Roadmap)

- Dashboard and price-trend charts
- Multi-language support
- Many sites and proxy rotation
- Anti-bot bypass
- CRM, Slack, and email integrations
- Automatic recommended-price calculation
- Connection to RAG search (the next project)

## Limitations and ethics

- Scrape public pages only, check `robots.txt` and the site's terms, and state this in the README.
- Do not collect users' personal data from reviews (text and rating only).
- State the cost of one run (LLM tokens) in the README.

## Definition of done

- [ ] The workflow runs manually and on a schedule without errors
- [ ] Data for 10+ products lands in the sheet in the expected schema
- [ ] When a price changes (verify by altering test data), a Telegram digest arrives
- [ ] When the scraper breaks, an alert arrives
- [ ] The workflow JSON is exported

## What goes into the portfolio package

- README: problem, result, workflow diagram, cost per run, limitations, Roadmap
- A screenshot of the workflow from n8n and a screenshot of the digest in Telegram
- A 60-90 second Loom: run, change the data, receive the digest
- Repository: `workflow.json`, `.env.example`, `README.md`, `concept.md`, `plan.md`

## Upwork copy (draft)

> I automate competitor monitoring: I collect prices, availability, promotions, and reviews, extract the data into a structured form with AI, and send a digest of changes to Telegram/Slack. I can set it up for your niche and your channels.

## Link to the next project

The collected descriptions and reviews become the data source for the "E-commerce semantic search" project (hybrid search + answers with quotes from reviews). In the profile, both projects read as one story: data collection → search over that data.

## Open questions

- Which niche and which site for the demo?
- Telegram or Slack for notifications?
- Which LLM (considering cost and access)?