"""Phase 0 (part 2): confirm tool-calling works on the configured OpenRouter model."""
import os, sys, json, time

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY or API_KEY.startswith("sk-or-xxx"):
    print("SKIP: no OPENROUTER_API_KEY"); sys.exit(0)

MODEL_ENV = os.environ.get("OPENROUTER_MODEL")
# Use the configured model if set; otherwise fall back to a known-good free model.
# (We still try the configured model even if it previously 404'd — the user may have fixed it.)
MODEL = MODEL_ENV if MODEL_ENV else "nvidia/nemotron-3.5-lightning:free"

import openai
client = openai.OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

tools = [{
    "type": "function",
    "function": {
        "name": "get_data_summary",
        "description": "Return what data the assistant has: columns, date range, product count.",
        "parameters": {
            "type": "object",
            "properties": {"verbose": {"type": "boolean", "description": "Include a sample row."}},
            "required": [],
        },
    },
}]

messages = [
    {"role": "system", "content": "You are helpful. When asked what data you have, call get_data_summary once, then stop. Do not answer from your own knowledge."},
    {"role": "user", "content": "What data do you have? Call the tool and nothing else."},
]
# Deterministic tool-call check: re-run up to 2 more times if the first turn returns text instead of a tool call.
MAX_RETRIES = 2
for attempt in range(1 + MAX_RETRIES):
    if attempt > 0:
        messages.append({"role": "user", "content": "Call the get_data_summary tool now. Nothing else."})
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, temperature=0, max_tokens=200)
    elapsed = time.time() - t0
    msg = resp.choices[0].message
    print(f"attempt {attempt+1} ({elapsed:.1f}s) finish={resp.choices[0].finish_reason} content={repr(msg.content)}")
    if msg.tool_calls:
        print(f"tool_calls: {len(msg.tool_calls)}")
        for tc in msg.tool_calls:
            print("  call:", tc.function.name, "->", tc.function.arguments)
        break
    if attempt == MAX_RETRIES:
        print("FAIL: model never called the function after retries")
        sys.exit(2)


print(f"MODEL: {MODEL}  | tool: get_data_summary")
t0 = time.time()
resp = client.chat.completions.create(model=MODEL, messages=messages, tools=tools, temperature=0, max_tokens=200)
elapsed = time.time() - t0
print(f"turn1 ({elapsed:.1f}s) finish={resp.choices[0].finish_reason}")
msg = resp.choices[0].message
print("content:", repr(msg.content))

if msg.tool_calls:
    print(f"tool_calls: {len(msg.tool_calls)}")
    for tc in msg.tool_calls:
        print("  call:", tc.function.name, "->", tc.function.arguments)

    fake = {
        "columns": ["url","title","price","currency","in_stock","promo","rating","reviews_count","scraped_at","description"],
        "date_range": {"first": "2026-09-13", "last": "2026-10-12"},
        "product_count": 42,
    }
    messages.append(msg)
    messages.append({
        "role": "tool",
        "tool_call_id": msg.tool_calls[0].id,
        "name": msg.tool_calls[0].function.name,
        "content": json.dumps(fake),
    })
    messages.append({"role": "user", "content": "Now answer the user concisely from that data. Keep it short."})

    t1 = time.time()
    r2 = client.chat.completions.create(model=MODEL, messages=messages, temperature=0, max_tokens=300)
    print(f"turn2 ({time.time()-t1:.1f}s) finish={r2.choices[0].finish_reason}")
    print("answer:", repr(r2.choices[0].message.content))

    print("turn1 usage:", dict(resp.usage) if resp.usage else "n/a")
    print(f"RESULT: tool-calling CONFIRMED with {MODEL}")
    print("ACTION: set OPENROUTER_MODEL=", MODEL, "in .env")
else:
    print("FAIL: model did not call the function. Try another model.")
    print("ACTION: pick a free model from https://openrouter.ai/models?supported_parameters=tools")
    sys.exit(2)

print("DONE")
