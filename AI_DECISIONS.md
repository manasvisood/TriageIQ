# AI_DECISIONS.md — FRONTLINE Triage System

## 1. Model Chosen

**Model:** `llama-3.3-70b-versatile` via **Groq**

**Why Groq:**
- Free tier with generous rate limits — ideal for a one-day challenge
- Extremely low latency (typically 300–800ms per call) vs OpenAI GPT-4o
- Supports `response_format: json_object` (JSON mode), which forces valid JSON output
- No credit card required for initial testing

**Why llama-3.3-70b-versatile:**
- Best instruction-following and reasoning quality available on Groq
- Handles non-English text, sarcasm, and adversarial inputs well
- 70B parameter scale provides sufficient nuance for classification tasks
- Deterministic enough at temperature=0.1 for consistent outputs

**Alternative considered:** GPT-4o-mini — rejected because no OpenAI key was available in the environment.

---

## 2. Tools / Libraries

| Library | Purpose | Why |
|---|---|---|
| `groq` | LLM API client | Official SDK, clean interface |
| `pydantic` v2 | Schema validation | Strict typing, clear errors, automatic coercion |
| `python-dotenv` | Env management | Keep secrets out of source code |
| `rich` | Console output | Makes the demo visually clear without a web UI |
| `pytest` | Testing | Standard Python testing, works with mocks |

**No web UI:** Rich console output is sufficient for a 3-minute demo and avoids building/debugging a frontend under time pressure.

**No tool/function calling:** The task is pure classification. No external data lookup is needed. Adding tool calls would introduce latency and complexity without improving accuracy on this task.

---

## 3. Prompt Strategy

**Architecture:** System prompt + user prompt template, with few-shot examples.

**System prompt design principles:**
1. Establishes the AI as a classification engine — not a chatbot
2. Explicitly names prompt injection as a threat and instructs the model to classify injection attempts rather than follow them
3. Wraps customer message in `<customer_message>` XML tags to create a clear semantic boundary between instructions and data
4. Defines all categories and priority levels with precise, non-overlapping semantics
5. Specifies `needs_human` trigger conditions explicitly
6. Prohibits hallucination with concrete examples ("do NOT invent transaction IDs")
7. Instructs the model to handle non-English messages gracefully

**Few-shot examples included (4):**
- Normal billing message → billing/P1/human
- Feature request → feature_request/P3/no-human
- Prompt injection → out_of_scope/P3/human
- Vague message → billing/P1/human with honest summary

These were chosen to cover the most common failure modes without exhausting the context window.

**Temperature:** 0.1 — low enough for consistency, not zero to avoid complete determinism.

---

## 4. Output Schema

```json
{
  "message_id": "string (injected by pipeline, never from LLM)",
  "category": "enum: billing|technical|account|feature_request|complaint|out_of_scope|general_inquiry",
  "priority": "enum: P0|P1|P2|P3",
  "summary": "string (5–300 chars, facts only)",
  "suggested_action": "string (5–300 chars)",
  "needs_human": "boolean",
  "confidence": "float [0.0, 1.0]"
}
```

`message_id` is always injected by the pipeline code — the model never sets it, preventing ID spoofing.

Pydantic v2 validates every response before it enters the results list.

---

## 5. Category and Priority Strategy

### Categories (7):
Chosen to cover the real support taxonomy without being overly granular:
- `billing` — any money/payment concern
- `technical` — any product malfunction
- `account` — any access/identity concern
- `feature_request` — improvement requests
- `complaint` — dissatisfaction without a specific technical/billing issue
- `out_of_scope` — adversarial, illegal, or completely irrelevant
- `general_inquiry` — informational questions

Multi-issue messages are classified by the highest-priority issue present.

### Priority (P0–P3):
- **P0:** Objective business impact NOW — security breach, full outage, data loss
- **P1:** Customer is blocked — login failure, payment failure, account suspension
- **P2:** Customer is inconvenienced — non-critical bug, billing question
- **P3:** No urgency — inquiry, feature request, feedback

Anger alone does not raise priority. An angry customer with a P3 question gets P3.

---

## 6. How Uncertainty is Handled

1. **Confidence score:** The model outputs a float 0.0–1.0 representing classification confidence
2. **Automatic escalation:** A Pydantic `@model_validator` forces `needs_human=True` whenever `confidence < 0.5`
3. **Prompt instruction:** The system prompt explicitly tells the model to reflect vagueness in lower confidence scores
4. **Vague messages:** Classified with lower confidence and escalated to humans
5. **Multi-issue messages:** Classified by dominant issue, escalated to human

---

## 7. How Prompt Injection is Handled

**Defense in depth:**

1. **System prompt declaration:** Explicitly names prompt injection as a threat. The model is told customer messages are untrusted input, not instructions.
2. **XML boundary:** Customer message is wrapped in `<customer_message>` tags, reinforcing the data/instruction separation
3. **Behavioral instruction:** Model is told to classify injection attempts as `out_of_scope` with `needs_human=true`
4. **Hard character cap:** Message text is truncated at 4000 characters before sending, limiting attack surface
5. **Schema enforcement:** Even if the model is partially fooled, Pydantic validates the output — an injected malicious priority or category value will fail validation and trigger a retry

**What this does NOT protect against:** Sufficiently sophisticated jailbreaks that fully override the system prompt. In production, a separate content-moderation layer (e.g., Llama Guard) would be added.

---

## 8. How Hallucination is Prevented

1. **Explicit grounding instruction:** "Base your summary ONLY on information actually present in the customer message."
2. **Negative examples in the prompt:** The prompt lists specific things the model must NOT invent (transaction IDs, amounts, dates, error codes)
3. **Low temperature:** 0.1 reduces creative generation
4. **Summary length cap:** 300 characters limits how much the model can write, reducing surface area for hallucination
5. **Human escalation for missing info:** If critical information is missing, the model should set `needs_human=true` and note the absence in the summary rather than inventing it

---

## 9. Retry / Failure Strategy

- **Max retries:** 3 per message
- **Retry delay:** Progressive (2s × attempt number)
- **What triggers retry:** Transient API errors (timeout, rate limit, 5xx), malformed JSON, schema validation failure
- **What does NOT retry:** 4xx client errors (bad request, auth failure)
- **On exhaustion:** A `FailedResult` is recorded and the pipeline continues to the next message
- **Pipeline isolation:** Each message is processed in a try/except block — one failure cannot cascade

---

## 10. Evaluation Methodology

- **Ground truth:** 10 of 40 messages have manually defined expected outputs
- **Metrics calculated:** Category accuracy, priority accuracy, needs_human accuracy, exact-match rate (all three fields correct)
- **Disagreements:** Every mismatch is shown with the expected value, actual value, and reason
- **Reproducibility:** `python evaluate.py` re-runs evaluation on saved results
- **Honesty:** Failed messages are counted as disagreements if they have ground truth — not silently excluded

**Limitations:** Ground truth was defined by a human for this challenge. The "correct" priority and category for some messages is genuinely ambiguous. The evaluation reflects agreement with one human's judgment, not absolute correctness.

---

## 11. Known Weaknesses

1. **Prompt injection is partially mitigated, not fully prevented.** A sophisticated multi-turn attack could still succeed. In production: add a dedicated moderation model.
2. **Ground truth is small (10 examples).** Accuracy metrics have high variance — one disagreement = 10% drop. Not statistically robust.
3. **No caching.** The same message re-run twice costs two API calls. In production: add a content-hash cache.
4. **Single-language summary.** Non-English messages are summarized in English. A real system might want to preserve the original language.
5. **No structured account lookup.** The system classifies based only on the message text — it has no access to CRM data, billing records, or previous tickets. This limits the accuracy of some decisions.
6. **Category ambiguity for multi-issue messages.** When a message spans billing + technical, the choice of primary category is heuristic.
7. **Groq rate limits.** At high volume, rate limiting could cause retries. Production would need a proper queue/backoff strategy.

---

## 12. What I Would Improve With More Time

1. **Add a moderation pre-filter** (e.g., Llama Guard or OpenAI Moderation API) before classification to catch prompt injection and harmful content before reaching the triage model
2. **Larger ground truth set** (50+ labeled examples) for statistically meaningful evaluation
3. **CRM tool integration** — give the model read-only access to customer account data via function calling to enable more accurate prioritization
4. **Confidence calibration** — evaluate whether the model's self-reported confidence scores are actually calibrated (do 80% confidence predictions match 80% of the time?)
5. **A/B test multiple models** (GPT-4o-mini vs llama-3.3-70b) on the same dataset to measure accuracy/cost tradeoff
6. **Async processing** for large datasets (10x throughput with `asyncio` + `groq.AsyncGroq`)
7. **Minimal web UI** (Streamlit, ~50 lines) for a more polished demo
8. **Structured logging** with correlation IDs for production observability
