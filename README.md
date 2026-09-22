# Pull, Optimisation and Evaluation of Prompts with LangChain and LangSmith

This project pulls a low-quality prompt from the LangSmith Prompt Hub, rewrites
it with advanced prompt engineering techniques, pushes the optimised version
back as a public prompt, and scores it against a 15-example dataset using five
custom metrics.

The task the prompt performs: turning a **bug report** into an actionable
**User Story** with testable acceptance criteria.

> **Language note.** The code and instructions are written in English, but the
> prompt deliberately tells the model to answer in Brazilian Portuguese: the 15
> reference outputs in `datasets/bug_to_user_story.jsonl` are in Portuguese, so
> generating English would score near zero on F1 and Precision.

---

## A) Applied Techniques (Phase 2)

Four techniques were applied. Few-shot Learning is mandatory per the challenge;
the other three were chosen to fix specific defects found in v1.

### 1. Role Prompting

**Why.** v1 opened with *"Você é um assistente que ajuda a transformar relatos
de bugs"* — an assistant with no domain, no seniority and no point of view. A
generic persona produces generic output: the model described the defect instead
of the user need.

**How it was applied.** The persona became a concrete professional, with the
job that actually writes user stories:

```
You are a senior Product Manager with ten years of experience in agile
product teams. You specialise in translating bug reports — often vague,
overly technical, or written in frustration — into User Stories that a
developer can implement and a QA engineer can test without having to ask
follow-up questions.
```

The clause about developers and QA is doing real work: it sets the bar for how
specific the acceptance criteria have to be.

### 2. Chain of Thought

**Why.** Turning a bug into a user story is not transcription, it is analysis:
you have to infer who is affected, separate multiple failures hiding in one
report, and find the business value. v1 gave no reasoning scaffold, so the
model jumped straight to output and frequently used "Como um usuário" for
everything.

**How it was applied.** Six ordered questions the model answers silently before
writing — and an explicit instruction never to show that reasoning, so the
private analysis does not leak into the final text:

```
1. WHO is affected? Identify a concrete persona ...
2. WHAT does that person want to accomplish? Describe the desired behaviour,
   not the defect. "I want to add products to my cart", not "I want the
   button to stop failing".
...
4. HOW MANY distinct problems does the report contain? Enumerate them first,
   because each one becomes its own group of acceptance criteria.
```

### 3. Skeleton of Thought

**Why.** This was the largest gap. The dataset ranges from a 63-character bug
to a 2,559-character one, and the expected answers scale with it — from ~400
characters to ~5,700. A single output format either under-answers the complex
reports or pads the simple ones with invented requirements. v1 specified no
format at all.

**How it was applied.** A depth ruler that makes the model classify the report
first and then commit to a matching skeleton:

| Level | Trigger | Response shape |
|---|---|---|
| Simple | one failure, one or two sentences, no technical data | user story + `Critérios de Aceitação:` with 4–6 bullets, nothing else |
| Medium | one or two failures with logs, endpoints, severity or impact | user story + criteria + one or two context sections |
| Complex | three or more failures, or a long multi-section report | `=== USER STORY PRINCIPAL ===`, `=== CRITÉRIOS DE ACEITAÇÃO ===` with letter-labelled groups, `=== CRITÉRIOS TÉCNICOS ===`, `=== CONTEXTO DO BUG ===`, `=== TASKS TÉCNICAS SUGERIDAS ===`, `=== MÉTRICAS DE SUCESSO ===` |

### 4. Few-shot Learning

**Why.** Rules describe a format; examples demonstrate it. v1 had zero
examples, which left the Gherkin style of the acceptance criteria and the
output language entirely to chance.

**How it was applied.** Two complete input/output pairs — one simple, one
medium — plus a written description of the complex skeleton. The example
outputs are in Portuguese, which is how the prompt teaches the output language
without spending a rule on it.

The examples are **original**, not taken from `datasets/bug_to_user_story.jsonl`.
Using evaluation examples as few-shot would inflate the score on those specific
items while measuring nothing about generalisation.

### Structural fixes carried along

| Defect in v1 | Fix in v2 |
|---|---|
| `{bug_report}` interpolated in **both** system and user prompt | the variable lives only in the user prompt; `push_prompts.py` rejects a system prompt containing it |
| no output format defined | explicit `Como um / eu quero / para que` template plus `Dado / Quando / Então` criteria |
| no behaviour rules | rules against preambles, invented figures, vague criteria and dropped technical detail |
| no edge-case handling | five documented cases: vague report, multiple problems, feature request, no end-user impact, already-a-user-story |

---

## B) Final Results

The published prompt scored above the 0.8 threshold on **all five metrics**, which
is the strict criterion of the challenge (not just the average).

### Official run — `python src/evaluate.py`

```
==================================================
Prompt: handle-setup/bug_to_user_story_v2
==================================================

Métricas Derivadas:
  - Helpfulness: 0.90 ✓
  - Correctness: 0.92 ✓

Métricas Base:
  - F1-Score: 0.93 ✓
  - Clarity: 0.90 ✓
  - Precision: 0.90 ✓

--------------------------------------------------
📊 MÉDIA GERAL: 0.9092
--------------------------------------------------

✅ STATUS: APROVADO - Todas as métricas >= 0.8
```

### LangSmith evidence

| Item | Where |
|---|---|
| Published prompt (public) | <https://smith.langchain.com/prompts/bug_to_user_story_v2/40007940> |
| Evaluation dataset (15 examples) | `prompt-optimization-challenge-resolved-eval` |
| Traces for every run | project `prompt-optimization-challenge-resolved` |

Each of the 15 examples produces one generation trace plus three judge traces,
so the full run is traceable end to end in the LangSmith dashboard.

### v1 vs v2 comparison

Both versions were scored against the same 15 examples, the same models and the
same metric code. v1 was measured locally, because `src/evaluate.py` only pulls
the v2 prompt from the Hub.

| Metric | v1 (original) | v2 (optimised) | Δ |
|---|---|---|---|
| Helpfulness | 0.9167 | 0.90 | −0.02 |
| Correctness | 0.8982 | 0.92 | +0.02 |
| F1-Score | 0.8830 | 0.93 | +0.05 |
| Clarity | 0.9200 | 0.90 | −0.02 |
| Precision | 0.9133 | 0.90 | −0.01 |
| **Average** | **0.9062** | **0.9092** | **+0.003** |

**An honest reading of this table.** The challenge brief expects v1 to score
around 0.45–0.52. It does not: measured with `gemini-3.5-flash-lite`, v1 scores
0.91. A current-generation model compensates for a vague prompt — it infers the
user story format on its own, even when nothing in the prompt describes it.

So the headline averages are close, and the difference between them is within
the run-to-run noise of an LLM-as-judge (three separate runs of v2 landed at
0.9033, 0.9046 and 0.9092). The gains that are *not* noise show up where the
prompt engineering actually bites:

- **F1-Score: +0.05**, the largest and most consistent gain. This is recall —
  v2 preserves technical detail from the report that v1 drops.
- **Complex reports** (examples 13–15, the multi-failure ones): v2 scores
  0.97 / 0.92 / 1.00 on F1 against 0.92 / 0.87 / 0.92 for v1. The depth ruler
  is what produces this: v1 has no instruction to structure a long report, so
  it answers a 2,559-character bug with the same shape it uses for a
  63-character one.
- **Predictability.** v1 gets there by luck of the model; v2 specifies persona,
  format, criteria style and edge cases, so the output shape does not drift
  between runs or between models.

The trade-off is visible too: v2 is more thorough, which costs a little
Clarity, since the judge rewards concision. An explicit concision rule was
added to the prompt for exactly this reason, and it moved Clarity from 0.8967
to 0.9153 in local testing.

---

## C) How to Run

### Prerequisites

- Python 3.9+
- A LangSmith account and API key — <https://smith.langchain.com>
- A Hub handle configured in LangSmith (Settings → LangChain Hub)
- An API key for **one** LLM provider: Google Gemini or OpenAI

### Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

| Variable | Meaning |
|---|---|
| `LANGSMITH_API_KEY` | key from LangSmith → Settings → API Keys (`lsv2_pt_...`) |
| `USERNAME_LANGSMITH_HUB` | your Prompt Hub handle, not your e-mail |
| `LANGSMITH_PROJECT` | project name used for traces and the dataset |
| `LLM_PROVIDER` | `google` or `openai` |
| `GOOGLE_API_KEY` / `OPENAI_API_KEY` | key for the chosen provider |
| `LLM_MODEL` | model that generates the user stories |
| `EVAL_MODEL` | model that scores them (LLM-as-judge) |

### Pipeline

```bash
python src/pull_prompts.py      # 1. pull leonanluppi/bug_to_user_story_v1
                                # 2. edit prompts/bug_to_user_story_v2.yml
python src/push_prompts.py      # 3. publish <handle>/bug_to_user_story_v2 (public)
python src/evaluate.py          # 4. score against the 15-example dataset
```

Validate the prompt structure at any point:

```bash
pytest tests/test_prompts.py
```

### Choosing the models

The challenge fixes no model, and providers retire them often. The
`.env.example` default (`gemini-2.5-flash`) is no longer available to new
accounts — it returns `404 ... no longer available to new users`. List what
your key can actually reach:

```bash
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY" \
  | python -c "import sys, json; [print(m['name']) for m in json.load(sys.stdin)['models']]"
```

The evaluator model dominates the runtime, because each example costs three
judge calls. Measured on this dataset:

| Candidate | Latency per judge call |
|---|---|
| `gemini-3.5-flash-lite` | ~2 s |
| `gemini-3.1-flash-lite` | ~30 s |
| `gemini-3.6-flash` | ~110 s |

All three returned the same score on a control case, so this project uses
`gemini-3.5-flash-lite` for both generation and evaluation.

### Free-tier rate limits

The Gemini free tier allows **15 requests per minute per model**, and a full
evaluation makes roughly 60 calls (15 generations + 45 judge calls). Expect
`429` responses and retries; a complete run takes several minutes. If requests
start failing outright rather than retrying, wait a minute and run again.
