# AI Interviewer Bot

An AI interviewer that reads a candidate's resume, follows a configurable interview pattern,
asks grounded questions using Retrieval-Augmented Generation (RAG), asks follow-ups and scores
every answer. At the end it writes a hiring report.

## How it works

```
Resume PDF / JD ──► chunk (LangChain splitter) ──► embed (MiniLM, local) ──► Chroma ("candidate")
Question bank   ──────────────────────────────► embed ──────────────────► Chroma ("questions")

For each round in config/interview_pattern.json:
  retrieve context (resume chunks or matching bank questions)
    └─► LLM writes one question ──► candidate answers ──► LLM scores (1–5, strengths, gaps)
                                                           └─► optional follow-up question
End: LLM writes the report ──► reports/interview_<time>.md + transcript .json
```

| Part | File | What it does |
|---|---|---|
| Retrieval | `src/rag.py` | Chunks and embeds documents, semantic search with metadata filters, skips chunks already used |
| Interview agent | `src/interviewer.py` | Runs the rounds, generates questions, evaluates answers, asks follow-ups, writes the report |
| LLM | `src/llm.py` | Groq API (`openai/gpt-oss-120b`), JSON mode, retries on rate limits |
| Interview pattern | `config/interview_pattern.json` | Rounds, number of questions, retrieval source, instructions |
| Question bank | `data/question_bank.json` | Questions by topic with the key points a strong answer covers |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env       # free key from console.groq.com
```

## Run

```bash
# 1. Load the resume (and optionally a job description) plus the question bank
python main.py ingest --resume path/to/resume.pdf --jd path/to/job_description.txt

# 2. Start the interview (type your answers)
python main.py interview

# Testing: feed scripted answers instead of typing
python main.py interview --answers sample_answers.json
```

The first `ingest` downloads the free local embedding model (~80 MB) once.

## Customise

- **Interview pattern:** edit `config/interview_pattern.json` to add rounds, change question counts or use `"topic_filter"` to restrict a round to one topic.
- **Question bank:** add entries to `data/question_bank.json` with `topic`, `question` and `key_points`.
- **Model:** set `GROQ_MODEL` in `.env`.

## Ideas for next versions

- Voice mode (speech-to-text and text-to-speech) so the interview runs by voice
- Web UI (FastAPI + simple frontend)
- Evaluation set to measure scoring consistency against human ratings
