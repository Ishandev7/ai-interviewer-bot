"""AI Interviewer Bot – command line entry point.

  python main.py ingest --resume path/to/resume.pdf [--jd path/to/job_description.txt]
  python main.py interview [--answers scripted_answers.json]
"""

import argparse
import json
from pathlib import Path

from src import rag
from src.interviewer import Interviewer

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="AI Interviewer Bot (RAG + LLM)")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Load the resume, job description and question bank")
    ing.add_argument("--resume", required=True, help="Resume PDF or text file")
    ing.add_argument("--jd", help="Optional job description (PDF or text)")
    ing.add_argument("--questions", default=str(ROOT / "data" / "question_bank.json"))

    itv = sub.add_parser("interview", help="Run the interview")
    itv.add_argument("--answers", help="JSON list of scripted answers (for testing)")

    args = parser.parse_args()

    if args.command == "ingest":
        print("Ingesting documents...")
        rag.ingest_candidate(args.resume, args.jd)
        rag.ingest_question_bank(args.questions)
        print("Done. Now run: python main.py interview")
    else:
        answers = json.loads(Path(args.answers).read_text(encoding="utf-8")) if args.answers else None
        Interviewer(answer_source=answers).run()


if __name__ == "__main__":
    main()
