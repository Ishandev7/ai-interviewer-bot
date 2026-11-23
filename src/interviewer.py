"""The interview agent: walks through the rounds in the interview pattern, uses RAG to
pick grounded questions, asks follow-ups and scores every answer."""

import json
from datetime import datetime
from pathlib import Path

from . import rag
from .llm import chat

ROOT = Path(__file__).resolve().parent.parent

GENERATE_PROMPT = """You are {name}, a friendly but rigorous interviewer hiring for the role: {role}.
Current round: {round_name}. Task for this question: {instruction}

Context retrieved for this question:
{context}

Questions already asked (do not repeat them or their topics):
{asked}

Write ONE focused interview question about a single thing, under 30 words, spoken directly
to the candidate. Do not bundle several questions together.
Return JSON: {{"question": "...", "topic": "short topic label",
"expected_points": ["3-5 points a strong answer should cover"]}}"""

EVALUATE_PROMPT = """You are evaluating a candidate for the role: {role}.
Question: {question}
Points a strong answer covers: {expected}
Candidate's answer: {answer}

Score the answer from 1 (poor) to 5 (excellent) on correctness, depth and clarity.
Ask a follow-up only if the answer is vague, incomplete or invites a deeper probe.
Return JSON: {{"score": 1-5, "strengths": ["..."], "gaps": ["..."],
"followup": "one short follow-up question, or empty string"}}"""

REPORT_PROMPT = """You are a hiring manager. Write a concise interview report for the role {role}
from the scored transcript below. Include: overall score out of 5, a recommendation
(Strong Hire / Hire / Lean No Hire / No Hire), top 3 strengths, top 3 areas to improve,
and one line per round. Use markdown headings and bullets.

Transcript (JSON):
{transcript}"""


class Interviewer:
    def __init__(self, pattern_path=ROOT / "config" / "interview_pattern.json", answer_source=None):
        self.pattern = json.loads(Path(pattern_path).read_text(encoding="utf-8"))
        self.answer_source = answer_source  # list of scripted answers for testing, or None for keyboard
        self.transcript = []
        self.used_chunks = set()

    # ---------- answers ----------
    def _get_answer(self):
        if self.answer_source is None:
            return input("\nYou: ").strip() or "(no answer)"
        answer = self.answer_source.pop(0) if self.answer_source else "I'm not sure."
        print(f"\nYou: {answer}")
        return answer

    # ---------- retrieval ----------
    def _context_for(self, round_cfg):
        source = round_cfg.get("source", "none")
        if source == "resume":
            # Use chunks not shown before, so each resume question covers a different project or job.
            hits = rag.search("candidate", "projects, experience, technical decisions and results",
                              k=2, exclude_ids=self.used_chunks)
            self.used_chunks.update(i for i, _, _ in hits)
            return "\n---\n".join(text for _, text, _ in hits)
        if source == "question_bank":
            where = {"topic": round_cfg["topic_filter"]} if "topic_filter" in round_cfg else {"topic": {"$ne": "behavioral"}}
            skills = " ".join(text for _, text, _ in rag.search("candidate", "technical skills and tools", k=2))
            hits = rag.search("questions", skills, k=6, where=where)
            return "\n".join(f"- {text} (strong answer covers: {meta['key_points']})" for _, text, meta in hits)
        return "(no retrieval needed for this round)"

    # ---------- one question ----------
    def _ask(self, round_cfg):
        asked = "\n".join(f"- {t['question']}" for t in self.transcript) or "(none yet)"
        q = chat([{"role": "user", "content": GENERATE_PROMPT.format(
            name=self.pattern["interviewer_name"], role=self.pattern["role"],
            round_name=round_cfg["name"], instruction=round_cfg["instruction"],
            context=self._context_for(round_cfg), asked=asked)}], json_mode=True)

        print(f"\n{self.pattern['interviewer_name']}: {q['question']}")
        answer = self._get_answer()
        entry = {"round": round_cfg["name"], "topic": q.get("topic", ""),
                 "question": q["question"], "answer": answer}

        if round_cfg["name"] != "Wrap-up":
            ev = self._evaluate(q["question"], q.get("expected_points", []), answer)
            if ev.get("followup") and self.pattern.get("max_followups_per_question", 0) > 0:
                print(f"\n{self.pattern['interviewer_name']}: {ev['followup']}")
                follow_answer = self._get_answer()
                entry["followup"] = {"question": ev["followup"], "answer": follow_answer}
                ev = self._evaluate(q["question"] + " Follow-up: " + ev["followup"],
                                    q.get("expected_points", []), answer + " | Follow-up answer: " + follow_answer)
            entry.update(score=ev["score"], strengths=ev.get("strengths", []), gaps=ev.get("gaps", []))
        self.transcript.append(entry)

    def _evaluate(self, question, expected, answer):
        return chat([{"role": "user", "content": EVALUATE_PROMPT.format(
            role=self.pattern["role"], question=question,
            expected="; ".join(expected), answer=answer)}], temperature=0.2, json_mode=True)

    # ---------- full interview ----------
    def run(self):
        print(f"\n=== {self.pattern['role']} interview with {self.pattern['interviewer_name']} ===")
        print("(Press Ctrl+C at any time to stop; a report is still saved for the answers so far.)")
        try:
            for round_cfg in self.pattern["rounds"]:
                print(f"\n--- Round: {round_cfg['name']} ---")
                for _ in range(round_cfg["questions"]):
                    self._ask(round_cfg)
        except KeyboardInterrupt:
            print("\n\nInterview stopped early.")
            if not self.transcript:
                return None
        return self._report()

    def _report(self):
        report = chat([{"role": "user", "content": REPORT_PROMPT.format(
            role=self.pattern["role"], transcript=json.dumps(self.transcript, indent=1))}],
            temperature=0.3, max_tokens=1200)
        scores = [t["score"] for t in self.transcript if "score" in t]
        avg = sum(scores) / len(scores) if scores else 0

        out_dir = ROOT / "reports"
        out_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (out_dir / f"interview_{stamp}.json").write_text(json.dumps(self.transcript, indent=2), encoding="utf-8")
        md_path = out_dir / f"interview_{stamp}.md"
        md_path.write_text(f"# Interview report\n\nAverage score: {avg:.1f} / 5\n\n{report}\n", encoding="utf-8")
        print(f"\n=== Interview finished. Average score: {avg:.1f} / 5 ===\nReport saved to {md_path}")
        return md_path
