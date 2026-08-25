"""Local eval runner (no LangSmith required).

Guardrail categories are graded deterministically and run cheaply (one LLM
call each, via the guardrail functions directly rather than the full graph).
Other categories are run-and-reported for manual review — see evaluators.py
for why they aren't auto-graded yet.

Run with: uv run python -m eval.run

A small delay runs between cases, and each call retries with backoff —
OpenRouter's free-tier shared pool has been observed to stall or return
transient upstream 502s under rapid back-to-back requests.
"""

import time

from tenacity import retry, stop_after_attempt, wait_exponential

from cga.graph.build_graph import ask
from cga.graph.guardrails.scope_classifier import classify_scope
from cga.graph.guardrails.urgent_check import is_urgent
from cga.graph.llm import get_chat_model
from eval.dataset import EVAL_CASES
from eval.evaluators import evaluate_scope, evaluate_urgent

_retry = retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
_call_urgent = _retry(is_urgent)
_call_scope = _retry(classify_scope)
_call_ask = _retry(ask)


def run() -> None:
    llm = get_chat_model()
    passed = 0
    graded = 0

    # Cheap single-call graded cases first (most reliable), full-pipeline
    # report-only cases last (multiple back-to-back calls, more exposed to
    # free-tier shared-pool stalls).
    cases = sorted(
        EVAL_CASES,
        key=lambda c: 0 if c.category in ("urgent_symptom_detection", "scope_guardrail") else 1,
    )

    for case in cases:
        try:
            if case.category == "urgent_symptom_detection":
                actual = _call_urgent(llm, case.question)
                ok = evaluate_urgent(case, actual)
                graded += 1
                passed += ok
                print(f"[{'PASS' if ok else 'FAIL'}] {case.id}: urgent={actual} (expected {case.expect_urgent})")

            elif case.category == "scope_guardrail" and case.expect_scope is not None:
                actual = _call_scope(llm, case.question)
                ok = evaluate_scope(case, actual)
                graded += 1
                passed += ok
                print(f"[{'PASS' if ok else 'FAIL'}] {case.id}: scope={actual} (expected {case.expect_scope})")

            else:
                result = _call_ask(case.question, thread_id=case.id)
                summary = result.get("redirect_reason") or result.get("comparison") or "?"
                print(f"[REPORT] {case.id} ({case.category}): {summary} — {(result.get('answer') or '')[:80]}")
        except Exception as exc:  # noqa: BLE001 — one flaky case shouldn't kill the whole run
            print(f"[ERROR] {case.id}: {exc}")

        time.sleep(3)

    print(f"\n{passed}/{graded} graded cases passed.")


if __name__ == "__main__":
    run()
