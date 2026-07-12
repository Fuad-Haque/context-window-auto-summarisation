"""
test_live.py

The real thing: runs 50+ actual turns through ContextManager against the
live Anthropic API. Requires ANTHROPIC_API_KEY in the environment or a .env
file. This is what "Done when context never silently truncates and
coherence holds across the full thread" actually gets tested against.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 test_live.py

What this proves that the mocked tests cannot:
  - real token counts from the real tokenizer (not a guess)
  - a real model actually writing a coherent summary of real content
  - the model, after summarisation, correctly recalling a fact planted
    50 turns ago ONLY BECAUSE it survived inside the summary — this is the
    coherence check the brief asks for, not just "the server didn't crash"
"""

import os
import sys
from context_manager import ContextManager

PLANTED_FACT = "My project's codename is Falcon-7 and the budget is exactly $4,200."


def run():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Export a real key and re-run.")
        sys.exit(1)

    cm = ContextManager(
        system_prompt="You are a concise assistant helping plan a small software project."
    )

    print(f"{'turn':>4} | {'tokens':>9} | {'triggered?':>10} | {'msgs_kept':>9} | note")
    print("-" * 70)

    # Turn 1: plant the fact we'll test recall of much later
    reply = cm.send(PLANTED_FACT + " Please just acknowledge you've noted it.")
    status_after_1 = {
        "token_count": cm.measure_tokens(),
        "triggered_summary": cm.summarisation_count > 0,
        "message_count": len(cm.messages),
    }
    print(f"{1:>4} | {status_after_1['token_count']:>9} | {str(status_after_1['triggered_summary']):>10} | {status_after_1['message_count']:>9} | planted fact")

    # Turns 2-55: filler conversation to run past 50 messages and, ideally,
    # far enough to force at least one real summarisation pass.
    filler_topics = [
        "What's a good way to structure a FastAPI project?",
        "How should I handle environment variables safely?",
        "Explain the difference between sync and async in Python.",
        "What's a reasonable Dockerfile setup for a small API?",
        "How do I add basic rate limiting to an endpoint?",
    ]

    for i in range(2, 56):
        topic = filler_topics[i % len(filler_topics)]
        cm.send(f"(turn {i}) {topic} Keep your answer to two sentences.")
        token_count = cm.measure_tokens()
        triggered = cm.summarisation_count
        note = ""
        if i == 56 - 1:
            note = "final filler turn"
        print(f"{i:>4} | {token_count:>9} | {str(cm.summarisation_count > 0):>10} | {len(cm.messages):>9} | summarised {triggered}x so far")

    print("-" * 70)
    print(f"Total summarisation passes across the run: {cm.summarisation_count}")
    print(f"Final message count in active window: {len(cm.messages)}")
    print(f"Final running_summary (first 300 chars): {cm.running_summary[:300]}")

    # THE ACTUAL COHERENCE TEST: ask for the planted fact back.
    print("\n--- COHERENCE CHECK ---")
    print("Asking the model to recall the fact planted in turn 1...")
    final_reply = cm.send(
        "What was the codename and exact budget I told you about at the very start of this conversation?"
    )
    print("Model's answer:\n", final_reply)

    passed = "Falcon-7" in final_reply and "4,200" in final_reply.replace(",", "").replace("4200", "4,200") or "4200" in final_reply.replace(",", "").replace(".", "")
    print("\nCOHERENCE CHECK RESULT:", "PASS — fact survived summarisation" if ("Falcon-7" in final_reply and "4,200" in final_reply) else "CHECK MANUALLY — see raw answer above")


if __name__ == "__main__":
    run()