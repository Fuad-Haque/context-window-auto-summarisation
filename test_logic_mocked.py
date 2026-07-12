"""
test_logic_mocked.py

Verifies the CONTROL FLOW of ContextManager without a live API key:
  - does should_summarise() fire at the right count?
  - does summarise() correctly collapse old messages and keep recent ones?
  - does the system prompt actually contain the injected summary afterward?

This does NOT prove real token counts or real summary quality — that
requires ANTHROPIC_API_KEY and is what test_live.py (Section 4 Step 3) is
for. This file exists so the LOGIC is verified before spending a single
real API call on it.
"""

from unittest.mock import MagicMock, patch
from context_manager import ContextManager, CONTEXT_WINDOW, KEEP_RECENT_TURNS


def fake_count_tokens_response(n):
    resp = MagicMock()
    resp.input_tokens = n
    return resp


def fake_create_response(text):
    resp = MagicMock()
    block = MagicMock()
    block.text = text
    resp.content = [block]
    return resp


def test_threshold_math():
    print("--- test_threshold_math ---")
    with patch("context_manager.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        cm = ContextManager(system_prompt="You are a test bot.")

        # below threshold
        client.messages.count_tokens.return_value = fake_count_tokens_response(799_999)
        assert cm.should_summarise() is False, "should NOT trigger at 799,999"
        print("799,999 tokens -> should_summarise() =", cm.should_summarise(), "(expected False)")

        # exactly at threshold
        client.messages.count_tokens.return_value = fake_count_tokens_response(800_000)
        assert cm.should_summarise() is True, "SHOULD trigger at exactly 800,000 (80% of 1M)"
        print("800,000 tokens -> should_summarise() =", cm.should_summarise(), "(expected True)")

        # above threshold
        client.messages.count_tokens.return_value = fake_count_tokens_response(950_000)
        assert cm.should_summarise() is True
        print("950,000 tokens -> should_summarise() =", cm.should_summarise(), "(expected True)")
    print("PASS\n")


def test_summarise_collapses_correctly():
    print("--- test_summarise_collapses_correctly ---")
    with patch("context_manager.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        cm = ContextManager(system_prompt="You are a test bot.")

        # seed 10 fake messages (more than KEEP_RECENT_TURNS=6)
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            cm.messages.append({"role": role, "content": f"message number {i}"})

        assert len(cm.messages) == 10
        client.messages.create.return_value = fake_create_response(
            "User and assistant discussed messages 0 through 3 in a test sequence."
        )

        summary = cm.summarise()

        print("messages before summarise():", 10)
        print("messages after summarise():", len(cm.messages))
        print("KEEP_RECENT_TURNS constant:", KEEP_RECENT_TURNS)
        assert len(cm.messages) == KEEP_RECENT_TURNS, f"expected {KEEP_RECENT_TURNS} messages left, got {len(cm.messages)}"

        # the messages that remain must be the LAST 6, not the first 6
        remaining_contents = [m["content"] for m in cm.messages]
        expected_remaining = [f"message number {i}" for i in range(4, 10)]
        assert remaining_contents == expected_remaining, f"wrong messages kept: {remaining_contents}"
        print("remaining messages are the correct (most recent) ones:", remaining_contents)

        assert cm.running_summary == summary
        assert summary in cm._effective_system_prompt()
        print("summary correctly injected into effective system prompt: True")
        print("summarisation_count incremented to:", cm.summarisation_count)
        assert cm.summarisation_count == 1
    print("PASS\n")


def test_no_collapse_when_under_keep_threshold():
    print("--- test_no_collapse_when_under_keep_threshold ---")
    with patch("context_manager.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        cm = ContextManager(system_prompt="You are a test bot.")

        for i in range(3):  # fewer than KEEP_RECENT_TURNS
            cm.messages.append({"role": "user", "content": f"msg {i}"})

        result = cm.summarise()
        print("messages count (should be unchanged at 3):", len(cm.messages))
        assert len(cm.messages) == 3, "summarise() should not touch messages when count <= KEEP_RECENT_TURNS"
        assert result == ""  # no summary produced yet
        assert client.messages.create.called is False, "should not have called the model at all"
        print("model was NOT called (correct — nothing old enough to summarise)")
    print("PASS\n")


def test_add_turn_triggers_end_to_end():
    print("--- test_add_turn_triggers_end_to_end ---")
    with patch("context_manager.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        cm = ContextManager(system_prompt="You are a test bot.")

        for i in range(10):
            cm.messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"seed {i}"})

        # First measure_tokens call (pre-summary check) returns over threshold,
        # second measure_tokens call (post-summary re-check) returns under threshold.
        client.messages.count_tokens.side_effect = [
            fake_count_tokens_response(850_000),
            fake_count_tokens_response(120_000),
        ]
        client.messages.create.return_value = fake_create_response("Collapsed summary of seed 0-7.")

        status = cm.add_turn("user", "one more message that pushes us over")

        print("status returned by add_turn():")
        for k, v in status.items():
            print(f"  {k}: {v}")

        assert status["triggered_summary"] is True
        assert status["token_count"] == 120_000, "should report the POST-summary count, not the pre-summary one"
        assert status["message_count"] == KEEP_RECENT_TURNS
    print("PASS\n")


if __name__ == "__main__":
    test_threshold_math()
    test_summarise_collapses_correctly()
    test_no_collapse_when_under_keep_threshold()
    test_add_turn_triggers_end_to_end()
    print("=== ALL LOGIC TESTS PASSED ===")