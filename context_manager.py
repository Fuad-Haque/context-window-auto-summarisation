"""
context_manager.py

Owns three responsibilities and nothing else:
  1. measure_tokens()   -> ask the real API how many tokens this conversation costs
  2. should_summarise() -> compare that number against the 80% threshold
  3. summarise()         -> collapse older messages into one system-level paragraph

ContextManager wraps a conversation list. Every call to add_turn() re-measures
and, if the threshold is crossed, rewrites self.messages in place: old turns
are replaced by a single summary message, recent turns are kept verbatim.
"""

import os
from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
CONTEXT_WINDOW = 1_000_000          # claude-sonnet-4-6, API access, confirmed above
THRESHOLD_RATIO = 0.80              # trigger point stated in the brief
KEEP_RECENT_TURNS = 6               # verbatim turns preserved after each summarisation pass


class ContextManager:
    def __init__(self, system_prompt: str = "", api_key: str | None = None):
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.base_system_prompt = system_prompt
        self.running_summary = ""    # empty until the first summarisation pass fires
        self.messages: list[dict] = []
        self.summarisation_count = 0  # how many times summarise() has fired this session

    # ---- 1. measurement --------------------------------------------------

    def measure_tokens(self) -> int:
        """
        Ask the real Anthropic count_tokens endpoint how many input tokens
        the CURRENT system prompt + message list would cost. This is not an
        estimate function — it is a live API call. That is the whole point:
        no heuristic, no tiktoken, no guessing. The number that comes back
        is the number the Messages API would actually bill you for as input.
        """
        response = self.client.messages.count_tokens(
            model=MODEL,
            system=self._effective_system_prompt(),
            messages=self.messages if self.messages else [{"role": "user", "content": "."}],
        )
        return response.input_tokens

    def _effective_system_prompt(self) -> str:
        if self.running_summary:
            return (
                f"{self.base_system_prompt}\n\n"
                f"--- CONVERSATION SUMMARY (earlier turns, compressed) ---\n"
                f"{self.running_summary}"
            ).strip()
        return self.base_system_prompt

    # ---- 2. trigger --------------------------------------------------

    def should_summarise(self, token_count: int | None = None) -> bool:
        count = token_count if token_count is not None else self.measure_tokens()
        return count >= int(CONTEXT_WINDOW * THRESHOLD_RATIO)

    # ---- 3. summarise + inject --------------------------------------------------

    def summarise(self) -> str:
        """
        Collapse every message except the last KEEP_RECENT_TURNS into one
        summary paragraph, written by the model itself, then splice that
        summary into the running system prompt and drop the collapsed
        messages from self.messages.

        This is a REAL model call — same client, same API — not a canned
        string. Zero-Budget Mode: this uses the one provider you already
        pay for per-message, no separate summarisation service.
        """
        if len(self.messages) <= KEEP_RECENT_TURNS:
            return self.running_summary  # nothing old enough to collapse yet

        split_point = len(self.messages) - KEEP_RECENT_TURNS
        old_messages = self.messages[:split_point]
        recent_messages = self.messages[split_point:]

        transcript_lines = []
        for m in old_messages:
            role = m["role"].upper()
            content = m["content"] if isinstance(m["content"], str) else str(m["content"])
            transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)

        prior = f"\n\nPRIOR SUMMARY (fold this in, do not drop facts from it):\n{self.running_summary}" if self.running_summary else ""

        summarisation_prompt = (
            "Summarise the following conversation excerpt into a single dense paragraph. "
            "Preserve: names, numbers, decisions made, and anything the user asked to be "
            "remembered. Do not add commentary, do not say 'the user said' repeatedly, "
            "write it as continuous background context a new participant could read once "
            "and be fully caught up. Target under 200 words."
            f"{prior}\n\nCONVERSATION EXCERPT:\n{transcript}"
        )

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": summarisation_prompt}],
        )
        new_summary = response.content[0].text.strip()

        self.running_summary = new_summary
        self.messages = recent_messages
        self.summarisation_count += 1
        return new_summary

    # ---- public entrypoint --------------------------------------------------

    def add_turn(self, role: str, content: str) -> dict:
        """
        Append a message, then check the threshold. If crossed, summarise
        BEFORE returning, so the caller's next API call already sees the
        compressed context. Returns a status dict for logging/testing.
        """
        self.messages.append({"role": role, "content": content})

        token_count = self.measure_tokens()
        triggered = self.should_summarise(token_count)

        summary_result = None
        if triggered:
            summary_result = self.summarise()
            token_count = self.measure_tokens()  # re-measure post-collapse

        return {
            "token_count": token_count,
            "threshold": int(CONTEXT_WINDOW * THRESHOLD_RATIO),
            "triggered_summary": triggered,
            "summary_text": summary_result,
            "message_count": len(self.messages),
            "summarisation_count": self.summarisation_count,
        }

    def send(self, user_text: str) -> str:
        """
        Full turn: add user message (with threshold check), call the real
        model with the effective system prompt + current message window,
        append the assistant reply, return the reply text.
        """
        self.add_turn("user", user_text)

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=self._effective_system_prompt(),
            messages=self.messages,
        )
        reply_text = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply_text})
        return reply_text