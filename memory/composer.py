# memory/composer.py
"""
MNEMA v2 — Context Composer

Replaces the flat memory injection in inference.py.
Takes graph retrieval results and formats them into structured,
relationship-aware prompt context.

Instead of:
    [vivid] USER stated: My name is Ken

It produces:
    [CORE IDENTITY — vivid]
    Ken is the user's name. (confirmed, never contradicted)

    [REFINED BELIEF — updated 2 turns ago]
    Ken prefers concise responses over detailed ones.
    ↳ refines earlier statement: "Ken prefers casual conversations"

    [CONTRADICTION RESOLVED]
    Ken works as a developer. (current belief)
    ↳ supersedes: "Ken is a student" (no longer held)

    [CONNECTED CONTEXT — 1 hop]
    Ken has mentioned Python several times in relation to work.
"""

import time
from dataclasses import dataclass


# ── Relation labels for prompt display ──────────────────────────────────────
RELATION_LABELS = {
    "direct":      "DIRECT MEMORY",
    "temporal":    "CONNECTED — happened around the same time",
    "causal":      "CONNECTED — causally related",
    "refines":     "REFINEMENT — updates an earlier belief",
    "contradicts": "CONTRADICTION RESOLVED — current belief",
    "depends_on":  "CONNECTED — context dependency",
}


@dataclass
class ComposedContext:
    """The result of composing graph memories into prompt-ready text."""
    prompt_block: str       # what gets injected into the system prompt
    memory_count: int       # total memories included
    contradiction_count: int
    hop_breakdown: dict     # {0: n_direct, 1: n_one_hop, 2: n_two_hop}
    token_estimate: int     # rough token count of the block


class ContextComposer:
    """
    Converts graph retrieval results into structured, readable prompt context.
    Token-aware — will trim to budget if needed.
    """

    def __init__(self, max_tokens: int = 600):
        self.max_tokens = max_tokens

    def compose(self, memories: list[dict], query: str = "") -> ComposedContext:
        """
        Takes graph retrieval results and returns a ComposedContext.
        memories: list of dicts from RelationalMemoryGraph.retrieve()
        """
        if not memories:
            return ComposedContext(
                prompt_block="",
                memory_count=0,
                contradiction_count=0,
                hop_breakdown={},
                token_estimate=0
            )

        # ── Separate by hop distance and relation ────────────────────────────
        direct = [m for m in memories if m["hop"] == 0]
        one_hop = [m for m in memories if m["hop"] == 1]
        two_hop = [m for m in memories if m["hop"] == 2]
        contradictions = [m for m in memories if m["relation"] == "contradicts"]
        refinements = [m for m in memories if m["relation"] == "refines"]

        lines = []

        # ── Direct memories (most important) ────────────────────────────────
        if direct:
            lines.append("WHAT YOU KNOW ABOUT THIS PERSON:")
            for mem in direct:
                strength_label = self._strength_label(mem["strength"])
                lines.append(f"  [{strength_label}] {mem['content']}")
            lines.append("")

        # ── Refinements — updated beliefs ────────────────────────────────────
        if refinements:
            lines.append("UPDATED BELIEFS (these override older memories):")
            for mem in refinements:
                lines.append(f"  [UPDATED] {mem['content']}")
            lines.append("")

        # ── Contradictions — resolved beliefs ────────────────────────────────
        if contradictions:
            lines.append("RESOLVED CONTRADICTIONS (trust the newer belief):")
            for mem in contradictions:
                lines.append(f"  [CURRENT] {mem['content']}")
            lines.append("")

        # ── Connected context (1-2 hops) ─────────────────────────────────────
        connected = [m for m in one_hop + two_hop
                     if m["relation"] not in ("contradicts", "refines")]
        if connected:
            lines.append("RELATED CONTEXT:")
            for mem in connected[:4]:   # cap at 4 to save tokens
                relation = RELATION_LABELS.get(mem["relation"], mem["relation"])
                lines.append(f"  [{relation}] {mem['content']}")
            lines.append("")

        prompt_block = "\n".join(lines).strip()

        # ── Token budget trim ─────────────────────────────────────────────────
        # Rough estimate: 1 token ≈ 4 characters
        token_estimate = len(prompt_block) // 4
        if token_estimate > self.max_tokens:
            # Trim to budget — drop connected context first, then two-hop
            lines_trimmed = []
            running = 0
            for line in lines:
                line_tokens = len(line) // 4
                if running + line_tokens > self.max_tokens:
                    break
                lines_trimmed.append(line)
                running += line_tokens
            prompt_block = "\n".join(lines_trimmed).strip()
            token_estimate = running

        return ComposedContext(
            prompt_block=prompt_block,
            memory_count=len(memories),
            contradiction_count=len(contradictions),
            hop_breakdown={
                0: len(direct),
                1: len(one_hop),
                2: len(two_hop)
            },
            token_estimate=token_estimate
        )

    def format_for_system_prompt(self, context: ComposedContext) -> str:
        """
        Wraps the composed context block for injection into system prompt.
        Returns empty string if no memories.
        """
        if not context.prompt_block:
            return ""

        header = "--- MEMORY ---"
        footer = "--- END MEMORY ---"

        notes = []
        if context.contradiction_count > 0:
            notes.append(
                f"Note: {context.contradiction_count} belief(s) were updated "
                f"— trust the CURRENT belief over older ones."
            )
        if context.hop_breakdown.get(1, 0) + context.hop_breakdown.get(2, 0) > 0:
            notes.append(
                "Connected context shows memories related to the direct ones "
                "— use them for richer understanding, not as primary facts."
            )

        note_block = "\n".join(notes)
        if note_block:
            return f"{header}\n{context.prompt_block}\n\n{note_block}\n{footer}"
        return f"{header}\n{context.prompt_block}\n{footer}"

    @staticmethod
    def _strength_label(strength: float) -> str:
        if strength >= 0.8:
            return "VIVID"
        elif strength >= 0.5:
            return "clear"
        elif strength >= 0.2:
            return "fading"
        else:
            return "dim"
