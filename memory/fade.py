# memory/fade.py
"""
MNEMA v2 — Multi-Speed Forgetting

Replaces single-rate Ebbinghaus decay with five biologically-inspired
memory tiers, each with distinct timescales and decay behaviors.

Based on Atkinson-Shiffrin memory model (1968) + Ebbinghaus forgetting curve.

Memory Tiers:
┌─────────────────┬──────────────────┬───────────────────────────────────┐
│ Tier            │ Timescale        │ Maps to                           │
├─────────────────┼──────────────────┼───────────────────────────────────┤
│ sensory         │ seconds          │ raw input fragments (auto-fades)  │
│ short_term      │ minutes–hours    │ casual chat, small talk           │
│ episodic        │ hours–days       │ facts, events, specific exchanges │
│ semantic        │ weeks–months     │ preferences, patterns, knowledge  │
│ identity        │ near-permanent   │ corrections, name, core beliefs   │
└─────────────────┴──────────────────┴───────────────────────────────────┘

Decay formula per tier:
    strength(t) = S₀ × e^(−λ_tier × Δt_hours)

Where λ_tier is determined by:
    1. Base tier rate
    2. Importance modifier (high importance → slower decay)
    3. Access count modifier (frequently accessed → slower decay)

Reinforcement on access:
    new_strength = old + boost × (1.0 - old)
    Where boost varies by tier — identity memories are harder to reinforce
    further (already near-permanent) and sensory memories reinforce strongly
    (attention spikes them temporarily).
"""

import math
import time
from config import cfg


# ── Five memory tiers ────────────────────────────────────────────────────────

TIERS = {
    "sensory": {
        "base_lambda":      24.0,    # decays in seconds-to-minutes
        "half_life_hours":  0.02,    # ~1 minute half-life
        "reinforce_boost":  0.6,     # attention spikes it strongly
        "description":      "raw fragments, fades almost immediately",
    },
    "short_term": {
        "base_lambda":      2.0,     # decays in hours
        "half_life_hours":  0.35,    # ~20 minute half-life
        "reinforce_boost":  0.5,
        "description":      "casual chat, small talk, passing mentions",
    },
    "episodic": {
        "base_lambda":      0.08,    # decays over days
        "half_life_hours":  8.7,     # ~9 hour half-life
        "reinforce_boost":  0.4,
        "description":      "specific facts, events, named exchanges",
    },
    "semantic": {
        "base_lambda":      0.008,   # decays over weeks
        "half_life_hours":  86.6,    # ~3.6 day half-life
        "reinforce_boost":  0.3,
        "description":      "preferences, patterns, general knowledge",
    },
    "identity": {
        "base_lambda":      0.0003,  # near-permanent
        "half_life_hours":  2310.0,  # ~96 day half-life
        "reinforce_boost":  0.15,    # already strong, small boosts
        "description":      "name, corrections, core beliefs, values",
    },
}

# ── Memory type → tier mapping ───────────────────────────────────────────────
# Maps v1 memory types to the appropriate tier

TYPE_TO_TIER = {
    "correction":  "identity",    # corrections = near-permanent
    "preference":  "semantic",    # preferences = weeks timescale
    "fact":        "episodic",    # facts = days timescale
    "casual":      "short_term",  # casual chat = hours timescale
    "event":       "episodic",    # events = days timescale
    "sensory":     "sensory",     # raw fragments = minutes
}

# Default tier for unknown types
DEFAULT_TIER = "episodic"


class MultiSpeedDecay:
    """
    Five-tier memory decay engine.
    Drop-in replacement for EbbinghausDecay — same interface,
    richer behavior.
    """

    def __init__(self, store):
        """
        store: RelationalMemoryGraph or any object with:
               - get_all_for_decay() → list[dict]
               - update_strength(id, strength)
        """
        self.store = store

    def run_decay_pass(self) -> int:
        """
        Apply tier-appropriate forgetting curve to all alive memories.
        Called every cfg.decay_interval_hours hours by the scheduler.

        Returns number of memories archived (strength fell below threshold).
        """
        now = time.time()
        memories = self.store.get_all_for_decay()
        archived_count = 0
        tier_counts = {tier: 0 for tier in TIERS}

        for mem in memories:
            # ── Determine tier ───────────────────────────────────────────────
            mem_type = mem.get("type", "fact")
            tier_name = TYPE_TO_TIER.get(mem_type, DEFAULT_TIER)
            tier = TIERS[tier_name]

            # ── Time since last access ───────────────────────────────────────
            last_accessed = mem.get("last_accessed", mem.get("created_at", now))
            hours_elapsed = (now - last_accessed) / 3600.0

            # ── Effective λ: base rate modulated by importance ───────────────
            importance = mem.get("importance", 0.5)
            # High importance → up to 60% slower decay
            importance_modifier = 1.0 - (0.6 * importance)
            effective_lambda = tier["base_lambda"] * importance_modifier

            # ── Forgetting curve ─────────────────────────────────────────────
            current_strength = mem.get("strength", 1.0)
            new_strength = current_strength * math.exp(
                -effective_lambda * hours_elapsed
            )
            new_strength = max(0.0, min(1.0, new_strength))

            # ── Update ───────────────────────────────────────────────────────
            self.store.update_strength(mem["id"], new_strength)
            tier_counts[tier_name] += 1

            if new_strength < cfg.min_strength_threshold:
                archived_count += 1

        # ── Summary ──────────────────────────────────────────────────────────
        total = len(memories)
        tier_summary = ", ".join(
            f"{t}={c}" for t, c in tier_counts.items() if c > 0
        )
        print(f"[Decay] {total} memories processed "
              f"({tier_summary}) — {archived_count} archived")

        return archived_count

    def get_tier(self, memory_type: str) -> str:
        """Return the tier name for a given memory type."""
        return TYPE_TO_TIER.get(memory_type, DEFAULT_TIER)

    def get_tier_info(self, memory_type: str) -> dict:
        """Return full tier config for a given memory type."""
        return TIERS[self.get_tier(memory_type)]

    def reinforce(self, mem_id: str, memory_type: str,
                  current_strength: float) -> float:
        """
        Boost a memory's strength on access (spaced repetition).
        Uses tier-specific boost rate — identity memories get smaller
        boosts since they're already near-permanent.

        Returns new strength value.
        """
        tier = TIERS[self.get_tier(memory_type)]
        boost = tier["reinforce_boost"]
        new_strength = current_strength + boost * (1.0 - current_strength)
        new_strength = min(1.0, new_strength)
        self.store.update_strength(mem_id, new_strength)
        return new_strength

    def estimate_survival_hours(self, memory_type: str,
                                 importance: float = 0.5,
                                 current_strength: float = 1.0) -> float:
        """
        Estimate how many hours until a memory falls below threshold.
        Useful for debugging and memory inspection.
        """
        tier = TIERS[self.get_tier(memory_type)]
        importance_modifier = 1.0 - (0.6 * importance)
        effective_lambda = tier["base_lambda"] * importance_modifier

        if effective_lambda <= 0:
            return float("inf")

        # Solve: threshold = strength * e^(-λt)  →  t = -ln(threshold/strength) / λ
        threshold = cfg.min_strength_threshold
        if current_strength <= threshold:
            return 0.0

        return -math.log(threshold / current_strength) / effective_lambda

    def tier_summary(self) -> dict:
        """
        Return a summary of all tiers with half-lives for display.
        """
        return {
            tier_name: {
                "half_life": f"{info['half_life_hours']:.1f}h",
                "description": info["description"],
            }
            for tier_name, info in TIERS.items()
        }


# ── Backward compatibility alias ─────────────────────────────────────────────
# Old code that imports EbbinghausDecay will still work
EbbinghausDecay = MultiSpeedDecay
