# main.py — MNEMA v2
import sys
from model.loader import load_model_and_tokenizer, verify_base_frozen
from memory.graph import RelationalMemoryGraph
from memory.extractor import MemoryExtractor
from model.inference import chat
from scheduler import MemoryScheduler

# ── Display settings ──────────────────────────────────────────────────────────
SHOW_THINKING  = True
SHOW_MEMORY_TAGS = True
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("\n" + "═" * 55)
    print("  MNEMA v2 — A mind that remembers")
    print("═" * 55 + "\n")

    model, tokenizer = load_model_and_tokenizer()
    verify_base_frozen(model)

    # v2: use RelationalMemoryGraph instead of flat MemoryStore
    memory_graph = RelationalMemoryGraph()
    extractor = MemoryExtractor()

    scheduler = MemoryScheduler(memory_graph, model, tokenizer)
    scheduler.start()

    conversation_history = []
    turn_counter = 0

    print("Type 'quit'     → exit")
    print("Type 'memory'   → inspect memory graph")
    print("Type 'graph'    → show graph stats")
    print("Type 'think on' → show inner monologue")
    print("Type 'think off'→ hide inner monologue")
    print("Type 'clear'    → wipe memory\n")

    global SHOW_THINKING

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # ── Commands ──────────────────────────────────────────────────────
            if user_input.lower() == "quit":
                break

            if user_input.lower() == "memory":
                _show_memories(memory_graph)
                continue

            if user_input.lower() == "graph":
                _show_graph_stats(memory_graph)
                continue

            if user_input.lower() == "think on":
                SHOW_THINKING = True
                print("  [Inner monologue visible]\n")
                continue

            if user_input.lower() == "think off":
                SHOW_THINKING = False
                print("  [Inner monologue hidden]\n")
                continue

            if user_input.lower() == "clear":
                _clear_memory()
                print("  [Memory cleared]\n")
                continue

            turn_counter += 1

            # ── Extract and store memories in the graph ───────────────────────
            new_memories = extractor.extract(user_input, turn_counter)
            for mem in new_memories:
                memory_graph.add(
                    content=mem["content"],
                    memory_type=mem["type"],
                    importance=mem["importance"],
                    turn=turn_counter
                )
                if SHOW_MEMORY_TAGS:
                    print(f"  \033[2m[memory: {mem['type']} · "
                          f"importance={mem['importance']:.1f}]\033[0m")

            # ── Generate response ──────────────────────────────────────────────
            spoken, monologue = chat(
                model, tokenizer, user_input,
                memory_graph, conversation_history,
                show_thinking=SHOW_THINKING
            )

            # ── Display ────────────────────────────────────────────────────────
            if SHOW_THINKING and monologue:
                print(f"\n  \033[2m\033[3m💭 {monologue}\033[0m\n")

            print(f"\nMNEMA: {spoken}\n")

            # ── Update history ─────────────────────────────────────────────────
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": spoken})

            if len(conversation_history) > 2000:
                conversation_history = conversation_history[-2000:]

    except KeyboardInterrupt:
        print("\n\n  Goodbye.")
    finally:
        scheduler.stop()


def _show_memories(graph: RelationalMemoryGraph):
    memories = graph.get_all_for_decay()
    if not memories:
        print("\n  [No memories stored yet]\n")
        return
    print(f"\n  ── Memory Graph ({len(memories)} alive nodes) ──")
    for mem in sorted(memories, key=lambda x: x["strength"], reverse=True)[:10]:
        bar_len = int(mem["strength"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {bar} {mem['strength']:.2f} [{mem['type']}] "
              f"{mem['content'][:60]}...")
    print()


def _show_graph_stats(graph: RelationalMemoryGraph):
    stats = graph.stats()
    print(f"\n  ── Graph Stats ──")
    print(f"  Alive nodes:       {stats['alive_nodes']}")
    print(f"  Superseded nodes:  {stats['superseded_nodes']}")
    print(f"  Total edges:       {stats['total_edges']}")
    print(f"  Contradictions:    {stats['contradictions']}")
    if stats['edge_breakdown']:
        print(f"  Edge breakdown:")
        for edge_type, count in stats['edge_breakdown'].items():
            print(f"    {edge_type:<15} {count}")

    contradictions = graph.get_contradictions()
    if contradictions:
        print(f"\n  ── Resolved Contradictions ──")
        for c in contradictions[:5]:
            print(f"  SUPERSEDED: {c['older_content'][:50]}...")
            print(f"  CURRENT:    {c['newer_content'][:50]}...")
            print(f"  confidence: {c['confidence']:.2f}\n")
    print()


def _clear_memory():
    import os, shutil
    paths = ["./data/memory_graph.db", "./data/chroma", "./data/memory.db"]
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)


if __name__ == "__main__":
    main()
