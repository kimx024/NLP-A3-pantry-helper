"""
Kitchen Assistant - Anti-Waste Pantry Tool
Terminal-based multi-turn conversational interface using Qwen2.5-3B-Instruct via Ollama.
"""

import ollama
import csv
import os
from datetime import date

MODEL = "qwen2.5:3b"

# ── Pantry Management ────────────────────────────────────────────────────────

def load_pantry_from_csv(filepath: str) -> list[dict]:
    """Load pantry ingredients from a CSV file."""
    pantry = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pantry.append({
                "ingredients": row["ingredients"].strip(),
                "quantity": row.get("quantity", "").strip(),
                "expiry": row.get("expiry_date", "").strip(),
            })
    return pantry


def enter_pantry_manually() -> list[dict]:
    """Let the user type in ingredients one by one."""
    pantry = []
    print("\nEnter your ingredients. Type 'done' when finished.")
    print("Format: ingredient name, quantity, expiry date (YYYY-MM-DD)")
    print("Example: spinach, 100g, 2026-05-10\n")
    while True:
        line = input("  > ").strip()
        if line.lower() == "done":
            break
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 1 or not parts[0]:
            print("  Please enter at least an ingredient name.")
            continue
        pantry.append({
            "ingredients": parts[0],
            "quantity": parts[1] if len(parts) > 1 else "",
            "expiry": parts[2] if len(parts) > 2 else "",
        })
        print(f"  Added: {parts[0]}")
    return pantry


def format_pantry_for_prompt(pantry: list[dict]) -> str:
    """Format pantry as a readable list for injection into the system prompt."""
    today = date.today()
    lines = []
    for item in pantry:
        line = f"- {item['ingredients']}"
        if item["quantity"]:
            line += f" ({item['quantity']})"
        if item["expiry"]:
            try:
                exp = date.fromisoformat(item["expiry"])
                days_left = (exp - today).days
                if days_left <= 0:
                    line += f" [EXPIRES TODAY or EXPIRED]"
                elif days_left <= 3:
                    line += f" [expires in {days_left} day(s) — use soon!]"
                else:
                    line += f" [expires {item['expiry']}]"
            except ValueError:
                line += f" [expiry: {item['expiry']}]"
        lines.append(line)
    return "\n".join(lines)


def show_pantry(pantry: list[dict]):
    """Print the current pantry to the terminal."""
    if not pantry:
        print("  (pantry is empty)")
        return
    print(format_pantry_for_prompt(pantry))


# ── System Prompt ────────────────────────────────────────────────────────────

def build_system_prompt(pantry: list[dict], strategy: str = "zero-shot") -> str:
    """
    Build the system prompt based on the prompting strategy.
    Strategies: zero-shot, few-shot, chain-of-thought
    """
    pantry_str = format_pantry_for_prompt(pantry)
    ingredient_names = ", ".join(
        [item["ingredients"].lower() for item in pantry]
    )

    base_instruction = f"""You are a kitchen assistant helping a student reduce food waste.
The user's current pantry contains ONLY these ingredients:

{pantry_str}

STRICT RULES:
- Only suggest recipes or substitutions using ingredients listed above.
- Do NOT introduce any ingredient not in the pantry.
- Always prioritise ingredients that expire soonest.
- Keep suggestions simple and student-friendly.
- Available ingredients: {ingredient_names}"""

    if strategy == "zero-shot":
        return base_instruction

    elif strategy == "few-shot":
        examples = """
EXAMPLES of good responses:

Example 1:
Pantry: eggs (3), butter (50g, expires soon), flour (200g)
User: What can I make?
Assistant: You could make simple crepes. Beat the eggs, melt the butter, mix with flour and a splash of water, then fry thin pancakes. This uses up the butter before it expires.

Example 2:
Pantry: olive oil (100ml), lemon (1), garlic (2 cloves)
User: I need butter for my pasta sauce but I don't have any. What can I use?
Assistant: Use olive oil instead. It won't give the same richness but works well in a savory sauce. Add a squeeze of lemon at the end to brighten it up.

Example 3:
Pantry: spinach (100g, expires tomorrow), eggs (2), olive oil (50ml)
User: What should I make first?
Assistant: Use the spinach today — it expires tomorrow. A quick spinach omelette works well: wilt the spinach in olive oil, beat the eggs, pour over, and fold."""
        return base_instruction + "\n" + examples

    elif strategy == "chain-of-thought":
        cot_instruction = """
Before giving your answer, reason step by step:
1. Identify which ingredients expire soonest.
2. Consider what role each ingredient can play (protein, fat, carb, flavouring).
3. Think about what combinations make sense given only what is available.
4. Then give your final suggestion.

Always show your reasoning before the final answer."""
        return base_instruction + "\n" + cot_instruction

    return base_instruction


# ── Chat Loop ────────────────────────────────────────────────────────────────

def chat(pantry: list[dict], strategy: str = "zero-shot"):
    """Run the multi-turn conversational loop."""
    print(f"\n── Kitchen Assistant ready (strategy: {strategy}) ──")
    print("Type your question. Commands: 'pantry' to show pantry, 'quit' to exit.\n")

    system_prompt = build_system_prompt(pantry, strategy)
    history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "pantry":
            print("\nCurrent pantry:")
            show_pantry(pantry)
            print()
            continue

        history.append({"role": "user", "content": user_input})

        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}] + history,
            options={"temperature": 0.7},
        )

        assistant_message = response["message"]["content"]
        history.append({"role": "assistant", "content": assistant_message})
        print(f"\nAssistant: {assistant_message}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Kitchen Assistant — Anti-Waste Pantry Tool")
    print("=" * 50)

    # Prompting strategy selection
    print("\nSelect prompting strategy:")
    print("  1. Zero-shot")
    print("  2. Few-shot")
    print("  3. Chain-of-thought")
    strategy_map = {"1": "zero-shot", "2": "few-shot", "3": "chain-of-thought"}
    choice = input("Choice (1/2/3): ").strip()
    strategy = strategy_map.get(choice, "zero-shot")
    print(f"Using: {strategy}")

    # Pantry input method
    print("\nHow do you want to load your pantry?")
    print("  1. Load from CSV file")
    print("  2. Enter ingredients manually")
    pantry_choice = input("Choice (1/2): ").strip()

    pantry = []
    if pantry_choice == "1":
        filepath = input("CSV file path: ").strip()
        if os.path.exists(filepath):
            pantry = load_pantry_from_csv(filepath)
            print(f"Loaded {len(pantry)} ingredients.")
        else:
            print("File not found. Switching to manual entry.")
            pantry = enter_pantry_manually()
    else:
        pantry = enter_pantry_manually()

    if not pantry:
        print("No ingredients entered. Exiting.")
        return

    print("\nYour pantry:")
    show_pantry(pantry)

    chat(pantry, strategy)


if __name__ == "__main__":
    main()
