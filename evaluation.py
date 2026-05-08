"""
Evaluation Script — Kitchen Assistant
Runs all three prompting strategies against the 20 hand-crafted pantry scenarios.
Computes: constraint adherence rate, substitution validity, consistency score.
"""

import ollama
import csv
import json
import re
from datetime import date
from collections import defaultdict

MODEL = "qwen2.5:3b"
SCENARIOS_FILE = "evaluation_scenarios.csv"
RUNS_PER_SCENARIO = 5  # for consistency scoring
CONSISTENCY_TEMPERATURE = 0.2
GENERATION_TEMPERATURE = 0.7

STRATEGIES = ["zero-shot", "few-shot", "chain-of-thought"]

# ── Prompt Builders ───────────────────────────────────────────────────────────

def build_eval_prompt(scenario: dict, strategy: str) -> tuple[str, str]:
    """Return (system_prompt, user_message) for a given scenario and strategy."""

    ingredients_raw = scenario["ingredients"]
    task_type = scenario["task_type"]

    # Parse ingredient names from the ingredients string
    ingredient_names = []
    for item in ingredients_raw.split("),"):
        name = item.strip().split("(")[0].strip().lower()
        if name:
            ingredient_names.append(name)

    pantry_str = ingredients_raw
    available = ", ".join(ingredient_names)

    base_system = f"""You are a kitchen assistant helping a student reduce food waste.
The user's pantry contains ONLY these ingredients:

{pantry_str}

STRICT RULES:
- Only suggest recipes or substitutions using ingredients listed above.
- Do NOT introduce any ingredient not in the pantry.
- Prioritise ingredients that expire soonest.
- Keep suggestions simple and student-friendly.
- Available ingredients: {available}"""

    few_shot_examples = """
EXAMPLES of good responses:

Example 1:
Pantry: eggs (3), butter (50g, expires soon), flour (200g)
User: What can I make?
Assistant: Make simple crepes — beat eggs, melt butter, mix with flour and water, fry thin. Uses up the butter before it expires.

Example 2:
Pantry: olive oil (100ml), lemon (1), garlic (2 cloves)
User: I need butter for my pasta sauce, what can I use instead?
Assistant: Use olive oil. It works well in a savory sauce — add lemon at the end to brighten it up.

Example 3:
Pantry: spinach (100g, expires tomorrow), eggs (2), olive oil (50ml)
User: What should I cook first?
Assistant: Use the spinach today — it expires tomorrow. Make a spinach omelette: wilt spinach in olive oil, add beaten eggs, fold and serve."""

    cot_instruction = """
Before answering, reason step by step:
1. Which ingredients expire soonest?
2. What role does each ingredient play (protein, fat, carb, flavouring)?
3. What combinations make sense with only what is available?
4. Give your final suggestion.

Show your reasoning, then your final answer."""

    if strategy == "zero-shot":
        system = base_system
    elif strategy == "few-shot":
        system = base_system + "\n" + few_shot_examples
    elif strategy == "chain-of-thought":
        system = base_system + "\n" + cot_instruction
    else:
        system = base_system

    # Build user message based on task type
    if task_type == "recipe":
        user_msg = "What can I cook with what I have? Please suggest a recipe."
    elif task_type == "substitution":
        disallowed = scenario.get("disallowed_ingredients", "an ingredient")
        user_msg = f"I don't have {disallowed}. What can I use instead from my pantry?"
    elif task_type == "expiry":
        user_msg = "What should I cook first given what is about to expire?"
    else:
        user_msg = "What can I make with these ingredients?"

    return system, user_msg


# ── Scoring ───────────────────────────────────────────────────────────────────

def check_constraint_adherence(response: str, disallowed: str) -> bool:
    """
    Returns True if the response does NOT mention any disallowed ingredient.
    """
    response_lower = response.lower()
    disallowed_items = [d.strip().lower() for d in disallowed.split(",")]
    for item in disallowed_items:
        if item and item in response_lower:
            return False
    return True


def check_substitution_validity(response: str, target: str) -> bool:
    """
    Returns True if any word from the target response appears in the model output.
    Simple keyword match against ground truth.
    """
    response_lower = response.lower()
    target_words = [w.strip().lower() for w in target.split() if len(w) > 3]
    matches = sum(1 for w in target_words if w in response_lower)
    return matches >= max(1, len(target_words) // 2)


def consistency_score(responses: list[str]) -> float:
    """
    Measures consistency across repeated runs.
    Uses a simple overlap heuristic: for each pair of responses,
    compute word overlap ratio, then average.
    """
    if len(responses) < 2:
        return 1.0
    scores = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            words_i = set(responses[i].lower().split())
            words_j = set(responses[j].lower().split())
            if not words_i or not words_j:
                continue
            overlap = len(words_i & words_j) / len(words_i | words_j)
            scores.append(overlap)
    return round(sum(scores) / len(scores), 3) if scores else 0.0


# ── Runner ────────────────────────────────────────────────────────────────────

def run_scenario(scenario: dict, strategy: str, n_runs: int = 1, temperature: float = GENERATION_TEMPERATURE) -> list[str]:
    """Run a single scenario n times and return list of responses."""
    system, user_msg = build_eval_prompt(scenario, strategy)
    responses = []
    for _ in range(n_runs):
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": temperature},
        )
        responses.append(response["message"]["content"])
    return responses


def evaluate(scenarios_file: str):
    """Main evaluation loop."""
    # Load scenarios
    scenarios = []
    with open(scenarios_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenarios.append(row)

    print(f"Loaded {len(scenarios)} scenarios.")
    results = []

    for strategy in STRATEGIES:
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'='*50}")

        adherence_scores = []
        validity_scores = []
        consistency_scores = []

        for scenario in scenarios:
            sid = scenario["scenario_id"]
            task = scenario["task_type"]
            disallowed = scenario.get("disallowed_ingredients", "")
            target = scenario.get("target_response", "")

            print(f"  Running {sid} ({task})...", end=" ", flush=True)

            # Single run for adherence + validity
            single_run = run_scenario(scenario, strategy, n_runs=1, temperature=GENERATION_TEMPERATURE)
            response = single_run[0]

            adherence = check_constraint_adherence(response, disallowed)
            adherence_scores.append(int(adherence))

            if task == "substitution":
                validity = check_substitution_validity(response, target)
                validity_scores.append(int(validity))

            # Multiple runs for consistency (low temperature)
            multi_run = run_scenario(scenario, strategy, n_runs=RUNS_PER_SCENARIO, temperature=CONSISTENCY_TEMPERATURE)
            cons = consistency_score(multi_run)
            consistency_scores.append(cons)

            print(f"adherence={'✓' if adherence else '✗'}, consistency={cons:.2f}")

            results.append({
                "scenario_id": sid,
                "task_type": task,
                "strategy": strategy,
                "constraint_adherence": int(adherence),
                "substitution_validity": int(validity_scores[-1]) if task == "substitution" else "N/A",
                "consistency_score": cons,
                "response_preview": response[:200].replace("\n", " "),
            })

        # Summary per strategy
        print(f"\n  Results for {strategy}:")
        print(f"    Constraint adherence rate : {sum(adherence_scores)/len(adherence_scores):.2%}")
        if validity_scores:
            print(f"    Substitution validity rate: {sum(validity_scores)/len(validity_scores):.2%}")
        print(f"    Mean consistency score    : {sum(consistency_scores)/len(consistency_scores):.3f}")

    # Save full results
    output_file = "evaluation_results.csv"
    fieldnames = ["scenario_id", "task_type", "strategy", "constraint_adherence",
                  "substitution_validity", "consistency_score", "response_preview"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    evaluate(SCENARIOS_FILE)
