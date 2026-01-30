import json
import time
import openai
import os
import random
from typing import Dict, Iterable, List, Tuple, Any
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np


def write_jsonl(filename: str, data: Iterable[Dict[str, Any]]) -> None:
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


class DifficultyComparator:
    def __init__(self, api_base: str, api_key: str, model: str = "gpt-4o"):
        openai.api_base = api_base
        openai.api_key = api_key
        self.model = model
        self.comparison_cache: Dict[Tuple[str, str], int] = {}

    def compare_two(
            self,
            q1_text: str,
            q2_text: str,
            n: int = 1,
            temp: float = 0.1,
            max_retries: int = 5,
            timeout: int = 120
    ) -> Tuple[int, int, int]:
        cache_key = tuple(sorted((q1_text, q2_text)))
        if cache_key in self.comparison_cache:
            cached_res = self.comparison_cache[cache_key]
            if q1_text == cache_key[0]:
                return cached_res, 0, 0
            else:
                return (3 - cached_res) if cached_res in [1, 2] else 0, 0, 0
        instruction = (
            "You are a judge for question difficulty.\n"
            "Decide which question is MORE DIFFICULT.\n"
            "Respond with ONLY one token: Q1 or Q2.\n"
        )
        query = f"Q1:\n{q1_text}\n\nQ2:\n{q2_text}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": query}
        ]
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    n=n,
                    temperature=temp,
                    request_timeout=timeout
                )
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                votes = []
                for choice in response["choices"]:
                    content = choice["message"]["content"].strip().upper()
                    if "Q1" in content:
                        votes.append(1)
                    elif "Q2" in content:
                        votes.append(2)
                    else:
                        votes.append(0)
                if votes:
                    res = Counter(votes).most_common(1)[0][0]
                    if q1_text == cache_key[0]:
                        self.comparison_cache[cache_key] = res
                    else:
                        self.comparison_cache[cache_key] = (3 - res) if res in [1, 2] else 0
                    return res, prompt_tokens, completion_tokens
                return 0, prompt_tokens, completion_tokens
            except Exception as e:
                print(f"[Compare Error] Attempt {attempt + 1}/{max_retries}: {e}")
                time.sleep(10)
        return 0, 0, 0


def llm_quicksort(
        items: List[Tuple[int, str]],
        comparator: DifficultyComparator,
        compare_n: int = 1,
        max_workers: int = 8
) -> Tuple[List[Tuple[int, str]], int, int]:
    if len(items) <= 1:
        return items, 0, 0
    total_prompt_tok, total_completion_tok = 0, 0
    pivot = random.choice(items)
    pivot_idx, pivot_text = pivot
    easier: List[Tuple[int, str]] = []
    harder: List[Tuple[int, str]] = []
    others = [item for item in items if item[0] != pivot_idx]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(comparator.compare_two, item[1], pivot_text, n=compare_n): item
            for item in others
        }
        for future in tqdm(as_completed(future_to_item), total=len(others), desc="QuickSort Compare"):
            res, p_tok, c_tok = future.result()
            total_prompt_tok += p_tok
            total_completion_tok += c_tok
            item = future_to_item[future]
            if res == 1:
                harder.append(item)
            else:
                easier.append(item)
    s_harder, p1, c1 = llm_quicksort(harder, comparator, compare_n, max_workers)
    s_easier, p2, c2 = llm_quicksort(easier, comparator, compare_n, max_workers)
    total_prompt_tok += p1 + p2
    total_completion_tok += c1 + c2
    return s_easier + [pivot] + s_harder, total_prompt_tok, total_completion_tok


def dsc_r_quicksort_ranking(
        questions: List[str],
        api_base: str,
        api_key: str,
        model: str = "gpt-4o",
        compare_n: int = 1,
        seed: int = 42
) -> Tuple[List[int], Dict[str, int]]:
    random.seed(seed)
    indexed_questions = list(enumerate(questions))
    comparator = DifficultyComparator(api_base, api_key, model)
    sorted_items, p_tok, c_tok = llm_quicksort(
        indexed_questions,
        comparator,
        compare_n=compare_n
    )
    sorted_indices = [idx for idx, _ in sorted_items]
    ranking_cost = {
        "input_tokens": p_tok,
        "output_tokens": c_tok
    }
    return sorted_indices, ranking_cost


def extract_answer_from_raw(item, dataset):
    if dataset == "GSM8K":
        return item['answer'].split('####')[-1].strip()
    else:
        return item['output']

if __name__ == "__main__":
    API_BASE = "--------------"
    API_KEY = "--------------"
    MODEL = "gpt-4o"
    DATASET = "GSM8K"

    input_map = {
        "GSM8K": "dataset/GSM8K.jsonl",
        "coin_flip": "dataset/coin_flip.jsonl",
        "last_letter": "dataset/last_letter.jsonl",
        "strategy": "dataset/strategy.jsonl",
        "common": "dataset/common.jsonl",
        "MATH": "dataset/MATH_all.jsonl"
    }
    input_path = input_map.get(DATASET)
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset file for {DATASET} not found at {input_path}")

    questions = []
    answers = []
    subjects = []
    hard_levels = []

    problem_field_name = "input" if DATASET != "MATH" else "problem"

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            questions.append(item[problem_field_name])
            answers.append(extract_answer_from_raw(item, DATASET))
            if DATASET == "MATH":
                subjects.append(item.get("subject", "unknown"))
                hard_levels.append(item.get("hard_level", 0))

    print(f"Loaded {len(questions)} questions for dataset: {DATASET}")

    sorted_idx, cost = dsc_r_quicksort_ranking(
        questions=questions,
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL,
        compare_n=3
    )

    print("\n--- Ranking Complete ---")
    print("Total ranking cost:", cost)

    eval_list = [0.0] * len(questions)
    num_questions = len(questions)
    for rank, idx in enumerate(sorted_idx):
        eval_list[idx] = rank / max(1, num_questions - 1)

    output_data = []

    if DATASET != "MATH":
        output_data.append({
            "questions": questions,
            "answer": answers,
            "eval": eval_list,
            "completion": [[] for _ in questions]
        })
    else:

        from collections import defaultdict
        subj_dict = defaultdict(lambda: {"problem": [], "answer": [], "eval": [], "hard_level": []})
        for i in range(len(questions)):
            subj = subjects[i]
            subj_dict[subj]["problem"].append(questions[i])
            subj_dict[subj]["answer"].append(answers[i])
            subj_dict[subj]["eval"].append(eval_list[i])
            subj_dict[subj]["hard_level"].append(hard_levels[i])

        for subj, info in subj_dict.items():
            output_data.append({
                "type": subj,
                "questions": info["problem"],
                "answer": info["answer"],
                "eval": info["eval"],
                "hard_level": info["hard_level"]
            })


    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/{DATASET}_{MODEL.replace('/', '_')}_converted_eval.jsonl"
    cost_filename = f"{output_dir}/{DATASET}_{MODEL.replace('/', '_')}_quicksort_cost.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved converted output to: {output_filename}")

    with open(cost_filename, "w", encoding="utf-8") as f:
        json.dump(cost, f, indent=4)
    print(f"Saved ranking cost to: {cost_filename}")

    print("\nProcess complete.")
