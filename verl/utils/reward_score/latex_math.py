import re
from typing import Dict
import random

from math_verify import parse, verify

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def extract_solution(solution_str):
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            return remove_boxed(string_in_last_boxed)
        else:
            return None
    except Exception as e:
        print(e)
        print("@@@"*20)
        print(solution_str)
        print("+++"*20)
        print("Failed to extract answer from solution string.")
        print("@@@"*20)
        return None



def compute_score(prompt_str, solution_str, ground_truth, format_score=0.1, full_score=1.) -> float:
    try:

        do_print = random.randint(1, 4096) == 1
        if solution_str is None:
            if do_print: print(">> Solution string is None")
            return 0.0

        ans_string = extract_solution(solution_str)
        if do_print:
            print("\n" + "="*80)
            print(" Processing New Sample ".center(80, '='))
            print(f">> Prompt: {prompt_str}")
            print("-"*80)
            print(f">> Output: {solution_str}")
            print(f">> Extracted Answer: {ans_string}")
            print(f">> Ground Truth: {ground_truth}")

        if ans_string is None:
            if do_print: print(f">> Failed to extract answer from solution string.")
            if do_print: print(f">> Score: {0.0}")
            return 0.0
        if not ans_string or len(ans_string.strip()) == 0:
            if do_print: print(f">> Extracted answer is empty")
            return 0.0

        if math_is_equivalent(ans_string, ground_truth):
            if do_print: print(f">> Answer validation: FULL MATCH")
            if do_print: print(f">> Score: {full_score}")
            return full_score
        else:
            if do_print: print(f">> Answer validation: MISMATCH")
            if do_print: print(f">> Score: {format_score}")
            return format_score
    except Exception as e:
        print(e)
        print("@@@"*20)
        print(solution_str)
        print("+++"*20)
        print("Failed to compute score.")
        print("@@@"*20)
        return 0.0

def wrap_math_in_latex(math_str):
    if type(math_str) is not str:
        math_str = str(math_str)
    math_str.strip()
    if not math_str.startswith("$"):
        math_str = f"${math_str}"
    if not math_str.endswith("$"):
        math_str = f"{math_str}$"
    return math_str

def math_is_equivalent(gold_answer, pred_answer):
    gold_answer = wrap_math_in_latex(gold_answer)
    pred_answer = wrap_math_in_latex(pred_answer)
    gold_answer = parse(gold_answer)
    pred_answer = parse(pred_answer)
    return verify(gold_answer, pred_answer)
