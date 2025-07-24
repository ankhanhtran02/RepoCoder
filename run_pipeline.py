# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from turtle import update
import re
import argparse

from sympy import ground_roots
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from make_window import MakeWindowWrapper
from build_vector import BuildVectorWrapper, BagOfWords
from search_code import CodeSearchWrapper
from build_prompt import BuildPromptWrapper
from datasets import load_dataset
import json
from utils import CONSTANTS, CodexTokenizer, FilePathBuilder
from collections import defaultdict
from generate import generate

patterns = [r'^.*?"""\n', r"^.*?'\n", r"^.*?'''\n", r'^.*?"\n']

def get_lineno(entry_point, ground_truth, fpath_tuple):
    fpath = "test-apps/test-apps/" + "/".join(fpath_tuple)
    ground_truth_lines = ground_truth.split('\n')
    function_start = "def " + entry_point
    context_start_lineno = -1
    line_no = -1
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if function_start in line.strip():
                context_start_lineno = i
            if context_start_lineno != -1 and ground_truth_lines[0].strip() == line.strip():
                line_no = i
                break
    return context_start_lineno, line_no, ground_truth_lines, lines

def process_solution(docstring, raw_solution, id):
    ground_truth = ""
    docstring_elements = docstring.replace('"""', '').replace("'''", '').split('\n')
    docstring_lines = []
    for element in docstring_elements:
        if element.strip() != "":
            docstring_lines.append(element.strip())
    assert len(docstring_lines) > 0, "Docstring must not be empty"
    last_docstring_text = docstring_lines[-1]
    index = raw_solution.find(last_docstring_text)
    if index != -1:
        raw_solution = raw_solution[index + len(last_docstring_text):]
    else:
        print(f"Error: marker '{last_docstring_text}' not found in {raw_solution}")
    pattern_found = False
    if id == 254:
        print(raw_solution)
        print(last_docstring_text)
    for pattern in patterns:
        if re.search(pattern, raw_solution, flags=re.DOTALL):
            ground_truth = re.sub(pattern, '', raw_solution, count=1, flags=re.DOTALL)
            pattern_found = True
            break
    return ground_truth, pattern_found

def find_ground_truth(sample, fpath_tuple):
    error = False
    error_message = {}
    task_id = sample["id"]
    ground_truth, pattern_found = process_solution(sample["original_docstring"], sample["raw_solution"], task_id)
    if not pattern_found:
        print(f"Warning: Prompt not found in raw solution for task {task_id}.")
        error = True
        error_message["prompt_not_found"] = True
    context_start_lineno, line_no, ground_truth_lines, repo_lines = get_lineno(sample["entry_point"], ground_truth, fpath_tuple)
    if context_start_lineno == -1:
        print(f"Warning: Context start line not found for task {task_id}.")
        error = True
        error_message["context_start_lineno_not_found"] = True
    if line_no == -1:
        print(f"Warning: Line number for ground truth not found for task {task_id}.")
        error = True
        error_message["line_no_not_found"] = True
    if error:
        error_message["task_id"] = sample["id"]
        error_message["fpath_tuple"] = fpath_tuple
        error_message["raw_solution"] = sample["raw_solution"]
        error_message["target_function_prompt"] = sample["target_function_prompt"]
        error_message["docstring"] = sample["original_docstring"]
        error_message["ground_truth"] = ground_truth
        error_message["ground_truth_lines"] = ground_truth_lines
        error_message["repo_lines"] = repo_lines
        return None, context_start_lineno, line_no, error_message
    return ground_truth, context_start_lineno, line_no, None

def save_ds_to_jsonl(ds, output_path, error_fpath='errors.jsonl'):
    updated_ds = []
    id_dict = defaultdict(int)
    errors = []
    for sample in ds:
        prompt = sample["target_function_prompt"]
        repo = sample["project"].split("/")[1]
        task_id = repo + "/" + str(id_dict[repo])
        id_dict[repo] += 1
        fname = sample["module"].split(".")[-1] + ".py"
        fpath_tuple = sample["project"].split("/")[1:] + sample["module"].split(".")[:-1] + [fname]
        ground_truth, context_start_lineno, line_no, error_message = find_ground_truth(sample, fpath_tuple)
        if error_message:
            errors.append(error_message)
        new_sample = {
            "prompt": prompt,
            "metadata": {
                "task_id": task_id,
                "ground_truth": ground_truth,
                "fpath_tuple": fpath_tuple,
                "context_start_lineno": context_start_lineno,
                "line_no": line_no,
            }
        }
        updated_ds.append(new_sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in updated_ds:
            f.write(json.dumps(sample) + "\n")
    print(f"Dataset saved to {output_path}")
    with open(error_fpath, 'w', encoding='utf-8') as f:
        for sample in errors:
            f.write(json.dumps(sample) + "\n")
    print(f"Errors saved to {error_fpath}")

def get_repos(base_dir):
    if base_dir == "RepoExec":
        return os.listdir(base_dir)
    return []

def make_repo_window(benchmark, repos, window_sizes, slice_sizes, repo_base_dir=FilePathBuilder.repo_base_dir):
    worker = MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes, repo_base_dir)
    worker.window_for_repo_files()

def run_RG1_and_oracle_method(repos, benchmark_path, window_sizes, slice_sizes, repo_base_dir, model, max_tokens, batch_size, cache_dir, save_dir, num_return_sequences, temperature, repetition_penalty, do_sample, top_p, top_k):
    # build code snippets for all the repositories
    make_repo_window(None, repos, window_sizes, slice_sizes, repo_base_dir)
    # # build code snippets for vanilla retrieval-augmented approach and ground truth
    MakeWindowWrapper(benchmark_path, repos, window_sizes, slice_sizes, repo_base_dir).window_for_baseline_and_ground()
    # # build vector for vanilla retrieval-augmented approach and ground truth
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark_path, vectorizer, repos, window_sizes, slice_sizes, repo_base_dir).vectorize_baseline_and_ground_windows()
    BuildVectorWrapper(benchmark_path, vectorizer, repos, window_sizes, slice_sizes, repo_base_dir).vectorize_repo_windows()
    # search code for vanilla retrieval-augmented approach and ground truth
    CodeSearchWrapper('one-gram', benchmark_path, repos, window_sizes, slice_sizes, repo_base_dir).search_baseline_and_ground()
    # build prompt for vanilla retrieval-augmented approach and ground truth
    tokenizer = CodexTokenizer
    # prediction_paths []
    for w in window_sizes:
        for s in slice_sizes:
            mode = CONSTANTS.rg
            output_file_path = f'prompts/rg-one-gram-ws-{w}-ss-{s}.jsonl'
            save_fn = f'rg-one-gram-ws-{w}-ss-{s}_samples.0.jsonl'
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            BuildPromptWrapper('one-gram', benchmark_path, repos, w, s, tokenizer).build_first_search_prompt(mode, output_file_path)
            generate(
                data=output_file_path,
                model=model,
                split="test",
                task_name="RepoExec-rg",
                max_tokens=max_tokens,
                batch_size=batch_size,
                cache_dir=cache_dir,
                save_dir=save_dir,
                save_fn=save_fn,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k
                )
            # prediction_paths.append(os.path.join(save_dir, save_fn))

            # mode = CONSTANTS.gt
            # output_file_path = f'prompts/gt-one-gram-ws-{w}-ss-{s}.jsonl'
            # save_fn = f'gt-one-gram-ws-{w}-ss-{s}_samples.0.jsonl'
            # BuildPromptWrapper('one-gram', benchmark_path, repos, w, s, tokenizer).build_first_search_prompt(mode, output_file_path)
            # generate(
            #     data=output_file_path,
            #     model=model,
            #     split="test",
            #     task_name="RepoExec-gt",
            #     max_tokens=max_tokens,
            #     batch_size=batch_size,
            #     cache_dir=cache_dir,
            #     save_dir=save_dir,
            #     save_fn=save_fn,
            #     num_return_sequences=num_return_sequences,
            #     temperature=temperature,
            #     repetition_penalty=repetition_penalty,
            #     do_sample=do_sample,
            #     top_p=top_p,
            #     top_k=top_k
            #     )

def run_RepoCoder_method(iter, benchmark, repos, window_sizes, slice_sizes, prediction_path, repo_base_dir, model, max_tokens, batch_size, cache_dir, save_dir, num_return_sequences, temperature, repetition_penalty, do_sample, top_p, top_k):
    mode = CONSTANTS.rgrg
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    MakeWindowWrapper(benchmark, repos, window_sizes, slice_sizes, repo_base_dir).window_for_prediction(mode, prediction_path)
    vectorizer = BagOfWords
    BuildVectorWrapper(benchmark, vectorizer, repos, window_sizes, slice_sizes).vectorize_prediction_windows(mode, prediction_path)
    CodeSearchWrapper('one-gram', benchmark, repos, window_sizes, slice_sizes, repo_base_dir).search_prediction(mode, prediction_path)
    tokenizer = CodexTokenizer
    for w in window_sizes:
        for s in slice_sizes:
            output_file_path = f'prompts/repocoder-one-gram-ws-{w}-ss-{s}.{iter}.jsonl'
            save_fn = f'repocoder-one-gram-ws-{w}-ss-{s}_samples.{iter}.jsonl'
            BuildPromptWrapper('one-gram', benchmark, repos, w, s, tokenizer).build_prediction_prompt(mode, prediction_path, output_file_path)
            generate(
                data=output_file_path,
                model=model,
                split="test",
                task_name="RepoExec-gt",
                max_tokens=max_tokens,
                batch_size=batch_size,
                cache_dir=cache_dir,
                save_dir=save_dir,
                save_fn=save_fn,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k
                )

def run_repocoder(num_iter, repo_base_dir, benchmark_path, window_sizes, slice_sizes, model, max_tokens, batch_size, cache_dir, save_dir, num_return_sequences, temperature, repetition_penalty, do_sample, top_p, top_k):
    repos = get_repos(repo_base_dir)
    if not repos:
        print(f"No repositories found in {repo_base_dir}. Please check the directory.")
        return
    # run_RG1_and_oracle_method(repos, benchmark_path, window_sizes, slice_sizes, repo_base_dir, model, max_tokens, batch_size, cache_dir, save_dir, num_return_sequences, temperature, repetition_penalty, do_sample, top_p, top_k)
    prediction_path = os.path.join(save_dir, "rg-one-gram-ws-20-ss-2_samples.0.jsonl")
    for i in range(1, num_iter + 1):
        run_RepoCoder_method(i, benchmark_path, repos, window_sizes, slice_sizes, prediction_path, repo_base_dir, model, max_tokens, batch_size, cache_dir, save_dir, num_return_sequences, temperature, repetition_penalty, do_sample, top_p, top_k)
        prediction_path = f"{save_dir}/repocoder-one-gram-ws-{window_size}-ss-{slice_size}_samples.{i}.jsonl"



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add two numbers.")

    parser.add_argument("--num_iter", type=int, default=1, help="Number of RepoCoder runs")
    parser.add_argument("--repo_base_dir", type=str, default="RepoExec", help="Path of the directory containing repositories code.")
    parser.add_argument("--benchmark_path", type=str, default="dataset/RepoExec_benchmark.jsonl", help="Path of the benchmark JSONL file")
    parser.add_argument("--window_sizes", type=int, nargs="+",required=True, help="List of integers")
    parser.add_argument("--slice_sizes", type=int, nargs="+", required=True, help="List of integers")

    # Prediction generation args
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cache_dir", type=str, default="/cache")
    parser.add_argument("--save_dir", type=str, default="predictions")
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument('--do_sample', action="store_true")
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)
    
    args = parser.parse_args()
    run_repocoder(**vars(args))
