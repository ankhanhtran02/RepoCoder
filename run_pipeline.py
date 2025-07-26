# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from make_window import MakeWindowWrapper
from build_vector import BuildVectorWrapper, BagOfWords
from search_code import CodeSearchWrapper
from build_prompt import BuildPromptWrapper
from utils import CONSTANTS, CodexTokenizer, FilePathBuilder
from generate import generate



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
