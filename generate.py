
from typing import Any
from string import ascii_uppercase
import argparse
import os
from code_eval import Evaluator
from code_eval.tasks.base import TaskBase
from transformers import AutoTokenizer


system_prompt = """You are a helpful coding assistant."""



class RepoCoderTask(TaskBase):
    def __init__(self, task_name, dataset_path, model_name, backend="vllm", split="test", system_prompt= None) -> None:
        self.TASK_NAME = task_name
        self.DATASET_NAME_OR_PATH = dataset_path
        self.split = split
        self.backend = backend
        self.system_prompt = system_prompt

        if self.backend == "vllm":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        super().__init__()
   
    def prepare_dataset(self, *args: Any, **kwargs: Any) -> Any:

        dataset = self.dataset[self.split]
        column_names = dataset.column_names

        def _preprocess(example):
   
            messages = []

            if self.system_prompt:
                messages = [
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                ]
           
            prompt = example["prompt"]
            messages.append({"role": "user", "content": prompt})
            if self.backend == "vllm":
                example['question'] = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            elif self.backend == "gpt":
                example["question"] = messages
           
            return example
       
        updated_dataset = dataset.map(_preprocess)
        return updated_dataset

def generate(data, model, task_name, split, max_tokens, batch_size, cache_dir, save_dir, save_fn, num_return_sequences, temperature, repetition_penalty, top_k, top_p, backend='vllm', base_url=None, add_latency=False, resume_last_generation=False):
    task = RepoCoderTask(task_name=task_name, dataset_path=data, split = split, system_prompt= system_prompt, model_name = model, backend=backend)
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    save_dir = save_dir
    evaluator = Evaluator(task=task,
                        model_name=model,
                        batch_size=batch_size,
                        save_dir=save_dir,
                        save_fn=save_fn,
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                        base_url=base_url,
                        add_latency=add_latency,
                        resume_last_generation=resume_last_generation
                        )
   
    print("="*25 + "Test sample" + "="*25)
    print(evaluator.dataset['question'][0])
    print(len(evaluator.dataset['question']))
    print("="*61 )

    prediction_fn = evaluator.generate(
                    backend=backend,
                    max_tokens=max_tokens,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    top_k=top_k)
    return prediction_fn

if __name__ == "__main__":
    generate(
        data="prompts/gt-one-gram-ws-20-ss-2.jsonl",
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        split="test",
        task_name="RepoExec-gt",
        max_tokens=2048,
        batch_size=8,
        cache_dir="/cache",
        save_dir="predictions",
        save_fn="gt.jsonl",
        num_return_sequences=5,
        temperature=0.2,
        repetition_penalty=1.2,
        do_sample=True,
        top_p=0.95,
        top_k=-1
        )
    
   

