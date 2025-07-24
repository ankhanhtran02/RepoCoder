from rank_bm25 import BM25Okapi
import json
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

sep = "/"

def create_prompt_with_bm25(windows_path, current_fpath, query, input_fpath_tuple, import_file_tuples, output_path=None):
    input_module = sep.join(input_fpath_tuple)
    updated_samples = []
    anchor_text = query
    with open(windows_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            metadata = data.get("metadata")
            if len(metadata) == 1:
                if metadata[0]["fpath_tuple"] != input_fpath_tuple and metadata[0]["fpath_tuple"] in import_file_tuples: # if the chunk is not from the same file but is from an import file, keep it
                    updated_samples.append(data)
            elif len(metadata) > 1:
                new_metadata = [
                    meta for meta in metadata
                    if meta["fpath_tuple"] != input_fpath_tuple and meta["fpath_tuple"] in import_file_tuples
                ]
                if new_metadata:
                    data["metadata"] = new_metadata
                    updated_samples.append(data)
    with open(current_fpath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["metadata"] = [{"fpath_tuple": input_fpath_tuple}]
            updated_samples.append(data) # Add the current file's context
    # with open("windows.jsonl", 'w', encoding='utf-8') as f:
    #     for sample in updated_samples:
    #         f.write(json.dumps(sample) + "\n")
    # return
            
    if updated_samples and anchor_text:
        tokenized_corpus = [context["context"].split() for context in updated_samples]
        tokenized_query = anchor_text.split()
        
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        for i in range(len(updated_samples)):
            updated_samples[i]["bm25_score"] = scores[i]

    top_10 = sorted(updated_samples, key=lambda x: x["bm25_score"], reverse=True)[:10]
    modules_dict = defaultdict(list)
    for sample in top_10:
        for metadata in sample["metadata"]:
            module_name = "/".join(metadata["fpath_tuple"])
            modules_dict[module_name].append(sample)
    
    prompt_elements = ["You are a Python programmer working with a repository. Here is all the context you may find useful to complete the function:", ]
    same_modules = []
    count = 0
    for module_name, samples in modules_dict.items():
        if module_name != input_module:
            prompt_elements.append(f"#FILE: {module_name}")
            for i, sample in enumerate(samples):
                count += 1
                prompt_elements.append(f"##CHUNK {i+1}")
                prompt_elements.append(sample['context'])
                prompt_elements.append("")
        else:
            same_modules.extend(samples)
    prompt_elements.append(f"#CURRENT FILE: {input_module}")
    for i, sample in enumerate(same_modules):
        prompt_elements.append(f"##CHUNK {i+1}")
        prompt_elements.append(sample['context'])
        prompt_elements.append("")    
    prompt_elements.append(f"Based on the information above, please complete the function in the current file {input_module}:")
    prompt_elements.append(anchor_text)
    prompt = "\n".join(prompt_elements)
    
    if count + len(same_modules) > 10:
        print(f"Different modules: {count}, Same module: {len(same_modules)}")

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
    return prompt

def create_prompt(example, input_dir = "cache/RepoExec/window"):
    if input_dir == "cache/RepoExec/window":
        project_tuple = example["project"].split(sep)
        repo = project_tuple[1]
        dir_list = []
        if len(project_tuple) > 2:
            dir_list = project_tuple[2:]
        id = example['id']
        query = example["target_function_prompt"]
        module = example["module"].split(".")
        fname = module.pop(-1) + ".py"
        module.append(fname)
        input_fpath_tuple: list[str] = dir_list + module
        import_file = example["import_file"]
        import_file_tuples =  [file.split('/')[1:] for file in import_file]
    else:
        input_fpath_tuple: list[str] = example["fpath_tuple"]
        repo = example["repo"]
        id = example["id"]
        query = example["target_function_prompt"]
        import_file = example["import_file"]
        import_file_tuples = [file.split('/') for file in import_file]
    windows_path = f"{input_dir}/repos/{repo}_ws20_ss2.jsonl"
    current_fpath = f"{input_dir}/current-files/{id}_ws20_ss2.jsonl"
    example["prompt"] = create_prompt_with_bm25(windows_path, current_fpath, query, input_fpath_tuple,import_file_tuples, None)
    return example

def run_bm25(samples, output_path, input_dir="cache/RepoExec/window"):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(samples):
            data = create_prompt(sample, input_dir)
            f.write(json.dumps(data) + "\n")
    print(f"BM25 prompts saved to {output_path}")

if __name__ == "__main__":
    with open('dataset/deveval_add_import_file_benchmark.jsonl', 'r') as file:
        samples = [json.loads(line) for line in file]
    output_path = "cache/deveval_add_import_file_bm25.jsonl"
    run_bm25(samples, output_path, input_dir="cache/Source_Code/window")
