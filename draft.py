import json 
from utils import Tools
import os
from collections import defaultdict
import re

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

def create_benchmark_file(input_path, output_path):
    with open(input_path, 'r') as inp_file, open(output_path, 'w', encoding='utf-8') as out_file:
        for line in inp_file:
            data = json.loads(line)
            fpath_tuple = data["completion_path"].split('/')
            sample = {
                "id": data["namespace"],
                "fpath_tuple": fpath_tuple,
                "solution_position": [data["signature_position"][0], data["body_position"][1]],
                "repo": "/".join(fpath_tuple[:2]),
                "target_function_prompt": data["target_function_prompt"],
                "import_file": data["import_file"],
            }
            out_file.write(json.dumps(sample) + "\n")

if __name__ == "__main__":
    data = Tools.load_jsonl("dataset/RepoExec_benchmark.jsonl")
    updated_samples = []
    for sample in data:
        metadata = sample['metadata']
        metadata['id'] = sample['id']
        updated_sample = {
            "prompt": sample['prompt'],
            "metadata": metadata
        }
        updated_samples.append(updated_sample)
    with open("dataset/RepoExec_benchmark.jsonl", "w", encoding="utf-8") as f:
        for s in updated_samples:
            f.write(json.dumps(s) + "\n")