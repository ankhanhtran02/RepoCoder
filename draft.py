import json 
from datasets import load_dataset

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
    ds = load_dataset("prompts/gt-one-gram-ws-20-ss-2.jsonl")
    print(ds)