wget https://huggingface.co/datasets/LJ0815/DevEval/resolve/main/Source_Code.tar.gz

tar -xvzf Source_Code.tar.gz

mv Source_Code DevEval

export PYTHONPATH=$PYTHONPATH:code-llm-evaluator/src

python run_pipeline.py --window_sizes 20 --slice_sizes 2 --num_iter 1 --repo_base_dir DevEval --benchmark_path dataset/DevEval_benchmark.jsonl --save_dir predictions/DevEval --final_fn deveval.final.generated.jsonl