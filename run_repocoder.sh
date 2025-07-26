unzip test-apps.zip

mv test-apps RepoExec

rm RepoExec/*.csv

export PYTHONPATH=$PYTHONPATH:code-llm-evaluator/src

python run_pipeline.py --window_sizes 20 --slice_sizes 2 