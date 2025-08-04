curl -O https://huggingface.co/datasets/LJ0815/DevEval/blob/main/Source_Code.tar.gz

tar -xvzf Source_Code.tar.gz

mv Source_Code DevEval

export PYTHONPATH=$PYTHONPATH:code-llm-evaluator/src
