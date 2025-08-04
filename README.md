# RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation

# Overview

In the paper, we present **RepoCoder**, a simple, generic, and effective framework to tackle the repository-level code completion task, which is to continue writing the unfinished code based on a broader context of the repository. RepoCoder incorporates a similarity-based retriever, a pre-trained code language model, and a novel iterative retrieval-generation paradigm. It streamlines the overall process and eliminates the need for heuristic rules, static code analysis, data labeling, and model re-training in previous studies. 

![framework](./figs/framework.png)
<center>
Figure 1. The illustration of our RepoCoder framework.
</center>

We also present a new benchmark, **RepoEval**, for the repository-level code completion task, which consists of the latest and high-quality real-world repositories covering line, API invocation, and function body completion scenarios. We test the performance of RepoCoder and show that it significantly improves the zero-shot code completion baseline by over 10% and consistently outperforms the vanilla retrieval-augmented code completion approach.

## Project

This project contains the basic components of RepoCoder. Here is an overview:

```shell
|-- make_window.py # slice the repository files and the model predictions into code snippets
|-- build_vector.py # build the vector representation for the code snippets
|-- search_code.py # search relevant code snippets with the vector representation
|-- build_prompt.py # build the prompt with the unfinished code and the retrieved code snippets
|-- run_pipeline.py # run the code completion pipeline
|-- compute_score.py # evaluate the performance of the code completion
|-- utils.py # utility functions
|-- datasets/datasets.zip # the input data for the code completion task
    |-- function_level_completion_4k_context_codex.test.jsonl
    |-- function_level_completion_2k_context_codex.test.jsonl
    |-- line_level_completion_4k_context_codex.test.jsonl
    |-- line_level_completion_2k_context_codex.test.jsonl
    |-- line_level_completion_2k_context_codegen.test.jsonl
    |-- line_level_completion_1k_context_codegen.test.jsonl
    |-- api_level_completion_4k_context_codex.test.jsonl
    |-- api_level_completion_2k_context_codex.test.jsonl
    |-- api_level_completion_2k_context_codegen.test.jsonl
    |-- api_level_completion_1k_context_codegen.test.jsonl
|-- repositories # the checkpoints of repositories used to build our benchmark
    |-- function_level.zip 
      |-- CarperAI_trlx
      |-- lucidrains_imagen-pytorch
      |-- deepmind_tracr
      |-- leopard-ai_betty
      |-- google_lightweight_mmm
      |-- amazon-science_patchcore-inspection
      |-- facebookresearch_omnivore
      |-- maxhumber_redframes
    |-- line_and_api_level.zip
      |-- pytorch_rl
      |-- opendilab_ACE
      |-- google_vizier
      |-- awslabs_fortuna
      |-- huggingface_evaluate
      |-- huggingface_diffusers
      |-- nerfstudio-project_nerfstudio
      |-- alibaba_FederatedScope
```

We utilize a private library to handle the execution and evaluation of the function-level completion. Due to the license issue, we cannot release the code. However, we provide the data for the function-level completion task in `datasets/datasets.zip` and `repositories/function_level.zip`.

# Quickstart

## Prepare Environment
First, we should set up a python environment in `RepoCoder` directory. This code base has been tested under python 3.9.

```bash
$ conda create -n repocoder python=3.9
$ conda activate repocoder
$ pip install -r requirements.txt
$ cd code-llm-evaluator
$ pip install -e .
$ pip install vllm
$ cd ..
```

## Run the Code Completion
To output predictions for **RepoExec** dataset, run commands in `run_repoexec.sh` on your terminal in order. 

To output predictions for **DevEval** dataset, run commands in `run_deveval.sh` on your terminal in order. 

View `run_pipeline.py` module for more information on arguments.

# Citation

If our work is useful, please consider citing our paper:

```bibtex
@article{zhang2023repocoder,
  title={RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation},
  author={Zhang, Fengji and Chen, Bei and Zhang, Yue and Liu, Jin and Zan, Daoguang and Mao, Yi and Lou, Jian-Guang and Chen, Weizhu},
  journal={arXiv preprint arXiv:2303.12570},
  year={2023}
}
```

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# License

Please note that this repo is under [MIT License](LICENSE).

# Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
