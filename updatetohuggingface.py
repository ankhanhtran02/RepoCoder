import re
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
import os

os.environ["HF_TOKEN"] = "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"

deveval = Dataset.from_parquet("cache/deveval_add_import_file_bm25.parquet")
repo_exec = Dataset.from_parquet("cache/repoexec_add_import_file_bm25.parquet")


deveval.push_to_hub("ankhanhtran02/deveval_add_import_file_bm25")

repo_exec.push_to_hub("ankhanhtran02/repoexec_add_import_file_bm25")