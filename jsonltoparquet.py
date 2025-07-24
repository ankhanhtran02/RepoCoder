import pandas as pd


df1 = pd.read_json("cache/deveval_add_import_file_bm25.jsonl", lines=True)


df1.to_parquet("cache/deveval_add_import_file_bm25.parquet", engine="pyarrow")  

df2 = pd.read_json("cache/repoexec_add_import_file_bm25.jsonl", lines=True)


df2.to_parquet("cache/repoexec_add_import_file_bm25.parquet", engine="pyarrow")  



