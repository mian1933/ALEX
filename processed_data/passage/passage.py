import pandas as pd
import json

HOUSING_PARQUET_PATH = "[PATH_TO_HOUSING_STATUTES_PARQUET]"
BAR_PARQUET_PATH = "[PATH_TO_BAR_EXAM_PASSAGES_PARQUET]"
OUTPUT_JSONL_PATH = "[PATH_TO_COMBINED_OUTPUT_JSONL]"

with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f_out:
    df_bar = pd.read_parquet(BAR_PARQUET_PATH, engine='pyarrow')
    for i, row in df_bar.iterrows():
        row_dict = row.to_dict()
        text_content = row_dict.pop('text', '')
        source = row_dict.pop('source', "")
        metadata = {}
        if source is not None:
            metadata['source'] = source
        record_id = f'bar_{i}'
        json_line = json.dumps({
            'id': record_id,
            'text': text_content,
            'metadata': metadata
        }, ensure_ascii=False)
        f_out.write(json_line + '\n')
        if (i + 1) % 100 == 0:
            print(f'Processing bar parquet rows: {i + 1}')

    df_housing = pd.read_parquet(HOUSING_PARQUET_PATH, engine='pyarrow')
    for i, row in df_housing.iterrows():
        row_dict = row.to_dict()
        text_content = row_dict.pop('text', '')
        metadata = row_dict
        record_id = f'housing_{i}'
        json_line = json.dumps({
            'id': record_id,
            'text': text_content,
            'metadata': metadata
        }, ensure_ascii=False)
        f_out.write(json_line + '\n')
        if (i + 1) % 100 == 0:
            print(f'Processing housing parquet rows: {i + 1}')

print(f'Processing complete. Output saved to {OUTPUT_JSONL_PATH}')