from single_column_features import get_single_column_features, single_column_features_names
import pandas as pd
from utils import load_raw_data
import json
from constants import FID, FIELD_ID, TRACE_TYPE, DATA, DTYPE, IS_XSRC, IS_YSRC
import argparse
import numpy as np

single_column_features_header = [FID, FIELD_ID, TRACE_TYPE, IS_XSRC, IS_YSRC] + single_column_features_names

def extract_single_column_features(input_file_name, output_file_name):
    with open(output_file_name, 'w') as f:
        pass
    print(f'Extracting features from {input_file_name}')
    with open(input_file_name, 'r') as f:
        data = load_raw_data(f)
        for i, chunk in enumerate(data):
            chunk_features = []
            for index, column in chunk.iterrows():
                column_data = json.loads(column[DATA])
                column_data = np.array(column_data)
                column_output = [column[FID], column[FIELD_ID], column[TRACE_TYPE], column[IS_XSRC], column[IS_YSRC]]
                features = column_output + get_single_column_features(column_data, column[DTYPE])
                chunk_features.append(features)
            df = pd.DataFrame(chunk_features, columns=single_column_features_header)
            df.to_csv(output_file_name, mode='a', index=False, header=(i == 0))
            print(f'finished processing chunk {i}, extracted features from {len(chunk_features)} columns.')


if __name__ == '__main__':
    sample_file_name1 = '../data/corpus_sample_3types.tsv'
    sample_file_name2 = '../data/corpus_sample_5types.tsv'
    output_file_name1 = '../features/single_column_features_3types.csv'
    output_file_name2 = '../features/single_column_features_5types.csv'
    extract_single_column_features(sample_file_name1, output_file_name1)
    print()
    extract_single_column_features(sample_file_name2, output_file_name2)

