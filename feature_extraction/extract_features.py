from single_column_features import get_single_column_features, single_column_features_names
from pairwise_column_features import get_pairwise_column_features, pairwise_column_features_names
import pandas as pd
from utils import load_raw_data
import json
from constants import FID, FIELD_ID, TRACE_TYPE, DATA, DTYPE, IS_XSRC, IS_YSRC
import argparse
import numpy as np
from itertools import combinations

single_column_features_header = [FID, FIELD_ID, TRACE_TYPE, IS_XSRC, IS_YSRC] + single_column_features_names
pairwise_column_features_header = [FID, 'a_field_id', 'b_field_id'] + pairwise_column_features_names

def extract_single_column_features(input_file_name, output_file_name):
    with open(output_file_name, 'w') as f:
        pass
    print(f'Extracting features from {input_file_name}')
    with open(input_file_name, 'r') as f:
        data = load_raw_data(f,chunk_size=1000)
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


def extract_pairwise_column_features(input_file_name, output_file_name):
    with open(output_file_name, 'w') as f:
        pass
    print(f'Extracting pairwise column features from {input_file_name}')
    with open(input_file_name, 'r') as f:
        data = load_raw_data(f)
        current_fid = None
        table_columns = []
        tables_processed = 0
        for i, chunk in enumerate(data):

            for row in chunk.iterrows():
                column = row[1]

                if current_fid != column[FID]:

                    if current_fid:            
                        _extract_pairwise_column_features(output_file_name, current_fid, table_columns, tables_processed == 0)
                        table_columns = []
                        tables_processed += 1

                    current_fid = column[FID]
                
                column[DATA] = np.array(json.loads(column[DATA]))
                table_columns.append(column)
            print(f'Finished processing chunk {i}')
        if table_columns:
            _extract_pairwise_column_features(output_file_name, current_fid, table_columns, tables_processed == 0)
            tables_processed += 1
        print(f'Finished, processed a total of {tables_processed} tables')


def _extract_pairwise_column_features(output_file_name, table_fid, table_columns, header):
    table_pairwise_features = []
    for a, b in combinations(table_columns, 2):
        pairwise_features = [
            table_fid,
            a[FIELD_ID],
            b[FIELD_ID]
        ]
        
        a_data, a_dtype = a[DATA], a[DTYPE]
        b_data, b_dtype = b[DATA], b[DTYPE]

        pairwise_features += get_pairwise_column_features(a_data, b_data, a_dtype, b_dtype)
        table_pairwise_features.append(pairwise_features)
    df = pd.DataFrame(table_pairwise_features, columns=pairwise_column_features_header)
    df.to_csv(output_file_name, mode='a', index=False, header=header)



if __name__ == '__main__':
    input_file_name = '../data/cleaned_corpus_columns.tsv'
    soutput_file_name = '../features/single_column_features.csv'
    poutput_file_name = '../features/pairwise_column_features.csv'

    #extract_single_column_features(input_file_name, soutput_file_name)
    extract_pairwise_column_features(input_file_name, poutput_file_name)

