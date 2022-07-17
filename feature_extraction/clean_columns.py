from utils import load_raw_data
import argparse
from constants import FID, TRACE_TYPE, IS_XSRC,IS_YSRC, N_TRACES, N_XSRC, N_YSRC, DATA, DTYPE, LENGTH
from data_types import DSTRING, DINT, DFLOAT, DBOOL, DDATE
import pandas as pd


def clean_corpus_columns(input_columns_file_name, input_tables_file_name, output_file_name):
    with open(output_file_name, 'w') as f:
        pass
    
    tables = None
    with open(input_tables_file_name, 'r') as f:
        tables_df = pd.read_csv(f)
        tables = set(tables_df[FID]) # Get the correct tables

    with open(input_columns_file_name, 'r') as f:
        raw_data = load_raw_data(f, chunk_size=2000)
        total_columns = 0
        for i, chunk in enumerate(raw_data):
            chunk_columns = []
            for row in chunk.iterrows():
                column = row[1]

                if column[FID] in tables:
                    chunk_columns.append(column)

            df = pd.DataFrame(chunk_columns, columns=chunk.columns)
            df.to_csv(output_file_name, mode='a', index=False, header=(i == 0), sep='\t')
            print(f'Processed chunk {i}, saved {len(chunk_columns)} columns.')
            total_columns += len(chunk_columns)
        print(f'Finished cleaning the corpus columns, saved a total of {total_columns} columns.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', help='Columns file path')
    parser.add_argument('--t', help='Tables file path')
    parser.add_argument('--o', help='Output file path')

    args = parser.parse_args()

    if args.c:
        input_columns_file_name = args.c
    else:
        input_columns_file_name = '../data/corpus_columns.tsv'
    
    if args.t:
        input_tables_file_name = args.t
    else:
        input_tables_file_name = '../data/corpus_tables_outputs.csv'
    
    if args.o:
        output_file_name = args.o
    else:
        output_file_name = '../data/cleaned_corpus_columns.tsv'
    
    clean_corpus_columns(input_columns_file_name, input_tables_file_name, output_file_name)
