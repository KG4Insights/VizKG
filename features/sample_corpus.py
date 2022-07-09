import pandas as pd
from constants import FID, TRACE_TYPE
from utils import load_raw_data
import argparse
import random


def sample_corpus(tables_file_name, columns_file_name, output_file_name, trace_types, sample_size, seed):
    tables_sample = sample_tables_by_trace_type(tables_file_name, trace_types, sample_size, seed)
    sample_columns_by_fid(tables_sample, columns_file_name, output_file_name)


def sample_tables_by_trace_type(tables_file_name, types, sample_size, seed):
    random.seed(seed)
    with open(tables_file_name, 'r') as f:
        tables_df = pd.read_csv(f)
        tables_sample = []
        for t in types:
            tables_x_type = tables_df[tables_df[TRACE_TYPE] == t]
            try:
                type_sample = random.sample(list(tables_x_type[FID]), sample_size)
            except:
                type_sample = list(tables_x_type[FID])
            tables_sample.extend(type_sample)
        return set(tables_sample)


def sample_columns_by_fid(tables_sample, columns_file_name, output_file_name):
    with open(output_file_name, 'w') as f:
        # Clean the output file in case it exists
        pass

    with open(columns_file_name, 'r') as f:
        raw_data = load_raw_data(f)
        for i, chunk in enumerate(raw_data):
            columns = []
            for row in chunk.iterrows():
                trace = row[1]
                fid = trace[FID]
                if fid in tables_sample:
                    columns.append(trace)
            df = pd.DataFrame(columns)
            df.to_csv(output_file_name, mode='a', index=False, header=(i == 0), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', help='Tables file path')
    parser.add_argument('--c', help='Columns file path')
    parser.add_argument('--seed', help='Random seed')

    args = parser.parse_args()

    if args.t:
        tables_file_name = args.t
    else:
        tables_file_name = '../data/corpus_tables.csv'
    
    if args.c:
        columns_file_name = args.c
    else:
        columns_file_name = '../data/corpus_columns.tsv'
    
    if args.seed:
        seed = args.seed
    else:
        seed = 30
    
   
    trace_5types = ['bar', 'box', 'histogram', 'line', 'scatter']
    output_file_name = '../data/corpus_sample_5types.tsv'
    sample_corpus(tables_file_name, columns_file_name, output_file_name, trace_5types, 4000, seed)

    trace_3types = ['bar', 'line', 'scatter']
    output_file_name = '../data/corpus_sample_3types.tsv'
    sample_corpus(tables_file_name, columns_file_name, output_file_name, trace_3types, 10000, seed)

