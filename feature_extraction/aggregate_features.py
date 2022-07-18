import pandas as pd
import numpy as np
from utils import load_raw_data
from constants import FID
from single_column_features import all_single_features_list
from pairwise_column_features import all_pairwise_features_list
import argparse

def mad_median(s):
    median = s.median()
    if median == np.nan:
        return None
    return (s-median).mean()

def coeff_var(s):
    mean = s.mean()
    var = s.var()
    if mean == np.nan or var == np.nan or not var:
        return None
    return mean / var


# aggregation functions for categorical features
c_aggregation_functions = {
        'num' : lambda s: s.sum(),
        'has' : lambda s : s.sum() > 0,
        'only_one' : lambda s : s.sum() == 1,
        'all' : lambda s : s.sum() == len(s),
        'percentage': lambda s : s.sum() / len(s)
}


# aggregation functions for quantitative features
q_aggregation_functions = {
        'mean' : lambda s: s.mean(),
        'var' : lambda s : s.var(),
        'std' : lambda s : s.std(),
        'mad_mean' : lambda s : s.mad(),
        'mad_median' : mad_median ,
        'coeff_var' : coeff_var,
        'min' : lambda s: s.min(),
        'max': lambda s : s.max(),
        'range' : lambda s : s.max() - s.min()
}


def aggregate_features(input_file_name, output_file_name, feature_list):
    with open(output_file_name, 'w') as f:
        pass

    header = [FID]
    for f in feature_list:
        f_name = f['name']
        f_type = f['type']

        aggregation_functions = c_aggregation_functions if f_type == 'boolean' else q_aggregation_functions

        for agg_func_name in aggregation_functions:
            header.append(f'{f_name}-{agg_func_name}')

    with open(input_file_name, 'r') as f:
        features = load_raw_data(f, chunk_size=2000, sep=',')

        current_fid = None
        table_columns = []

        tables_processed = 0

        for i, chunk in enumerate(features):
            
            chunk_aggregated = []
            
            for row in chunk.iterrows():
                column_features = row[1]
                
                if current_fid != column_features['fid']:
                    if current_fid:
                        #table_columns_df = pd.DataFrame(table_columns, columns=chunk.columns)
                        aggregated = _aggregate_features(current_fid, table_columns, feature_list)
                        chunk_aggregated.append(aggregated)
                        table_columns = []
                        tables_processed += 1

                    current_fid = column_features[FID]
                    
                table_columns.append(column_features)
            df = pd.DataFrame(chunk_aggregated, columns=header)
            df.to_csv(output_file_name, mode='a', index=False, header=(i == 0))
            print('Finished processing chunk ', i)
        if table_columns:
            aggregated = _aggregate_features(current_fid, table_columns, feature_list, output_file_name)
            df = pd.DataFrame([aggregated])
            df.to_csv(output_file_name, mode='a', index=False, header=False)


def _aggregate_features(table_fid, table_columns, feature_list):
    df = pd.DataFrame(table_columns)
    aggregated = [table_fid]
    
    for f in feature_list:
        f_name = f['name']
        f_type = f['type']

        v = df[f_name]

        aggregation_functions = c_aggregation_functions if f_type == 'boolean' else q_aggregation_functions

        for agg_func in aggregation_functions.values():
            aggregated.append(agg_func(v))

    return aggregated

                

         
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='Input file path')
    parser.add_argument('-o', required=True, help='Output file path')
    parser.add_argument('-s', help='Aggregate single column features', action='store_const', const=True, default=False)
    parser.add_argument('-p', help='Aggregate pairwise column features', action='store_const', const=True, default=False)
    args = parser.parse_args()

    input_file_name = args.i
    output_file_name = args.o 
    
    if not args.s and not args.p:
        print('A type of feature must be specified')
        exit(1)

    if args.s and args.p:
        print('Only one type of feature can be selected')
        exit(1)

    if args.s:
        print('Aggregating single column features from file ', input_file_name)
        aggregate_features(input_file_name, output_file_name, all_single_features_list)

    if args.p:
        print('Aggregating pairwise column features from file ', input_file_name)
        aggregate_features(input_file_name, output_file_name, all_pairwise_features_list)

