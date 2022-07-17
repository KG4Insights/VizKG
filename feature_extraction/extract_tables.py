from utils import load_raw_data
import argparse
from constants import FID, TRACE_TYPE, IS_XSRC,IS_YSRC, N_TRACES, N_XSRC, N_YSRC, DATA, DTYPE, LENGTH
from data_types import DSTRING, DINT, DFLOAT, DBOOL, DDATE
import pandas as pd
import json


def extract_tables(input_file_name, output_file_name):
    with open(output_file_name, 'w') as f:
        pass

    with open(input_file_name, 'r') as f:
        raw_data = load_raw_data(f, chunk_size=500)

        current_fid = None # The current table 
        table_error = False # A True value indicates that a column in the table has an error
        table_info = None # Information about the table

        for i, chunk in enumerate(raw_data):
            chunk_datasets = []

            for row in chunk.iterrows():
                column = row[1]

                # All columns are sorted by FID 
                
                if current_fid != column[FID]: # Founded new table
                    
                    if table_info and not table_error: 
                        # Save the results of the processed table
                        chunk_datasets.append(list(table_info.values()))

                    # Setup for the new table
                    current_fid = column[FID] 
                    table_info = {
                        FID : column[FID],
                        TRACE_TYPE : column[TRACE_TYPE],
                        N_TRACES : 0,
                        N_XSRC : 0,
                        N_YSRC : 0,
                        LENGTH : None
                    }
                    table_error = False

                if table_error:
                    continue # skip the columns until a new table is found
    
                # Exclusion criteria for tables

                # Has a column that is used in both axis at the same time or is not used at all
                if (column[IS_XSRC] and column[IS_YSRC]) or (not column[IS_XSRC] and not column[IS_YSRC]) :
                    table_error = True

                # The chart of the table has multiple trace types, this is a relaxation of the problem
                if column[TRACE_TYPE] != table_info[TRACE_TYPE]:
                    table_error = True

                # Account for errors during the extraction of the column data
                data = column[DATA]
                if data is None:
                    table_error = True
                    continue

                try: 
                    data = json.loads(data)
                except Exception as e:
                    table_error = True
                    continue
                

                if len(data) < 2:
                    table_error = True
                    continue
                
                # All data columns in a chart must have the same dimension.
                if table_info[LENGTH] is None:
                    table_info[LENGTH] = len(data)
                elif table_info[LENGTH] != len(data):
                    table_error = True
                    continue
                

                table_info[N_TRACES] += 1
                if column[IS_XSRC]:
                    table_info[N_XSRC] += 1
                if column[IS_YSRC]:
                    table_info[N_YSRC] += 1
                
                # The chart of the table must have a single column on the x-axis
                if table_info[N_XSRC] > 1:
                    table_error = True

            df = pd.DataFrame(chunk_datasets, columns=[FID, TRACE_TYPE, N_TRACES, N_XSRC, N_YSRC, LENGTH])
            df.to_csv(output_file_name, mode='a', index=False, header=(i == 0))

        if not table_error: # check the last table :)
            df = pd.DataFrame([list(table_info.values())], columns=[FID, TRACE_TYPE, N_TRACES, N_XSRC, N_YSRC, LENGTH])
            df.to_csv(output_file_name, mode='a', index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help='Input file path')
    parser.add_argument('--o', help='Output file path')

    args = parser.parse_args()

    if args.i:
        input_file_name = args.i
    else:
        input_file_name = '../data/corpus_columns.tsv'
    
    if args.o:
        output_file_name = args.o
    else:
        output_file_name = '../data/corpus_tables.csv'
    
    extract_tables(input_file_name, output_file_name)
