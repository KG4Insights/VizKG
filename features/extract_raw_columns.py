import tty
import pandas as pd
import argparse,  sys
from utils import load_raw_data
import json
from data_types import detect_dtype, cast_dtype, fill_dtype
from constants import FID, FIELD_ID, TRACE_TYPE, IS_XSRC,IS_YSRC, DTYPE, DATA


def extract_raw_data(input_file_name, output_file_name, verbose=False):
    with open(output_file_name, 'w') as output_file:
        # just to clean the output file
        pass

    with open(input_file_name, 'r') as input_file:
        raw_data = load_raw_data(input_file)
        
        print('Raw data loaded...')

        total_chunks = None
        total_columns = 0

        for i, chunk in enumerate(raw_data):
            total_chunks = i
            chunk_columns = []

            for chart_num, chart_obj in chunk.iterrows():
                fid = chart_obj.fid
                clean_fid = fid.split(':')[0]
            
                # Extract columns data from the dataset

                data = json.loads(chart_obj.table_data)
                columns = list(data.popitem()[1]['cols'].values())

                columns_info = {}

                for column in columns:
                    uid = column['uid']
                    column_data = column['data']                

                    # Infer the data type of the column using a small sample of elements
                    infered_dtype = detect_dtype(column_data) 

                    # Try to cast the column elements to the infered type, this returns the casted column
                    # and true_dtype, if the cast to the infered type is successful then infered_dtype == true_dtype
                    # otherwise true_dtype is a default, in this case string.
                    column_data, true_dtype = cast_dtype(column_data, infered_dtype) 

                    fill_dtype(column_data, true_dtype) # Fill missing data points

                    column_data = column_data.to_list()
                    
                    columns_info[uid] = { 
                        FID : fid, 
                        FIELD_ID : f'{clean_fid}:{uid}',
                        TRACE_TYPE : None,
                        IS_XSRC : False,
                        IS_YSRC : False,
                        DTYPE : true_dtype,
                        DATA : column_data
                    }

                # Extract columns outputs (i.e trace type and axis)
                specification = json.loads(chart_obj.chart_data)

                for trace in specification:
                    ttype = get_trace_type(trace)

                    try:
                        xsrc = trace.get('xsrc')
                    except:
                        xsrc = None

                    try:
                        ysrc = trace.get('ysrc')
                    except:
                        ysrc = None

                    if xsrc:
                        try:
                            xsrc = get_src_uid(xsrc)
                            columns_info[xsrc][IS_XSRC] = True
                            columns_info[xsrc][TRACE_TYPE] = ttype
                        except KeyError:
                            pass
                    if ysrc:
                        try:
                            ysrc = get_src_uid(ysrc)
                            columns_info[ysrc][IS_YSRC] = True
                            columns_info[ysrc][TRACE_TYPE] = ttype
                        except KeyError:
                            pass

                del specification

                chunk_columns.extend(list(columns_info.values()))

               
            df = pd.DataFrame(chunk_columns, columns=[FID, FIELD_ID, TRACE_TYPE, IS_XSRC, IS_YSRC, DTYPE, DATA])
            df.to_csv(output_file_name, mode='a', index=False, header=(i == 0), sep='\t')

            if verbose:
                print(f'Finished processing chunk {i}, saved {len(chunk_columns)} data columns.') 
            
            total_columns += len(chunk_columns)

            del df, chunk_columns

        print('Finished execution')
        print('Processed chunks: ', total_chunks + 1)
        print('Processed columns: ', total_columns)


def get_trace_type(trace):
    try:
        ttype = trace.get('type')
    except:
        ttype = None
    if ttype:
        if ttype == 'scatter':
            try:
                if trace.get('mode') in ['lines+markers', 'lines']:
                    ttype = 'line'
                elif trace.get('line') and len(trace.get('line').keys()) > 0:
                    ttype = 'line'
                elif trace.get('marker') and trace.get('marker').get(
                    'line') and trace.get('marker').get('line').get('color') != 'transparent':
                    ttype = 'line'
            except:
                pass
        return ttype
    return None


def get_src_uid(src):
    return src.split(':')[2]  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', help='Input file path')
    parser.add_argument('--o', help='Output file path')

    args = parser.parse_args()

    try:
        input_file_name = args.i
    except:
        input_file_name = '../data/raw_charts.tsv'
    
    try:
        output_file_name = args.o
    except:
        output_file_name = './raw_columns.tsv'
    
    extract_raw_data(input_file_name, output_file_name, verbose=True)