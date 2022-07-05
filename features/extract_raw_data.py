import pandas as pd
from utils import load_raw_data
import json
from type_detection import detect_dtype, cast_dtype, fill_dtype
import numpy as np
import warnings
warnings.filterwarnings("error")

# Columns of the result dataset
FID = 'fid'
FIELD_ID = 'field_id'
TRACE_TYPE = 'trace_type'
IS_XSRC = 'is_xsrc'
IS_YSRC = 'is_ysrc'
DATA = 'data'
DTYPE = 'dtype'

def get_trace_type(trace):
    ttype = trace.get('type')
    if ttype:
        if ttype == 'scatter':
            if trace.get('mode') in ['lines+markers', 'lines']:
                ttype = 'line'
            elif trace.get('line') and len(trace.get('line').keys()) > 0:
                ttype = 'line'
            elif trace.get('marker') and trace.get('marker').get(
                'line') and trace.get('marker').get('line').get('color') != 'transparent':
                ttype = 'line'
        return ttype
    return None

def get_src_uid(src):
    return src.split(':')[2]  


input_file_name = '../data/plot_data.tsv'
output_file_name = './traces_data_with_types.tsv'
with open(input_file_name, 'r') as f:
    raw_data = load_raw_data(f)

    for i, chunk in enumerate(raw_data):
        chunk_traces = []

        for chart_num, chart_obj in chunk.iterrows():
            fid = chart_obj.fid
            clean_fid = fid.split(':')[0]
        
            # Extract columns data from the dataset

            data = json.loads(chart_obj.table_data)
            columns = list(data.popitem()[1]['cols'].values())

            # Indicates if an error occurred while type casting
            error  = False 

            columns_info = {}
            for column in columns:
                uid = column['uid']
                column_data = column['data']                

                specific_dtype = detect_dtype(column_data) # Detect the data type
                try:
                    column_data = cast_dtype(column_data, specific_dtype) # Cast the data
                    fill_dtype(column_data, specific_dtype) # Fill missing data points
                    column_data = column_data.to_list()
                except (RuntimeWarning, Exception) as e:
                    error = True
                    break
                
                # Initialize the information structure of the column
                columns_info[uid] = { 
                    FID : fid, 
                    FIELD_ID : f'{clean_fid}:{uid}',
                    TRACE_TYPE : None,
                    IS_XSRC : False,
                    IS_YSRC : False,
                    DTYPE : specific_dtype,
                    DATA : column_data
                }

            # Extract columns outputs (i.e trace type and axis)
            if error:
                del columns_info
                continue

            specification = json.loads(chart_obj.chart_data)

            try:
                for trace in specification:
                    ttype = get_trace_type(trace)
                    xsrc = trace.get('xsrc')
                    ysrc = trace.get('ysrc')
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
            except:
                continue
            finally:
                del specification
            chunk_traces.extend(list(columns_info.values()))
            
        df = pd.DataFrame(chunk_traces, columns=[FID, FIELD_ID, TRACE_TYPE, IS_XSRC, IS_YSRC, DTYPE, DATA])
        df.to_csv(output_file_name, mode='a', index=False, header=(i == 0), sep='\t')
        del df