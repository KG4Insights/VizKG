import pandas as pd


def load_raw_data(data_file_stream, chunk_size=500):

    df = pd.read_table(
        data_file_stream,
        on_bad_lines='skip',
        chunksize=chunk_size,
        encoding='utf-8'
    )
    
    return df