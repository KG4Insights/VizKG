import pandas as pd
import numpy as np
from time import time, strftime
from collections import OrderedDict, Counter
from scipy.stats import entropy

def load_raw_data(data_file_stream, chunk_size=500):

    df = pd.read_table(
        data_file_stream,
        on_bad_lines='skip',
        chunksize=chunk_size,
        encoding='utf-8'
    )
    
    return df


def get_unique(li, preserve_order=False):
    if preserve_order:
        seen = set()
        seen_add = seen.add
        return [x for x in li if not (x in seen or seen_add(x))]
    else:
        return np.unique(li)