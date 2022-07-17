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


def get_list_uniqueness(l):
    if len(l):
        return len(np.unique(l)) / len(l)
    else:
        return None

def calculate_overlap(a_data, b_data):
    a_min, a_max = np.min(a_data), np.max(a_data)
    a_range = a_max - a_min
    b_min, b_max = np.min(b_data), np.max(b_data)
    b_range = b_max - b_min
    has_overlap = False
    overlap_percent = 0
    if (a_max >= b_min) and (b_min >= a_min):
        has_overlap = True
        overlap = (a_max - b_min)
    if (b_max >= a_min) and (a_min >= b_min):
        has_overlap = True
        overlap = (b_max - a_min)
    if has_overlap:
        overlap_percent = max(overlap / a_range, overlap / b_range)
    if ((b_max >= a_max) and (b_min <= a_min)) or (
            (a_max >= b_max) and (a_min <= b_min)):
        has_overlap = True
        overlap_percent = 1
    return has_overlap, overlap_percent
