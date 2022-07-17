from datetime import datetime
from collections.abc import Hashable
import numpy as np
import random
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

CATEG = 'c'
QUANT = 'q'
TIME = 't'

var_types_list = [CATEG, QUANT, TIME]

DINT = 'int'
DFLOAT = 'float'
DSTRING = 'string'
DDATE = 'datetime'
DBOOL = 'bool'

data_types_list = [DINT, DFLOAT, DSTRING, DDATE, DBOOL]

dtype_to_vtype = {
    DINT : QUANT,
    DFLOAT : QUANT,
    DSTRING : CATEG,
    DBOOL : CATEG,
    DDATE : TIME
}

dtypes = {
    'bool': DBOOL,

    'int8' : DINT,
    'int16' : DINT,
    'int32' : DINT,
    'int64' : DINT,

    'float32' : DFLOAT,
    'float64' : DFLOAT,

    'O' : DSTRING,
    'object' : DSTRING,
}


def detect_dtype(elements : list, sample_size=500, confidence=0.01):

    # Gets a sample from the elements
    if sample_size >= len(elements):
        elements_sample = pd.Series(elements, dtype=np.dtype('object'))
    else:
        elements_sample = random.sample(elements, sample_size)
   
    elements_sample = [ e for e in elements_sample if e is not None]
    
    elements_sample = pd.Series(elements_sample, dtype=np.dtype('object'))
    
    # This is the maximun misses allow when casting to a specific type
    max_errors = len(elements_sample) / 2

    
    try: 
        # Try casting to numeric types (e.g. float and int)
        numeric_cast = pd.to_numeric(elements_sample, errors='coerce')
        numeric_errors = numeric_cast.isna().sum()
        if numeric_errors < max_errors:  # Matched numeric
            numeric_cast.dropna(inplace=True)
            numeric_cast = pd.to_numeric(numeric_cast, downcast='integer')
            return dtypes[str(numeric_cast.dtype)]
    except: 
        pass

    del numeric_cast

    try:
        # Try casting to datetime
        datetime_cast = pd.to_datetime(elements_sample, infer_datetime_format=True, errors='coerce')
        datetime_errors = datetime_cast.isna().sum()
        if datetime_errors < max_errors: # Matched datetime
            return DDATE
    except:
        pass

    return DSTRING # The default type of a column is string


def cast_dtype(elements : pd.Series, dtype : str):

    if dtype == DINT:
        try:
            temp = pd.to_numeric(elements, errors='coerce', downcast='integer')
            return pd.Series(temp, dtype=temp.dtype), DINT
        except:
            pass
    
    if dtype == DFLOAT:
        try: 
            return pd.Series(pd.to_numeric(elements, errors='coerce')), DFLOAT
        except:
            pass
    
    if dtype == DDATE:
        try:
            temp = pd.to_datetime(elements, infer_datetime_format=True, errors='coerce', utc=True)
            # Cast datetime to milliseconds for more efficient storage and computation
            temp = [ e.timestamp() if e is not pd.NaT else np.nan for e in temp]
            return pd.Series(temp), DDATE
        except:
            pass
    
    if dtype == DBOOL:
        try:
            return pd.Series(elements, dtype=np.int8), DBOOL
        except:
            pass
    
    # Use bag of words representation for string arrays since it's more efficient
    # for storage and computation speed.
    bag = {}
    temp = []
    for e in elements:
        if not e or not isinstance(e, str): 
            temp.append(np.nan)
        elif e in bag:
            temp.append(bag[e])
        else:
            bag[e] = len(bag)
            temp.append(bag[e])

    return pd.Series(temp), DSTRING


def fill_dtype(elements : pd.Series , dtype : str):

    if dtype in (DINT, DFLOAT):

        elements.replace([np.inf, -np.inf], np.nan, inplace=True)
        mean = elements.mean(skipna=True)
        if mean == np.nan or np.isinf(mean): # Due to overflows
            return pd.Series([]), False # Could be possible to use the mode for inputation if mean fails
        elements.fillna(mean, inplace=True)

    else:

        mode = elements.mode(dropna=True)
        if len(mode) > 1:
            mode = mode[0]
        elif len(mode) < 1: # When all elements are NaN
            return pd.Series([]), False
        elements.fillna(mode, inplace=True)

    return elements, True
