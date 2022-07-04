from ctypes.wintypes import BOOL
import numpy as np
import random
import pandas as pd

DINT = 'int'
DDECIMAL = 'decimal'
DSTRING = 'string'
DDATE = 'datetime'
DBOOL = 'bool'

TNUMERIC = 'numeric'
TSTRING = 'string'
TDATE = 'date'

specific_dtypes = {
    'bool': DBOOL,

    'int8' : DINT,
    'int16' : DINT,
    'int32' : DINT,
    'int64' : DINT,

    'float32' : DDECIMAL,
    'float64' : DDECIMAL,

    'O' : DSTRING,
    'object' : DSTRING,
}

general_dtypes = {
    DBOOL : DBOOL,
    DINT : TNUMERIC,
    DDECIMAL : TNUMERIC,
    DSTRING : TSTRING,
    DDATE : TDATE,
}

def detect_dtype(elements : list, sample_size=500, confidence=0.01):
    if sample_size >= len(elements):
        elements_sample = pd.Series(elements, dtype=np.dtype('object'))
    else:
        elements_sample = pd.Series(random.sample(elements, sample_size), dtype=np.dtype('object'))
    
    max_errors = len(elements_sample) * confidence + 1

    numeric_cast = pd.to_numeric(elements_sample, errors='coerce')
    numeric_errors = numeric_cast.isna().sum()
    if numeric_errors < max_errors:
        numeric_cast.dropna(inplace=True)
        numeric_cast = pd.to_numeric(numeric_cast, downcast='integer')
        return specific_dtypes[str(numeric_cast.dtype)]

    del numeric_cast

    try:
        datetime_cast = pd.to_datetime(elements_sample, infer_datetime_format=True, errors='coerce')
        datetime_errors = datetime_cast.isna().sum()
        if datetime_errors < max_errors:
            return DDATE
    except Exception as e:
        print(e)

    return DSTRING

def cast_dtype(elements : pd.Series, specific_dtype : str):
    general_dtype = general_dtypes[specific_dtype]
    if general_dtype == TNUMERIC:
        if specific_dtype == DINT:
            temp = pd.to_numeric(elements, errors='coerce', downcast='integer')
            return pd.Series(temp, dtype=temp.dtype)
        else:
            return pd.Series(pd.to_numeric(elements, errors='coerce'))
    if general_dtype == TDATE:
        # cast dates to milliseconds so the ordinal characteristics are preserved
        temp = pd.to_datetime(elements, infer_datetime_format=True, errors='coerce')
        return pd.Series(temp, dtype=temp.dtype) 
    if general_dtype == BOOL:
        return pd.Series(elements, dtype=pd.BooleanDtype)
    return pd.Series(elements, dtype='string')

def fill_dtype(elements : pd.Series , specific_dtype : str):
    general_dtype = general_dtypes[specific_dtype]
    if general_dtype == TNUMERIC and specific_dtype != DBOOL:
        mean = elements.mean()
        elements.fillna(mean, inplace=True)
    else:
        mode = elements.mode()
        elements.fillna(mode, inplace=True)
    



