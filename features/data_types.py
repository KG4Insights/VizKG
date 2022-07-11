from datetime import datetime
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

ISO_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'

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
        elements_sample = pd.Series(random.sample(elements, sample_size), dtype=np.dtype('object'))
    
    # This is the maximun misses allow when casting to a specific type
    max_errors = len(elements_sample) * confidence + 1

    
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
            temp = [ e.strftime(ISO_FORMAT) for e in temp ]
            return pd.Series(temp, dtype=pd.StringDtype()), DDATE
        except:
            pass
    
    if dtype == DBOOL:
        try:
            return pd.Series(elements, dtype=np.int8), DBOOL
        except:
            pass

    elements = [str(e) for e in elements]
    return pd.Series(elements, dtype=pd.StringDtype()), DSTRING


def fill_dtype(elements : pd.Series , dtype : str):
    try:
        if dtype in (DINT, DFLOAT):
            elements.replace([np.inf, -np.inf, None], np.nan, inplace=True)
            mean = elements.mean(skipna=True)
            elements.fillna(mean, inplace=True)
        else:
            elements.replace([None], np.nan, inplace=True)
            mode = elements.mode(dropna=True)
            elements.fillna(mode, inplace=True)
    except:
        pass


def cast_to_numpy(elements, dtype):
    if dtype == DSTRING:
        return pd.Series(elements, dtype=pd.StringDtype()).to_numpy()
    elif dtype == DINT:
        return pd.Series(pd.to_numeric(elements, downcast='integer')).to_numpy()
    elif dtype == DFLOAT:
        return pd.Series(pd.to_numeric(elements)).to_numpy()
    elif dtype == DBOOL:
        return pd.Series(pd.to_numeric(elements, downcast='integer')).to_numpy()
    elif dtype == DDATE:
        temp = [datetime.strptime(e, ISO_FORMAT) for e in elements]
        return pd.Series(temp).to_numpy().astype('int')


