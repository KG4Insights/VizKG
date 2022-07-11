import numpy as np
from scipy.stats import entropy, normaltest, mode, kurtosis, skew, pearsonr, moment
import pandas as pd
from data_types import data_types_list, var_types_list, dtype_to_vtype, CATEG, QUANT, TIME
from utils import get_unique



basic_features_list = [{'name': 'length', 'type': 'numeric'}]


for data_type in data_types_list:
    basic_features_list.append({'name': f'data_type_is_{data_type}', 'type': 'boolean'})


for var_type in var_types_list:
    basic_features_list.append({'name': f'var_type_is_{var_type}', 'type': 'boolean'})


uniqueness_features_list = [
    {'name': 'num_unique_elements', 'type': 'numeric'},
    {'name': 'unique_percent', 'type': 'numeric'},
    {'name': 'is_unique', 'type': 'boolean'}
]

statistical_features_list = [
    {'name': 'mean', 'type': 'numeric'},
    {'name': 'median', 'type': 'numeric'},
    {'name': 'var', 'type': 'numeric'},
    {'name': 'std', 'type': 'numeric'},
    {'name': 'coeff_var', 'type': 'numeric'},
    {'name': 'min', 'type': 'numeric'},
    {'name': 'max', 'type': 'numeric'},
    {'name': 'range', 'type': 'numeric'},

    {'name': 'entropy', 'type': 'numeric'},
    #{'name': 'gini', 'type': 'numeric'},
    {'name': 'q25', 'type': 'numeric'},
    {'name': 'q75', 'type': 'numeric'},
    {'name': 'med_abs_dev', 'type': 'numeric'},
    {'name': 'avg_abs_dev', 'type': 'numeric'},
    {'name': 'quant_coeff_disp', 'type': 'numeric'},
    {'name': 'skewness', 'type': 'numeric'},
    {'name': 'kurtosis', 'type': 'numeric'},
    {'name': 'moment_5', 'type': 'numeric'},
    {'name': 'moment_6', 'type': 'numeric'},
    {'name': 'moment_7', 'type': 'numeric'},
    {'name': 'moment_8', 'type': 'numeric'},
    {'name': 'moment_9', 'type': 'numeric'},
    {'name': 'moment_10', 'type': 'numeric'},

    {'name': 'percent_outliers_15iqr', 'type': 'numeric'},
    {'name': 'percent_outliers_3iqr', 'type': 'numeric'},
    {'name': 'percent_outliers_1_99', 'type': 'numeric'},
    {'name': 'percent_outliers_3std', 'type': 'numeric'},
    {'name': 'has_outliers_15iqr', 'type': 'boolean'},
    {'name': 'has_outliers_3iqr', 'type': 'boolean'},
    {'name': 'has_outliers_1_99', 'type': 'boolean'},
    {'name': 'has_outliers_3std', 'type': 'boolean'},

    {'name': 'normality_statistic', 'type': 'numeric'},
    {'name': 'normality_p', 'type': 'numeric'},
    {'name': 'is_normal_5', 'type': 'boolean'},
    {'name': 'is_normal_1', 'type': 'boolean'},
]

sequence_features_list = [
    {'name': 'is_sorted', 'type': 'boolean'},
    {'name': 'is_monotonic', 'type': 'boolean'},
    {'name': 'sortedness', 'type': 'numeric'},

    {'name': 'lin_space_sequence_coeff', 'type': 'numeric'},
    {'name': 'log_space_sequence_coeff', 'type': 'numeric'},
    {'name': 'is_lin_space', 'type': 'boolean'},
    {'name': 'is_log_space', 'type': 'boolean'},
]

single_column_features_names = [f['name'] for f in basic_features_list + uniqueness_features_list + statistical_features_list + sequence_features_list]

def get_basic_features(v, dtype, vtype):
    r = dict([(f['name'], None) for f in basic_features_list])
    r['length'] = len(v)
    r[f'data_type_is_{dtype}'] = True
    r[f'var_type_is_{vtype}'] = True
    return r


def get_uniqueness_features(v, dtype, vtype):
    r = dict([(f['name'], None) for f in uniqueness_features_list])
    
    if not len(v):
        return r
    
    if vtype in [CATEG, TIME] or dtype == 'integer':
        unique_elements = get_unique(v)
        r['num_unique_elements'] = unique_elements.size
        r['unique_percent'] = (r['num_unique_elements'] / len(v))
        r['is_unique'] = (r['num_unique_elements'] == len(v))
    return r


def get_statistical_features(v, var_type):
    r = dict([(f['name'], None) for f in statistical_features_list])

    if not len(v):
        return r
    
    if var_type in (CATEG, TIME):
        # if the variable is categorical perform the analysis over the histogram of the data
        v = np.array(pd.value_counts(v))
        
    sample_mean = np.mean(v)
    sample_median = np.median(v)
    sample_var = np.var(v)
    sample_min = np.min(v)
    sample_max = np.max(v)
    sample_std = np.std(v)
    q1, q25, q75, q99 = np.percentile(v, [0.01, 0.25, 0.75, 0.99])
    iqr = q75 - q25

    r['mean'] = sample_mean
    r['median'] = sample_median
    r['var'] = sample_var
    r['std'] = sample_std
    r['coeff_var'] = (sample_mean / sample_var) if sample_var else None
    r['min'] = sample_min
    r['max'] = sample_max
    r['range'] = r['max'] - r['min'] # cuantitative

    if var_type in (CATEG, TIME):
        r['entropy'] = entropy(v / np.sum(v))
    
    #r['gini'] = gini(v) # i dont know how this is computed 
    r['q25'] = q25
    r['q75'] = q75
    r['med_abs_dev'] = np.median(np.absolute(v - sample_median))
    r['avg_abs_dev'] = np.mean(np.absolute(v - sample_mean))
    r['quant_coeff_disp'] = (q75 - q25) / (q75 + q25)
    r['coeff_var'] = sample_var / sample_mean
    r['skewness'] = skew(v)
    r['kurtosis'] = kurtosis(v)
    r['moment_5'] = moment(v, moment=5)
    r['moment_6'] = moment(v, moment=6)
    r['moment_7'] = moment(v, moment=7)
    r['moment_8'] = moment(v, moment=8)
    r['moment_9'] = moment(v, moment=9)
    r['moment_10'] = moment(v, moment=10)

    # Outliers
    outliers_15iqr = np.logical_or(v < (q25 - 1.5 * iqr), v > (q75 + 1.5 * iqr))
    outliers_3iqr = np.logical_or(v < (q25 - 3 * iqr), v > (q75 + 3 * iqr))
    outliers_1_99 = np.logical_or(v < q1, v > q99)
    outliers_3std = np.logical_or(v < (sample_mean -3 *sample_std), v > (sample_mean + 3 *sample_std))

    r['percent_outliers_15iqr'] = np.sum(outliers_15iqr) / len(v)
    r['percent_outliers_3iqr'] = np.sum(outliers_3iqr) / len(v)
    r['percent_outliers_1_99'] = np.sum(outliers_1_99) / len(v)
    r['percent_outliers_3std'] = np.sum(outliers_3std) / len(v)

    r['has_outliers_15iqr'] = np.any(outliers_15iqr)
    r['has_outliers_3iqr'] = np.any(outliers_3iqr)
    r['has_outliers_1_99'] = np.any(outliers_1_99)
    r['has_outliers_3std'] = np.any(outliers_3std)

    # Statistical Distribution
    if len(v) >= 8:
        normality_k2, normality_p = normaltest(v)
        r['normality_statistic'] = normality_k2
        r['normality_p'] = normality_p
        r['is_normal_5'] = (normality_p < 0.05)
        r['is_normal_1'] = (normality_p < 0.01)

    return r


def get_sequence_features(v, vtype):
    r = dict([(f['name'], None) for f in sequence_features_list])
    if not len(v):
        return r
    sorted_v = np.sort(v)

    if vtype == 'c':
        r['is_sorted'] = np.array_equal(sorted_v, v)

    if vtype == 't':
        v = v.astype('int')
        sorted_v = sorted_v.astype('int')
    if vtype in ['t', 'q']:
        sequence_incremental_subtraction = np.subtract(sorted_v[:-1], sorted_v[1:]).astype(int)
        r['is_monotonic'] = np.all(sequence_incremental_subtraction <= 0) or np.all(sequence_incremental_subtraction >= 0)
        r['sortedness'] = np.absolute(pearsonr(v, sorted_v)[0]) 
        r['is_sorted'] = np.array_equal(sorted_v, v)
    if vtype == 'q':
        sequence_incremental_division = np.divide(sorted_v[:-1], sorted_v[1:])
        sequence_incremental_subtraction = np.diff(sorted_v)
        r['lin_space_sequence_coeff'] = np.std(sequence_incremental_subtraction) / np.mean(sequence_incremental_subtraction)
        r['log_space_sequence_coeff'] = np.std(sequence_incremental_division) / np.mean(sequence_incremental_division)
        r['is_lin_space'] = r['lin_space_sequence_coeff'] <= 0.001
        r['is_log_space'] = r['log_space_sequence_coeff'] <= 0.001
    return r


def get_single_column_features(v, dtype):
    vtype = dtype_to_vtype[dtype]
    v_hist = v
    if dtype in [CATEG, TIME]:
        v_hist = np.array(pd.value_counts(v)) 
    
    basic_features = get_basic_features(v, dtype, vtype)
    uniqueness_features = get_uniqueness_features(v, dtype, vtype)
    statistical_features = get_statistical_features(v_hist, vtype)
    sequence_features = get_sequence_features(v, vtype)
    return list(basic_features.values()) + list(uniqueness_features.values()) + list(statistical_features.values()) + list(sequence_features.values())


    
