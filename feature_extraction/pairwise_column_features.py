from cmath import e
import numpy as np
import pandas as pd 
from scipy.stats import pearsonr, f_oneway, chi2_contingency, ks_2samp
from utils import get_unique, get_list_uniqueness, calculate_overlap
from data_types import dtype_to_vtype


general_pairwise_features_list = [
    {'name': 'has_shared_elements', 'type': 'boolean'},
    {'name': 'num_shared_elements', 'type': 'numeric'},
    {'name': 'percent_shared_elements', 'type': 'numeric'}, 
    {'name': 'identical', 'type': 'boolean'},
    {'name': 'has_shared_unique_elements', 'type': 'boolean'},
    {'name': 'num_shared_unique_elements', 'type': 'numeric'},
    {'name': 'percent_shared_unique_elements', 'type': 'numeric'}, 
    {'name': 'identical_unique', 'type': 'boolean'},    
]


qq_pairwise_features_list = [
    {'name': 'correlation_value', 'type': 'numeric'},
    {'name': 'correlation_p', 'type': 'numeric'},
    {'name': 'correlation_significant_005', 'type': 'boolean'},
    {'name': 'ks_statistic', 'type': 'numeric'},
    {'name': 'ks_p', 'type': 'numeric'},  
    {'name': 'ks_significant_005', 'type': 'boolean'},    
    {'name': 'percent_range_overlap', 'type': 'numeric'},
    {'name': 'has_range_overlap', 'type': 'numeric'},
]


cc_pairwise_features_list = [
    {'name': 'chi2_statistic', 'type': 'numeric'},
    {'name': 'chi2_p', 'type': 'numeric'},
    {'name': 'chi2_significant_005', 'type': 'numeric'},
    {'name': 'is_nested', 'type': 'boolean'}, 
    {'name': 'nestedness', 'type': 'numeric'}, 
    {'name': 'nestedness_95', 'type': 'boolean'}, 
]


cq_pairwise_features_list = [
    {'name': 'one_way_anova_statistic', 'type': 'numeric'},
    {'name': 'one_way_anova_p', 'type': 'numeric'},  
    {'name': 'one_way_anova_significant_005', 'type': 'boolean'},
]


statistical_pairwise_features_list = \
    qq_pairwise_features_list + \
    cc_pairwise_features_list + \
    cq_pairwise_features_list

all_pairwise_features_list = \
    general_pairwise_features_list + \
    statistical_pairwise_features_list

pairwise_column_features_names = [f['name'] for f in all_pairwise_features_list]

def get_general_pairwise_features(a_data, b_data, a_unique_data, b_unique_data):
    r = dict([ (f['name'], None) for f in general_pairwise_features_list ])
    
    a_unique_data = set(a_unique_data)
    b_unique_data = set(b_unique_data)
    num_identical_elements = np.count_nonzero(a_data == b_data)
    r['has_shared_elements'] = (num_identical_elements > 0)
    r['num_shared_elements'] = num_identical_elements
    r['percent_shared_elements'] = num_identical_elements / len(a_data)
    r['identical'] = num_identical_elements == len(a_data)

    num_shared_unique_elements = len(a_unique_data.intersection(b_unique_data))
    r['has_shared_unique_elements'] = (num_shared_unique_elements > 0)
    r['num_shared_unique_elements'] = num_shared_unique_elements
    r['percent_shared_unique_elements'] = num_shared_unique_elements/ max(len(a_unique_data), len(b_unique_data))
    r['identical_unique'] = (a_unique_data == b_unique_data)  
    return r

def get_statistical_pairwise_features(a_data, b_data, a_unique_data, b_unique_data, a_vtype, b_vtype, MAX_GROUPS=50):
    r = dict([ (f['name'], None) for f in statistical_pairwise_features_list ])

    a = ('a_data', a_unique_data)
    b = ('b_data', b_unique_data)

    if (a_vtype == 'q' and b_vtype == 'q'):
        correlation_value, correlation_p = pearsonr(a_data, b_data)
        ks_statistic, ks_p = ks_2samp(a_data, b_data)
        has_overlap, overlap_percent = calculate_overlap(a_data, b_data)

        r['correlation_value'] = correlation_value
        r['correlation_p'] = correlation_p
        r['correlation_significant_005'] = (correlation_p < 0.05)

        r['ks_statistic'] = ks_statistic
        r['ks_p'] = ks_p
        r['ks_significant_005'] = (ks_p < 0.05)

        r['has_range_overlap'] = has_overlap
        r['percent_range_overlap'] = overlap_percent

    if (a_vtype == 'c' and b_vtype == 'c'):
        if len(a_unique_data) > MAX_GROUPS or len(b_unique_data) > MAX_GROUPS:
            return r
        df = pd.DataFrame({ 'a_data': a_data, 'b_data': b_data })
        ct = pd.crosstab(a_data, b_data)
        chi2_statistic, chi2_p, dof, exp_frequencies = chi2_contingency(ct)

        r['chi2_statistic'] = chi2_statistic
        r['chi2_p'] = chi2_p
        r['chi2_significant_005'] = (chi2_p < 0.05)

        nestedness_values = []
        for parent, child in ([a, b], [b, a]):
            
            parent_name, parent_unique_data = parent
            child_name, child_unique_data = child
           
            child_field_unique_corresponding_values = []
            unique_parent_field_values = parent_unique_data
            for unique_parent_field_value in unique_parent_field_values:
                child_field_unique_corresponding_values.extend(set(df[df[parent_name] == unique_parent_field_value][child_name]))
            nestedness = get_list_uniqueness(child_field_unique_corresponding_values)     
            nestedness_values.append(nestedness) 

        r['nestedness'] = max(nestedness_values)
        r['nestedness_95'] == (nestedness > 0.95)

    if (a_vtype == 'q' and b_vtype == 'c') or (a_vtype == 'c' and  b_vtype == 'q'):
        c_field = a if a_vtype == 'c' else b
        q_field = a if a_vtype == 'q' else b
        
        c_field_name, c_field_unique_data = c_field
        q_field_name, q_field_unique_data = q_field

        unique_c_field_values = c_field_unique_data
        if 1 < len(unique_c_field_values) <= MAX_GROUPS:
            df = pd.DataFrame({ 'a_data': a_data, 'b_data': b_data })
            group_values = [ df[df[c_field_name] == v][q_field_name] for v in unique_c_field_values ]
            anova_result = f_oneway(*group_values)  

            r['one_way_anova_statistic'] = anova_result.statistic
            r['one_way_anova_p'] = anova_result.pvalue
            r['one_way_anova_significant_005'] = (anova_result.pvalue < 0.05)
    return r

def get_pairwise_column_features(a_data, b_data, a_dtype, b_dtype):
    a_vtype = dtype_to_vtype[a_dtype]
    b_vtype = dtype_to_vtype[b_dtype]

    a_unique_data = get_unique(a_data)
    b_unique_data = get_unique(b_data)

    general_pairwise_features = get_general_pairwise_features(a_data, b_data, a_unique_data, b_unique_data)
    statistical_pairwise_features = get_statistical_pairwise_features(a_data, b_data, a_unique_data, b_unique_data, a_vtype, b_vtype)
        
    return list(general_pairwise_features.values()) + list(statistical_pairwise_features.values())
