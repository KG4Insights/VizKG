# Feature Extraction

## Corpus

[Plotly](https://plot.ly) is a software company that develops online data analytics and visualization tools, this company launched a collaborative platform called [Plotly Community Feed ](https://plot.ly/feed) in which the users can publish charts and also search, sort and filter the charts published by other users of the platform. The charts published in this platform can be accessed using the [REST API](https://api.plot.ly/v2) provided by the platform.

A subset of the charts of this platform where sampled in the [VizML](https://vizml.media.mit.edu/) study and it was called the *Plotly Corpus* with 120k tables, a later study [KG4Vis](https://kg4vis.github.io) fallowed the same sampling strategy and constructed another corpus of 80k tables.

The charts obtained are described by 4 objects:

- `fid`: the identifier of the chart.

* `table_data`: contains the data columns used in each one of the traces of the graph.
* `chart_data`: associates each data column with its graphical configurations such as `trace_type` and `axis`.
* `layout`: contains advanced graphical configurations for the hole chart such as `color`, `margins`, `ranges`,  etc...

This last 3 elements are encoded in [JSON](https://www.json.org/) format. 

## Motivation

The implementation of this module is based on the implementation provided by [VizML](https://vizml.media.mit.edu/), however we modified the procedure of feature extraction to address certain considerations that could impact the results of the studies that use the extracted features by this module.

## Procedure

#### Column extraction

We start by extracting the columns data and encodings from the charts. This is performed with the script `extract_columns.py`.

```bash
python extract_columns.py -i input_file_path -o output_file_path -v
```

The process implemented in this script is:

1. Data columns are extracted from the `table_data` structure and processed in the fallowing way:

   - The data type of the column is inferred by attempting multiple type castings over a small random sample of the data, when the number of errors occurred during type casting is below a certain threshold the data type is selected as valid. Available data types are `int`, `float`, `bool`, `datetime`, `string`. Notice that `string` is always a valid data type so the number of errors of the type `string` is computed as the maximum of successful castings for the other data types.

   - The data column is casted to the inferred type, however could be the case in which the inferred type is not correct, therefore the cast will result in an error, in case of errors the default type is `string` which is always a valid choice. `datetime` and `string` data types are encoded to `int` by obtaining the milliseconds in the `datetime` object and using a *bag of words* representation for the `string` data type. This is done to reduce storage requirements and improve the processing speed in the feature extraction by only having arrays of `float` and `int` types which are faster in the `numpy` library. 

   - Finally we impute the missing data of the column, for `quantitative` data *mean substitution* is used and for `categorical` data *mode substitution* is used. By performing data imputation in this stage we aim to preserve the heterogeneity of the corpus. The previous study performed the data imputation process after the feature extraction was performed, this could lead to an scenario where we had a column like:

     ```python
     c = [ 1.5, 2, 3.5, 4, NaN]
     ```

     Since there exist a `NaN` (`Not a Number`) value the feature extraction process will return an error for several features. For example the mean computed for this array will be `None` and then imputed by the mean of the means of all the columns in the corpus, so the mean associated to the column could be really far from the actual mean.

     Using our approach we impute in this stage and we get the column:

     ```python
     c = [1.5, 2, 3.5, 4, 2.75] # mean = 2.75
     ```

     This is not a perfect solution to the problem of missing data, since it is not well suited for a large number of missing values or multivariate analysis, but it ensures that the computed feature will be consistent with the observed data.

   The methods required for this step are implemented in the file `data_types.py` 

   - `detect_dtype` performs the type inference.
   - `cast_dtype` performs the hard cast to the inferred type.
   - `fill_dtype` performs the data imputation according to the rules for the specific data type.

2. The graphical configurations are extracted from the `chart_data` structure and are associated to each data column.

3. Each chart is now represented by a list of its data columns where each data column is described by the fallowing attributes:

   - `FID` is the identifier of the chart.
   - `FIELD_ID` is the identifier of the column, unique in the entire corpus.
   - `TRACE_TYPE` is the type of trace (i.e. `bar`, `scatter`, `line`, etc...)
   - `IS_XSRC` indicates that the column is on the `x` axis.
   - `IS_YSRC` indicates that the column is on the `y` axis.
   - `IS_ONLY_XSRC` indicates that the column is the only one in the `x` axis.
   - `IS_ONLY_YSRC` indicates that the column is the only one in the `y` axis.
   - `DATA` the raw data of the column.

#### Chart output extraction and Data cleaning 

After the columns are extracted we obtain the outputs of the charts, this are properties as: 

- `TRACE_TYPE` : The type of the chart (i.e `bar`, `line`, `scatter`, etc...) assuming that all traces  are of the same type.
- `N_XSRC` : The number of columns in the `x` axis.
- `N_YSRC`: The number of columns in the `y` axis.
- `LENGTH` : The length of the columns.

This can be done with the file `extract_tables_outputs.py`

```bash
python extract_tables_outputs.py -i input_file_path -o output_file_path
```

The `input_file_path` must be the path to the file of extracted columns in the previous step.

Since we need to iterate over all the corpus we perform data cleaning at the same time so the extracted columns are checked for errors:

- Check the extraction of basic information like `FID` and `FIELD_ID`

- Check that all columns of a chart have the same trace type.

- Check that every column is on one and only one axis.

- Check that the columns have at least 2 elements.

- Check that all the columns have the same length.

If a column has errors then the hole chart is compromised therefore all the columns of that chart are dropped as well.

After this process is concluded we obtain a file with all the tables that fulfill our constraints, having this tables we discard those columns from the corpus that does not belong to any of those charts with the script `clean_columns.py`

#### Compute features

Single column features and pairwise column features are computed from the columns, the name features proposed in [VizML](https://vizml.media.mit.edu/) where dropped and as an addition the statistical features that were previously exclusive for quantitative variables now are computed for categorical variables using the histogram of the categorical data as the input vector.

```bash
python extract_features.py -i input_file_name -os soutput_file_name -op poutput_file_name
```

The `os` argument refers to the output file path for single column features and the `op` argument refers to the output file for pairwise column features, also you can drop one of the output arguments in case you only want to compute a specific type of features.

```bash
python extract_features.py -i input_file_name -os soutput_file_name
```

After the features are extracted we compute aggregation functions over the extracted features grouping by the chart id.

```bash
python aggregate_features.py -i input_file_name -o output_file_name
```

#### Example

```bash
python extract_columns.py -i ../data/raw_charts.tsv -o ../data/raw_columns.tsv -v
python extract_tables_outputs.py  -i ../data/raw_columns.tsv -o ../data/tables_output.csv
python clean_columns.py -t ../data/tables_output.csv -c ../data/raw_columns.tsv -o ../data/cleaned_raw_columns.tsv
python extract_features.py -i ../data/cleaned_raw_columns.tsv -os ../features/single_column_features.csv -op ../features/pairwise_column_features.csv
python aggregate_features.py -i ../features/single_column_features.csv -o ../features/aggregated_single_column_features.csv -s
python aggregate_features.py -i ../features/pairwise_column_features.csv -o ../features/aggregated_pairwise_column_features.csv -s
```

