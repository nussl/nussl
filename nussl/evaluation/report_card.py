import pandas as pd
import json
import termtables
import numpy as np
import os
import textwrap

def aggregate_score_files(json_files, aggregator=np.nanmedian):
    """
    Takes a list of json files output by an Evaluation method in nussl
    and aggregates all the metrics into a Pandas dataframe. Sample
    output:

    .. code-block:: none

                                SDR        SIR        SAR
        drums  oracle0.json   9.086025  15.025801  10.362709
               random0.json  -6.539877  -6.087538   3.508338
               oracle1.json   9.591432  14.335700  11.365882
               random1.json  -1.358840  -0.993666   9.577297
        bass   oracle0.json   7.936720  12.843092   9.631929
               random0.json  -4.190299  -3.730649   5.802003
               oracle1.json   8.581090  12.513445  10.831370
               random1.json   0.365171   0.697621  11.693103
        other  oracle0.json   2.024207   6.133359   4.158805
               random0.json  -9.857085  -9.481909   0.965199
               oracle1.json   3.961383   6.861785   7.085745
               random1.json  -4.042277  -3.707997   7.260934
        vocals oracle0.json  12.169686  16.650161  14.085037
               random0.json  -2.440166  -1.884026   6.760966
               oracle1.json  12.409913  16.248470  14.725983
               random1.json   1.609577   1.958037  12.738970
    
    Args:
        json_files (list): List of JSON files that will be parsed for metrics.
        aggregator ([type], optional): How to aggregate results within a single
          track. Defaults to np.median.
    
    Returns:
        pd.DataFrame: Pandas dataframe containing the aggregated metrics.
    """
    metrics = {}
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        json_key = os.path.basename(json_file)
        for name in data:
            if name not in ['combination', 'permutation']:
                if name not in metrics:
                    metrics[name] = {}
                if json_key not in metrics[name]:
                    metrics[name][json_key] = {}
                for key in data[name]:
                    _data = aggregator(data[name][key])
                    metrics[name][json_key][key] = _data
    
    df = pd.concat({
        k: pd.DataFrame(v).T for k, v in metrics.items()
    }, axis=0, names=['source', 'file'])
    df.reset_index(inplace=True) 
    
    return df

def _get_mean_and_std(df):
    """
    Gets the mean and standard deviation of each metric in the pandas
    DataFrame and returns it as a list of strings.
    """
    excluded_columns = ['source', 'file']

    metrics = [x for x in list(df.columns) if x not in excluded_columns]
    metrics.insert(0, '#')

    means = [f'{m:5.1f}' for m in np.array(df.mean()).T]
    stds = [f'{s:4.1f}' for s in np.array(df.std()).T]
    data = [f'{m} +/- {s}' for m, s in zip(means, stds)]
    data.insert(0, df.shape[0])

    return metrics, data

def _get_medians(df):
    """
    Gets the median of each metric in the pandas
    DataFrame and returns it as a list of strings.
    """
    excluded_columns = ['source', 'file']

    metrics = [x for x in list(df.columns) if x not in excluded_columns]
    metrics.insert(0, '#')

    data = [f'{m:5.1f}' for m in np.array(df.median()).T]
    data.insert(0, df.shape[0])
    return metrics, data

def _format_title(title, length, marker=" "):
    pad = (length - len(title)) // 2
    pad = ''.join([marker for _ in range(pad)])
    border = pad + title + pad
    if len(title) % 2:
        border = border + marker
    return border

def _get_report_card(df, func, report_each_source=True):
    """
    Gets a report card for a DataFrame using a specific function.
    """
    labels, data = func(df)

    data.insert(0, 'OVERALL')
    data = [data]

    if report_each_source:
        for name in np.unique(df['source']):
            _df = df[df['source'] == name]
            _, _data = func(_df)
            _data.insert(0, name.upper())
            data.append(_data)
    
    # transposing data so each column has the source and its metrics
    data = list(map(list, zip(*data)))
    header = data.pop(0)
    header.insert(0, 'METRIC')
    for i in range(1, len(header)):
        header[i] = _format_title(header[i], 16)
    for l, d in zip(labels, data):
        d.insert(0, l)
        
    alignment = ["c" for _ in header]
    alignment[0] = "l"
    alignment = ''.join(alignment)
    
    report_card = termtables.to_string(
        data, header=header, padding=(0, 1), alignment=alignment)

    return report_card

def report_card(df, notes=None, report_each_source=True):
    """
    Given a Pandas dataframe, usually the output of ``aggregate_score_files``,
    returns a string that looks like this::

    .. code-block:: none
                                                      
                            MEAN +/- STD OF METRICS                      
                                                                        
        ┌─────────┬─────────────────┬─────────────────┬──────────────────┐
        │         │       SAR       │       SDR       │       SIR        │
        ╞═════════╪═════════════════╪═════════════════╪══════════════════╡
        │ OVERALL │ 7.630 +/- 2.769 │ 5.298 +/- 2.955 │ 10.322 +/- 4.714 │
        ├─────────┼─────────────────┼─────────────────┼──────────────────┤
        │  BASS   │ 7.060 +/- 2.501 │ 4.665 +/- 3.351 │ 9.323 +/- 5.132  │
        ├─────────┼─────────────────┼─────────────────┼──────────────────┤
        │  DRUMS  │ 8.571 +/- 2.026 │ 6.487 +/- 2.472 │ 11.448 +/- 4.018 │
        ├─────────┼─────────────────┼─────────────────┼──────────────────┤
        │  OTHER  │ 7.296 +/- 3.439 │ 3.906 +/- 2.041 │ 7.859 +/- 3.606  │
        ├─────────┼─────────────────┼─────────────────┼──────────────────┤
        │ VOCALS  │ 7.594 +/- 2.756 │ 6.133 +/- 3.067 │ 12.657 +/- 4.549 │
        └─────────┴─────────────────┴─────────────────┴──────────────────┘
                                                                        
                                MEDIAN OF METRICS                         
                                                                        
        ┌─────────┬─────────────────┬─────────────────┬──────────────────┐
        │         │       SAR       │       SDR       │       SIR        │
        ╞═════════╪═════════════════╪═════════════════╪══════════════════╡
        │ OVERALL │      7.437      │      5.201      │      10.274      │
        ├─────────┼─────────────────┼─────────────────┼──────────────────┤
        │  BASS   │      6.940      │      4.448      │      9.450       │
        ├─────────┼─────────────────┼─────────────────┼──────────────────┤
        │  DRUMS  │      8.255      │      6.409      │      11.144      │
        ├─────────┼─────────────────┼─────────────────┼──────────────────┤
        │  OTHER  │      6.819      │      4.600      │      8.634       │
        ├─────────┼─────────────────┼─────────────────┼──────────────────┤
        │ VOCALS  │      7.934      │      6.657      │      14.270      │
        └─────────┴─────────────────┴─────────────────┴──────────────────┘
                                                                        
                                    NOTES                               
                                                                        
        Uses scale-dependent BSSEval metrics. Evaluated on MUSDB18 at
        44100 Hz sample rate.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the metrics computed during
          evaluation.
        notes (str, optional): Any additional notes you want to be printed at the
          bottom of the report card. Defaults to None.
        report_each_source (bool, optional): Whether or not to report the metrics
          for each individual source type. Defaults to True.
    
    Returns:
        str: A report card for your experiment.
    """
    mean_report_card = _get_report_card(
        df, _get_mean_and_std, report_each_source=report_each_source)
    median_report_card = _get_report_card(
        df, _get_medians, report_each_source=report_each_source)

    line_break = mean_report_card.index('\n')

    report_card = (
        f"{_format_title('', line_break)}\n"
        f"{_format_title(' MEAN +/- STD OF METRICS ', line_break)}\n"
        f"{_format_title('', line_break)}\n"
        f"{mean_report_card}\n"
        f"{_format_title('', line_break)}\n"
        f"{_format_title(' MEDIAN OF METRICS ', line_break)}\n"
        f"{_format_title('', line_break)}\n"
        f"{median_report_card}\n"
    )
    
    if notes is not None:
        notes = '\n'.join(textwrap.wrap(notes, line_break))
        report_card += (
            f"{_format_title('', line_break)}\n"
            f"{_format_title(' NOTES ', line_break)}\n"
            f"{_format_title('', line_break)}\n"
            f"{notes}"
        )
    return report_card
