import pandas as pd
import json
import termtables
import numpy as np
import os
import textwrap
import copy


def truncate(values, decs=2):
    return np.trunc(values*10**decs)/(10**decs)


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
            if name not in ['combination', 'permutation', 'metadata']:
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


def _get_mean_and_std(df, decs=2):
    """
    Gets the mean and standard deviation of each metric in the pandas
    DataFrame and returns it as a list of strings.
    """
    excluded_columns = ['source', 'file']

    metrics = [x for x in list(df.columns) if x not in excluded_columns]
    metrics.insert(0, '#')

    means = [
        f'{truncate(m, decs=decs):{4+decs}.{decs}f}' 
        for m in np.array(df.mean()).T
    ]
    stds = [
        f'{truncate(s, decs=decs):{3+decs}.{decs}f}' 
        for s in np.array(df.std()).T
    ]
    data = [f'{m} +/- {s}' for m, s in zip(means, stds)]
    data.insert(0, df.shape[0])

    return metrics, data


def _get_medians(df, decs=2):
    """
    Gets the median of each metric in the pandas
    DataFrame and returns it as a list of strings.
    """
    excluded_columns = ['source', 'file']

    metrics = [x for x in list(df.columns) if x not in excluded_columns]
    metrics.insert(0, '#')

    data = [
        f'{truncate(m, decs=decs):{4+decs}.{decs}f}' 
        for m in np.array(df.median()).T
    ]
    data.insert(0, df.shape[0])
    return metrics, data


def _format_title(title, length, marker=" "):
    pad = (length - len(title)) // 2
    pad = ''.join([marker for _ in range(pad)])
    border = pad + title + pad
    if len(title) % 2:
        border = border + marker
    return border


def _get_report_card(df, func, report_each_source=True, decs=2):
    """
    Gets a report card for a DataFrame using a specific function.
    """
    labels, data = func(df, decs=decs)

    data.insert(0, 'OVERALL')
    data = [data]

    if report_each_source:
        for name in np.unique(df['source']):
            _df = df[df['source'] == name]
            _, _data = func(_df, decs=decs)
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


def report_card(df, notes=None, report_each_source=True, decimals=2):
    """
    Given a Pandas dataframe, usually the output of ``aggregate_score_files``,
    returns a string that looks like this::

    .. code-block:: none
                                                      
                                                                           
                            MEAN +/- STD OF METRICS                       
                                                                            
        ┌─────────┬──────────────────┬──────────────────┬──────────────────┐
        │ METRIC  │     OVERALL      │        S1        │        S2        │
        ╞═════════╪══════════════════╪══════════════════╪══════════════════╡
        │ #       │       6000       │       3000       │       3000       │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SI-SDR  │   11.2 +/-  3.8  │   12.5 +/-  3.5  │    9.8 +/-  3.5  │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SI-SIR  │   22.7 +/-  5.4  │   22.9 +/-  5.0  │   22.6 +/-  5.7  │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SI-SAR  │   11.6 +/-  3.6  │   13.0 +/-  3.3  │   10.1 +/-  3.3  │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SD-SDR  │   10.8 +/-  4.0  │   12.2 +/-  3.8  │    9.3 +/-  3.7  │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SNR     │   11.6 +/-  3.3  │   12.9 +/-  3.1  │   10.3 +/-  3.0  │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SRR     │   22.8 +/-  6.5  │   25.5 +/-  6.3  │   20.0 +/-  5.6  │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SI-SDRi │   11.2 +/-  3.7  │   10.0 +/-  3.4  │   12.3 +/-  3.6  │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SD-SDRi │   10.8 +/-  3.8  │    9.7 +/-  3.6  │   11.8 +/-  3.7  │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SNRi    │   11.6 +/-  3.3  │   10.3 +/-  3.0  │   12.9 +/-  3.1  │
        └─────────┴──────────────────┴──────────────────┴──────────────────┘
                                                                            
                                MEDIAN OF METRICS                          
                                                                            
        ┌─────────┬──────────────────┬──────────────────┬──────────────────┐
        │ METRIC  │     OVERALL      │        S1        │        S2        │
        ╞═════════╪══════════════════╪══════════════════╪══════════════════╡
        │ #       │       6000       │       3000       │       3000       │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SI-SDR  │       11.6       │       13.1       │       10.4       │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SI-SIR  │       23.6       │       23.6       │       23.6       │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SI-SAR  │       12.0       │       13.5       │       10.6       │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SD-SDR  │       11.3       │       12.9       │       10.0       │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SNR     │       11.9       │       13.3       │       10.7       │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SRR     │       23.4       │       26.5       │       20.6       │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SI-SDRi │       11.6       │       10.5       │       12.9       │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SD-SDRi │       11.3       │       10.3       │       12.5       │
        ├─────────┼──────────────────┼──────────────────┼──────────────────┤
        │ SNRi    │       11.9       │       10.7       │       13.3       │
        └─────────┴──────────────────┴──────────────────┴──────────────────┘

                                                                        
                                    NOTES                               
                                                                        
        Uses scale-invariant BSSEval metrics. Evaluated on WSJ0-2Mix at
        8000 Hz sample rate.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the metrics computed during
          evaluation.
        notes (str, optional): Any additional notes you want to be printed at the
          bottom of the report card. Defaults to None.
        report_each_source (bool, optional): Whether or not to report the metrics
          for each individual source type. Defaults to True.
        decimals (int): Number of decimal places to display.
    
    Returns:
        str: A report card for your experiment.
    """
    mean_report_card = _get_report_card(
        df, _get_mean_and_std, report_each_source=report_each_source, decs=decimals)
    median_report_card = _get_report_card(
        df, _get_medians, report_each_source=report_each_source, decs=decimals)

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


def associate_metrics(separation_model, df, test_dataset):
    """
    For a given pandas dataframe (the output of ``aggregate_score_files``), this
    function will associate the high level summary statistics with a model.

    Args:
        separation_model (SeparationModel): A separation object that will have the metrics
            associated with it.
        df (pandas.DataFrame): DataFrame containing the metrics computed during
            evaluation.
        test_dataset (BaseDataset): A dataset object used for the evaluation of the
            metrics.

    Returns:
        (SeparationBase)
    """
    excluded_columns = ['source', 'file']
    metrics = [x for x in list(df.columns) if x not in excluded_columns]
    results = {
        m: {
            'mean': np.mean(df[m]),
            'median': np.median(df[m]),
            'std': np.std(df[m])
        }
        for m in metrics
    }
    separation_model.metadata['evaluation'] = results
    separation_model.metadata['test_dataset'] = copy.deepcopy(test_dataset.metadata)
    return separation_model