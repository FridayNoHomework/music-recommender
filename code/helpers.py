import pandas as pd

def mean_df(df_list):
    """
    Calculate the mean DataFrame from a list of DataFrames.

    Parameters:
    - df_list: List of DataFrames to be merged and averaged.

    Returns:
    - DataFrame: A new DataFrame containing the mean values for each index across all input DataFrames.
    """
    merged_df = pd.concat(df_list)
    mean_df = merged_df.groupby(level=0).mean()
    return mean_df

def std_df(df_list):
    merged_df = pd.concat(df_list)
    std_df = merged_df.groupby(level=0).std()
    return std_df

def pick_from_dict(super_dict, keys):
    """
    Extracts a subset of a dictionary based on the specified keys.

    Parameters:
    - super_dict (dict): The dictionary from which to extract a subset.
    - keys (list): A list of keys to select from the super_dict.

    Returns:
    - dict or None: A dictionary containing the selected keys and their corresponding values.
                   Returns None if any key is not found in the super_dict.

    Example:
    >>> super_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    >>> selected_keys = ['a', 'c', 'e']
    >>> pick_from_dict(super_dict, selected_keys)
    {'a': 1, 'c': 3}
    """
    try:
        sub_dict = {key: super_dict[key] for key in keys}
        return sub_dict
    except KeyError as e:
        print(f"Error: Could not find the key {e} in the super dictionary")
        raise  # Re-raise the exception for better error handling

def textFeatures2list_series(dataset,cols_for_soup):
    """
    Merge multiple text feature columns into a single column, creating a list of text features for each sample.

    Parameters:
    - dataset (pd.DataFrame): The input DataFrame containing text feature columns.
    - cols_for_soup (list): A list of column names representing the text features to be merged.

    Returns:
    - pd.Series: A Pandas Series where each element is a list of text features for a sample.
    """  
    text_features = dataset[cols_for_soup].copy()
    for col in cols_for_soup:
        text_features.loc[:,col] = text_features.loc[:,col].str.replace('[^a-zA-Z]', '', regex=True).str.lower()
    text_features.loc[:,'text features'] = text_features.apply(lambda row: ' '.join(str(row[col]) for col in cols_for_soup), axis=1)
    return text_features.loc[:,'text features'].str.split()