import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_ottawa2023_splits(df, train_size=0.6, random_state=42):
    """
    Split a DataFrame into training, validation, and test sets ensuring that 
    each unique bearing_id appears in only one set.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be split.
        
    train_size : float, optional
        The proportion of the data to include in the training set. 
        Default is 0.6 (60%).
        
    random_state : int, optional
        The seed used by the random number generator for reproducibility. 
        Default is 42.
        
    Returns
    -------
    train_df : pandas.DataFrame
        The training subset of the input DataFrame.
        
    val_df : pandas.DataFrame
        The validation subset of the input DataFrame.
        
    test_df : pandas.DataFrame
        The test subset of the input DataFrame.
    """
    unique_ids = df['bearing_id'].unique()
    train_ids, temp_ids = train_test_split(unique_ids, train_size=train_size, random_state=random_state)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=random_state)

    train_df = df[df['bearing_id'].isin(train_ids)].copy()
    val_df = df[df['bearing_id'].isin(val_ids)].copy()
    test_df = df[df['bearing_id'].isin(test_ids)].copy()

    return train_df, val_df, test_df