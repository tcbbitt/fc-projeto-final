import numpy as np
import autorootcwd  
import pandas as pd
from sklearn.model_selection import train_test_split

def get_ottawa2023_splits(df, train_size=0.6, random_state=42):
    """
    Split a DataFrame into training, validation, and test sets.
    
    This function splits the input DataFrame into three subsets: training, 
    validation, and test. The training set size is determined by the 
    `train_size` parameter, while the validation and test sets are split 
    equally from the remaining data.
    
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

    train_df, temp_df = train_test_split(df, train_size = train_size, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state)
    
    return train_df, val_df, test_df