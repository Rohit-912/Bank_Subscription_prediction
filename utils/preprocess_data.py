import numpy as np
import pandas as pd

def preprocess_data(df, min_balance=None):
    """
    Preprocess raw banking data by replicating steps from the preprocessing notebook.
    
    Args:
        df (pd.DataFrame): Raw input data
        min_balance (float, optional): Min balance from training data. 
            If None, calculates from current df.
    
    Returns:
        pd.DataFrame: Preprocessed data ready for modeling
    """
    # Create a copy to avoid modifying original data
    df = df.copy()
    
    # 1. Merge rare job categories
    rare_jobs = ['self-employed', 'entrepreneur', 'housemaid', 'unemployed', 'student', 'unknown']
    df['job'] = df['job'].replace(rare_jobs, 'other')
    
    # 2. Handle balance transformation
    if min_balance is None:
        min_balance = df['balance'].min()
    df['balance_shifted'] = df['balance'] - min_balance + 1
    df['log_balance'] = np.log(df['balance_shifted'])
    
    # 3. Map education to numerical values
    education_map = {'unknown':0, 'primary':1, 'secondary':2, 'tertiary':3}
    df['education'] = df['education'].map(education_map).fillna(0)  # Handle unseen categories
    
    # 4. Create safe features (no leakage)
    # Contact sensitivity
    df['contact_sensitivity'] = np.where(
        df['pdays'] == -1,
        0,
        df['previous'] / (df['pdays'] + 1e-6)
    )
    
    # Previous campaign engagement
    df['prev_campaign_engaged'] = (df['poutcome'] == 'success').astype(int)
    
    # Responsiveness
    df['responsiveness'] = np.select(
        [
            df['previous'] == 0,
            (df['previous'] > 0) & (df['poutcome'] == 'success'),
            (df['previous'] > 0) & (df['poutcome'] != 'success')
        ],
        ['new', 'responsive', 'unresponsive'],
        default='unknown'
    )
    
    # 5. Duration log transformation
    df['duration_log'] = np.log(df['duration'] + 1)
    
    # 6. Drop unnecessary columns
    cols_to_drop = ['id', 'day', 'month', 'balance', 'balance_shifted', 'previous', 'default']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    return df