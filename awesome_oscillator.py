# awesome_oscillator.py

import pandas as pd

def calculate_ao(df: pd.DataFrame, short_window: int = 5, long_window: int = 34) -> pd.DataFrame:
    """
    Calculate the Awesome Oscillator (AO) for each column in a DataFrame of closing prices.
    If df has a 'Close' column and other OHLC columns, it'll just use df['Close'].
    Otherwise it treats every column as a close-price series.

    Returns a DataFrame of the same shape, where each column is the AO series.
    """
    # Determine which data to use as the price series
    if 'Close' in df.columns:
        price = df['Close']
    else:
        price = df.copy()

    # Compute SMAs
    sma_short = price.rolling(window=short_window).mean()
    sma_long  = price.rolling(window=long_window).mean()
    ao = sma_short - sma_long

    # If it came back as a Series, wrap into a DataFrame
    if isinstance(ao, pd.Series):
        ao = ao.to_frame(name='AO')

    return ao


def save_ao_to_csv(ao_df: pd.DataFrame, output_csv: str):
    """
    Saves the AO DataFrame to CSV with 'Date' as the first column.
    Assumes the index of ao_df is datetime-like.
    """
    # Round AO values to 4 decimal places
    ao_df = ao_df.round(4)
    
    out = ao_df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.reset_index().rename(columns={'index': 'Date'})
    out.to_csv(output_csv, index=False, date_format='%Y-%m-%d')
    print(f"Saved Awesome Oscillator to {output_csv}")
