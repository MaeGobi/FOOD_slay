import pandas as pd
import numpy as np

def remove_physical_outliers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Cleans physical impossibilities from Open Food Facts data.
    Ensures weights are 0-100g, energy is within physical limits, 
    and cumulative macros do not exceed 100g.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the food product data.
    verbose : bool
        If True, prints detailed execution metrics and dropped row counts.
        
    Returns:
    --------
    pd.DataFrame
        The cleaned, physically consistent DataFrame.
    """
    # Work on a copy to prevent SettingWithCopyWarning
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # 1. Identify weight vs. energy columns dynamically
    all_100g_cols = [col for col in df_clean.columns if col.endswith('_100g')]
    weight_cols = [col for col in all_100g_cols if 'energy' not in col.lower()]
    energy_cols = [col for col in all_100g_cols if 'energy' in col.lower()]
    
    # 2. Rule 1: Weights must strictly stay between 0g and 100g
    before_weight = len(df_clean)
    for col in weight_cols:
        df_clean = df_clean[(df_clean[col].isna()) | ((df_clean[col] >= 0) & (df_clean[col] <= 100.0))]
    weight_dropped = before_weight - len(df_clean)
    
    # 3. Rule 2: Energy caps based on pure fat density limits
    before_energy = len(df_clean)
    for col in energy_cols:
        if 'kcal' in col.lower():
            # Theoretical max limit: 900 kcal / 100g
            df_clean = df_clean[(df_clean[col].isna()) | ((df_clean[col] >= 0) & (df_clean[col] <= 900.0))]
        elif 'kj' in col.lower() or col == 'energy_100g': 
            # Theoretical max limit: 3700 kJ / 100g
            df_clean = df_clean[(df_clean[col].isna()) | ((df_clean[col] >= 0) & (df_clean[col] <= 3700.0))]
    energy_dropped = before_energy - len(df_clean)
    
    # 4. Rule 3: Cumulative Macronutrient sum check (fat + carbs + protein + salt)
    macro_cols = ['fat_100g', 'carbohydrates_100g', 'proteins_100g', 'salt_100g']
    existing_macros = [col for col in macro_cols if col in df_clean.columns]
    
    macro_dropped = 0
    if len(existing_macros) > 1:
        before_macro = len(df_clean)
        # Sum macros, filling NaNs with 0 strictly for the calculation
        macro_sum = df_clean[existing_macros].fillna(0).sum(axis=1)
        df_clean = df_clean[macro_sum <= 101.0] # 1g allowance for rounded values
        macro_dropped = before_macro - len(df_clean)
        
    # Logging
    if verbose:
        total_dropped = initial_rows - len(df_clean)
        print("=== PHYSICAL OUTLIER DETECTION REPORT ===")
        print(f" * Weight violations removed (<0g or >100g):      {weight_dropped:,} rows")
        print(f" * Thermodynamic energy limit violations removed:  {energy_dropped:,} rows")
        print(f" * Macronutrient mass budget violations (>100g):   {macro_dropped:,} rows")
        print(f"------------------------------------------")
        print(f" * Initial Row Count: {initial_rows:,}")
        print(f" * Cleaned Row Count: {len(df_clean):,} (Dropped {total_dropped:,} corrupted rows)")
        print("==========================================\n")
        
    return df_clean