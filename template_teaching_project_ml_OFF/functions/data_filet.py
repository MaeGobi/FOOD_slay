import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_and_save_subsamples():
    path = '/Users/giorgiocannavacciuolo/DATA/DESU/Projects/food_ml/dataset.csv'
    output_dir = os.path.dirname(path)
    
    # 1. Define collinear/redundant variables to ignore
    explicit_drops = [
        'energy-kj_100g',    
        'energy-kcal_100g',  
        'salt_100g',         
        'added-salt_100g',   
        'added-sugars_100g'  
    ]

    print("Analyzing column completeness in a data-driven way...")
    df_sample = pd.read_csv(path, sep="\t", nrows=50000, low_memory=False)

    # Keep columns that have LESS than 85% missing values
    missing_percentage = df_sample.isnull().mean()
    missing_threshold = 0.85
    auto_kept_columns = missing_percentage[missing_percentage < missing_threshold].index.tolist()

    # Filter out our explicit drops and keep our target column
    columns_to_keep = [col for col in auto_kept_columns if col not in explicit_drops]

    # Ensure crucial metadata and target are kept
    crucial_cols = ['nutriscore_score', 'categories_tags', 'brands_tags', 'countries_tags']
    for col in crucial_cols:
        if col not in columns_to_keep and col in df_sample.columns:
            columns_to_keep.append(col)

    print(f"Broad Filter completed. Selected {len(columns_to_keep)} columns.")

    # 2. Stream and apply the Quality Gate
    chunk_size = 50000
    labeled_chunks = []
    unlabeled_chunks = []
    total_rows_scanned = 0

    print("\nStreaming dataset and applying 30% completeness quality gate...")
    for chunk in pd.read_csv(path, sep="\t", usecols=columns_to_keep, chunksize=chunk_size, low_memory=False):
        total_rows_scanned += len(chunk)
        
        # Calculate completeness based only on feature columns
        feature_cols = [c for c in columns_to_keep if c != 'nutriscore_score']
        chunk['feature_completeness'] = chunk[feature_cols].notna().sum(axis=1)
        
        # Keep rows with >= 30% of chosen columns populated
        min_features_required = int(len(feature_cols) * 0.30)
        high_quality_chunk = chunk[chunk['feature_completeness'] >= min_features_required].copy()
        
        # Split into Labeled and Unlabeled
        has_target = high_quality_chunk['nutriscore_score'].notna()
        
        clean_labeled = high_quality_chunk[has_target].drop(columns=['feature_completeness'])
        clean_unlabeled = high_quality_chunk[~has_target].drop(columns=['feature_completeness'])
        
        labeled_chunks.append(clean_labeled)
        unlabeled_chunks.append(clean_unlabeled)

    df_labeled = pd.concat(labeled_chunks, axis=0).reset_index(drop=True)
    df_unlabeled = pd.concat(unlabeled_chunks, axis=0).reset_index(drop=True)

    print(f"\nTotal raw rows scanned: {total_rows_scanned:,}")
    print(f"High-quality Labeled Pool: {len(df_labeled):,} rows")
    print(f"High-quality Unlabeled Pool: {len(df_unlabeled):,} rows")

    # 3. Clean and convert target to numeric
    print("\nCasting target column to numeric and dropping corrupt strings...")
    df_labeled['nutriscore_score'] = pd.to_numeric(df_labeled['nutriscore_score'], errors='coerce')
    df_labeled = df_labeled.dropna(subset=['nutriscore_score']).copy()

    # 4. Generate reproducible random bins for stratification
    print("Generating reproducible random bins for stratification...")
    rng = np.random.default_rng(seed=42)
    df_labeled['random_bin'] = rng.integers(low=0, high=10, size=len(df_labeled))

    # 5. Extract 100k Stratified Labeled Subsample
    print("Creating stratified 100k subsample of the Labeled Pool...")
    df_labeled_sub, _ = train_test_split(
        df_labeled,
        train_size=100000,
        stratify=df_labeled['random_bin'],
        random_state=42
    )
    df_labeled_sub = df_labeled_sub.drop(columns=['random_bin'])

    # 6. Extract 100k Unlabeled Subsample
    print("Creating 100k subsample of the Unlabeled Pool...")
    df_unlabeled_sub = df_unlabeled.sample(n=100000, random_state=42).reset_index(drop=True)

    # 7. Save to CSV
    labeled_out = os.path.join(output_dir, 'labeled_subsample_100k.csv')
    unlabeled_out = os.path.join(output_dir, 'unlabeled_subsample_100k.csv')
    
    print(f"\nSaving CSV files to {output_dir}...")
    df_labeled_sub.to_csv(labeled_out, index=False)
    df_unlabeled_sub.to_csv(unlabeled_out, index=False)
    
    print("--- SUCCESS ---")
    print(f"Saved: {labeled_out}")
    print(f"Saved: {unlabeled_out}")

if __name__ == "__main__":
    prepare_and_save_subsamples()