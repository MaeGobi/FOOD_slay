import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_clean_subsamples(
    file_path: str, 
    target_size: int = 100000, 
    missing_threshold: float = 0.85, 
    completeness_gate: float = 0.30, 
    seed: int = 42
):
    """
    Reads the massive Open Food Facts dataset, performs feature selection, 
    decontaminates target leaks and URLs, filters empty rows, and outputs 
    two highly representative, deduplicated 100k subsamples (Labeled and Unlabeled).
    
    Parameters:
    -----------
    file_path : str
        The full system path to the raw dataset.csv.
    target_size : int
        The number of rows for the final subsamples (default: 100,000).
    missing_threshold : float
        Drop columns with missing values greater than this rate (default: 85%).
    completeness_gate : float
        Drop rows that have less than this ratio of features filled (default: 30%).
    seed : int
        Seeding value for reproducibility (default: 42).
    """
    output_dir = os.path.dirname(file_path)
    
    # -------------------------------------------------------------
    # STEP 1: DYNAMIC COLUMN SELECTION & DECONTAMINATION
    # -------------------------------------------------------------
    print("--- STEP 1: ANALYZING COLUMNS AND DECONTAMINATING TARGET LEAKS ---")
    df_sample = pd.read_csv(file_path, sep="\t", nrows=50000, low_memory=False)
    
    # Identify dynamic completeness
    missing_rates = df_sample.isnull().mean()
    auto_kept_columns = missing_rates[missing_rates < missing_threshold].index.tolist()
    
    # Explicitly define known redundant/collinear columns to drop
    collinear_drops = [
        'energy-kj_100g', 'energy-kcal_100g', 
        'salt_100g', 'added-salt_100g', 'added-sugars_100g'
    ]
    
    # Explicitly define direct target leaks & proxies to drop
    target_leaks = [
        'nutriscore_grade', 'environmental_score_grade', 'nutrient_levels_tags'
    ]
    
    # Filter columns dynamically: remove collinear, target leaks, and ANY column containing 'url'
    columns_to_keep = []
    for col in auto_kept_columns:
        is_collinear = col in collinear_drops
        is_leak = col in target_leaks
        is_url = 'url' in col.lower()
        
        if not (is_collinear or is_leak or is_url):
            columns_to_keep.append(col)
            
    # Force inclusion of crucial metadata and target columns (if available in raw file)
    crucial_cols = ['nutriscore_score', 'categories_tags', 'brands_tags', 'countries_tags']
    for col in crucial_cols:
        if col not in columns_to_keep and col in df_sample.columns:
            columns_to_keep.append(col)
            
    print(f"Selected {len(columns_to_keep)} clean, leak-free, non-URL columns out of {df_sample.shape[1]}.")
    
    # -------------------------------------------------------------
    # STEP 2: STREAMING & ROW QUALITY GATING
    # -------------------------------------------------------------
    print("\n--- STEP 2: STREAMING DATA & APPLYING COMPLETENESS FILTER ---")
    chunk_size = 50000
    labeled_chunks = []
    unlabeled_chunks = []
    total_rows_scanned = 0
    
    # Feature columns list (all selected columns except our target label)
    feature_cols = [c for c in columns_to_keep if c != 'nutriscore_score']
    min_features_required = int(len(feature_cols) * completeness_gate)
    
    for chunk in pd.read_csv(file_path, sep="\t", usecols=columns_to_keep, chunksize=chunk_size, low_memory=False):
        total_rows_scanned += len(chunk)
        
        # Row quality validation based on selected features
        chunk['feature_completeness'] = chunk[feature_cols].notna().sum(axis=1)
        high_quality_chunk = chunk[chunk['feature_completeness'] >= min_features_required].copy()
        
        # Split on target presence
        has_target = high_quality_chunk['nutriscore_score'].notna()
        
        clean_labeled = high_quality_chunk[has_target].drop(columns=['feature_completeness'])
        clean_unlabeled = high_quality_chunk[~has_target].drop(columns=['feature_completeness'])
        
        labeled_chunks.append(clean_labeled)
        unlabeled_chunks.append(clean_unlabeled)
        
    df_labeled = pd.concat(labeled_chunks, axis=0).reset_index(drop=True)
    df_unlabeled = pd.concat(unlabeled_chunks, axis=0).reset_index(drop=True)
    
    print(f"Total raw rows scanned: {total_rows_scanned:,}")
    print(f"Passes quality gate (Labeled): {len(df_labeled):,} rows")
    print(f"Passes quality gate (Unlabeled): {len(df_unlabeled):,} rows")
    
    # -------------------------------------------------------------
    # STEP 3: TARGET CLEANING & TYPE CASTING
    # -------------------------------------------------------------
    print("\n--- STEP 3: CASTING TARGET & REMOVING CORRUPT LABELS ---")
    df_labeled['nutriscore_score'] = pd.to_numeric(df_labeled['nutriscore_score'], errors='coerce')
    df_labeled = df_labeled.dropna(subset=['nutriscore_score']).copy()
    print(f"Valid, numeric labeled rows remaining: {len(df_labeled):,}")
    
    # -------------------------------------------------------------
    # STEP 4: SEEDED REPRODUCIBLE SUBSAMPLING
    # -------------------------------------------------------------
    print("\n--- STEP 4: GENERATING REPRESENTATIVE SUBSAMPLES ---")
    
    # Stratified sample for Labeled (using seeded pseudo-random bins)
    rng = np.random.default_rng(seed=seed)
    df_labeled['random_bin'] = rng.integers(low=0, high=10, size=len(df_labeled))
    
    df_labeled_sub, _ = train_test_split(
        df_labeled,
        train_size=target_size,
        stratify=df_labeled['random_bin'],
        random_state=seed
    )
    df_labeled_sub = df_labeled_sub.drop(columns=['random_bin'])
    df_labeled = df_labeled.drop(columns=['random_bin'])
    
    # Standard random sample for Unlabeled
    df_unlabeled_sub = df_unlabeled.sample(n=target_size, random_state=seed)
    
    # -------------------------------------------------------------
    # STEP 5: DEDUPLICATION & INTEGRITY VALIDATION
    # -------------------------------------------------------------
    print("\n--- STEP 5: RESOLVING DOUBLONS & SAVING FILES ---")
    
    # Remove any duplicate rows
    df_labeled_sub = df_labeled_sub.drop_duplicates().reset_index(drop=True)
    df_unlabeled_sub = df_unlabeled_sub.drop_duplicates().reset_index(drop=True)
    
    # Save the files
    labeled_out_path = os.path.join(output_dir, f'labeled_subsample_{target_size // 1000}k.csv')
    unlabeled_out_path = os.path.join(output_dir, f'unlabeled_subsample_{target_size // 1000}k.csv')
    
    df_labeled_sub.to_csv(labeled_out_path, index=False)
    df_unlabeled_sub.to_csv(unlabeled_out_path, index=False)
    
    print("Verification Statistics:")
    print(f" * Final Labeled Subsample: {df_labeled_sub.shape[0]:,} unique rows | {df_labeled_sub.shape[1]} columns")
    print(f" * Final Unlabeled Subsample: {df_unlabeled_sub.shape[0]:,} unique rows | {df_unlabeled_sub.shape[1]} columns")
    print(f"\n✅ Files successfully exported to:")
    print(f"   -> {labeled_out_path}")
    print(f"   -> {unlabeled_out_path}")

# Run the pipeline
if __name__ == "__main__":
    raw_dataset_path = '/Users/giorgiocannavacciuolo/DATA/DESU/Projects/food_ml/dataset.csv'
    generate_clean_subsamples(raw_dataset_path)