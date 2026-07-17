import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def visualize_collinearity(df, numeric_cols=None, figsize=(12, 10)):
    """
    Affiche une matrice de corrélation (heatmap) pour repérer visuellement
    les variables numériques fortement corrélées entre elles.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matrice de corrélation des variables numériques")
    plt.tight_layout()
    plt.show()

    return corr_matrix


def select_relevant_columns(file_path, missing_threshold=0.85, sample_nrows=50000):
    """
    Analyse un échantillon du dataset pour déterminer quelles colonnes garder :
    exclut les colonnes trop vides, les colinéaires connues, les leaks de la
    cible, et les URLs.

    Parameters
    ----------
    file_path : str ou Path
        Chemin vers le dataset brut.
    missing_threshold : float
        Seuil au-delà duquel une colonne est jugée trop incomplète (défaut 85%).
    sample_nrows : int
        Nombre de lignes à lire pour l'analyse rapide des colonnes.

    Returns
    -------
    list[str]
        Liste des colonnes à conserver.
    """
    df_sample = pd.read_csv(file_path, sep="\t", nrows=sample_nrows, low_memory=False)

    missing_rates = df_sample.isnull().mean()
    auto_kept_columns = missing_rates[missing_rates < missing_threshold].index.tolist()

    collinear_drops = [
        'energy-kj_100g', 'energy-kcal_100g',
        'salt_100g', 'added-salt_100g', 'added-sugars_100g'
    ]
    target_leaks = [
        'nutriscore_grade', 'environmental_score_grade', 'nutrient_levels_tags'
    ]

    columns_to_keep = []
    for col in auto_kept_columns:
        is_collinear = col in collinear_drops
        is_leak = col in target_leaks
        is_url = 'url' in col.lower()
        if not (is_collinear or is_leak or is_url):
            columns_to_keep.append(col)

    crucial_cols = ['nutriscore_score', 'categories_tags', 'brands_tags', 'countries_tags']
    for col in crucial_cols:
        if col not in columns_to_keep and col in df_sample.columns:
            columns_to_keep.append(col)

    print(f"Selected {len(columns_to_keep)} clean, leak-free, non-URL columns out of {df_sample.shape[1]}.")
    return columns_to_keep


def stream_and_filter_rows(file_path, columns_to_keep, completeness_gate=0.30, chunk_size=50000):
    """
    Lit le fichier par chunks, filtre les lignes selon leur taux de complétude,
    puis sépare en deux DataFrames : labeled (avec nutriscore) et unlabeled (sans).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_labeled, df_unlabeled)
    """
    labeled_chunks = []
    unlabeled_chunks = []
    total_rows_scanned = 0

    feature_cols = [c for c in columns_to_keep if c != 'nutriscore_score']
    min_features_required = int(len(feature_cols) * completeness_gate)

    for chunk in pd.read_csv(file_path, sep="\t", usecols=columns_to_keep, chunksize=chunk_size, low_memory=False):
        total_rows_scanned += len(chunk)

        chunk['feature_completeness'] = chunk[feature_cols].notna().sum(axis=1)
        high_quality_chunk = chunk[chunk['feature_completeness'] >= min_features_required].copy()

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

    return df_labeled, df_unlabeled


def clean_target_column(df_labeled):
    """
    Convertit nutriscore_score en numérique et retire les lignes où la
    conversion échoue.
    """
    df_labeled = df_labeled.copy()
    df_labeled['nutriscore_score'] = pd.to_numeric(df_labeled['nutriscore_score'], errors='coerce')
    df_labeled = df_labeled.dropna(subset=['nutriscore_score']).copy()
    print(f"Valid, numeric labeled rows remaining: {len(df_labeled):,}")
    return df_labeled


def subsample_datasets(df_labeled, df_unlabeled, target_size=100000, seed=42):
    """
    Échantillonne les deux DataFrames à target_size lignes.
    Labeled : échantillonnage stratifié (bins pseudo-aléatoires).
    Unlabeled : échantillonnage aléatoire simple.

    Raises
    ------
    ValueError
        Si l'un des deux DataFrames contient moins de lignes que target_size.
    """
    if len(df_labeled) < target_size:
        raise ValueError(
            f"Seulement {len(df_labeled)} lignes labellisées disponibles, "
            f"target_size={target_size} trop élevé."
        )
    if len(df_unlabeled) < target_size:
        raise ValueError(
            f"Seulement {len(df_unlabeled)} lignes non labellisées disponibles, "
            f"target_size={target_size} trop élevé."
        )

    rng = np.random.default_rng(seed=seed)
    df_labeled = df_labeled.copy()
    df_labeled['random_bin'] = rng.integers(low=0, high=10, size=len(df_labeled))

    df_labeled_sub, _ = train_test_split(
        df_labeled,
        train_size=target_size,
        stratify=df_labeled['random_bin'],
        random_state=seed
    )
    df_labeled_sub = df_labeled_sub.drop(columns=['random_bin'])

    df_unlabeled_sub = df_unlabeled.sample(n=target_size, random_state=seed)

    return df_labeled_sub, df_unlabeled_sub


def deduplicate_and_save(df_labeled_sub, df_unlabeled_sub, output_dir, target_size=100000):
    """
    Déduplique les deux sous-échantillons et les sauvegarde en CSV.

    Returns
    -------
    tuple
        (df_labeled_sub, df_unlabeled_sub, labeled_out_path, unlabeled_out_path)
    """
    df_labeled_sub = df_labeled_sub.drop_duplicates().reset_index(drop=True)
    df_unlabeled_sub = df_unlabeled_sub.drop_duplicates().reset_index(drop=True)

    if len(df_labeled_sub) < target_size:
        print(f"⚠️ Après déduplication, il ne reste que {len(df_labeled_sub)} lignes labellisées (< {target_size}).")
    if len(df_unlabeled_sub) < target_size:
        print(f"⚠️ Après déduplication, il ne reste que {len(df_unlabeled_sub)} lignes non labellisées (< {target_size}).")

    output_dir = Path(output_dir)
    labeled_out_path = output_dir / f'labeled_subsample_{target_size // 1000}k.csv'
    unlabeled_out_path = output_dir / f'unlabeled_subsample_{target_size // 1000}k.csv'

    df_labeled_sub.to_csv(labeled_out_path, index=False)
    df_unlabeled_sub.to_csv(unlabeled_out_path, index=False)

    print("Verification Statistics:")
    print(f" * Final Labeled Subsample: {df_labeled_sub.shape[0]:,} unique rows | {df_labeled_sub.shape[1]} columns")
    print(f" * Final Unlabeled Subsample: {df_unlabeled_sub.shape[0]:,} unique rows | {df_unlabeled_sub.shape[1]} columns")
    print("\n✅ Files successfully exported to:")
    print(f"   -> {labeled_out_path}")
    print(f"   -> {unlabeled_out_path}")

    return df_labeled_sub, df_unlabeled_sub, labeled_out_path, unlabeled_out_path


def generate_clean_subsamples(
    file_path: str,
    output_dir: str = None,
    target_size: int = 100000,
    missing_threshold: float = 0.85,
    completeness_gate: float = 0.30,
    seed: int = 42
):
    """
    Fonction d'orchestration : lit le dataset brut Open Food Facts, sélectionne
    les colonnes pertinentes, filtre les lignes de mauvaise qualité, sépare les
    aliments avec/sans nutriscore, échantillonne et sauvegarde deux
    sous-datasets de target_size lignes chacun.

    Parameters
    ----------
    file_path : str
        Chemin complet vers le dataset brut (dataset-openfoodfacts.csv).
    output_dir : str, optional
        Dossier de sortie pour les fichiers CSV générés.
        Par défaut : dossier "processed" à côté du fichier source.
    target_size : int
        Nombre de lignes voulues dans chaque sous-échantillon (défaut : 100 000).
    missing_threshold : float
        Seuil de suppression des colonnes trop incomplètes (défaut : 85%).
    completeness_gate : float
        Taux minimal de complétude requis par ligne (défaut : 30%).
    seed : int
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_labeled_sub, df_unlabeled_sub)
    """
    if output_dir is None:
        output_dir = Path(file_path).resolve().parent

    print("--- STEP 1: ANALYZING COLUMNS AND DECONTAMINATING TARGET LEAKS ---")
    columns_to_keep = select_relevant_columns(file_path, missing_threshold)

    print("\n--- STEP 2: STREAMING DATA & APPLYING COMPLETENESS FILTER ---")
    df_labeled, df_unlabeled = stream_and_filter_rows(file_path, columns_to_keep, completeness_gate)

    print("\n--- STEP 3: CASTING TARGET & REMOVING CORRUPT LABELS ---")
    df_labeled = clean_target_column(df_labeled)

    print("\n--- STEP 4: GENERATING REPRESENTATIVE SUBSAMPLES ---")
    df_labeled_sub, df_unlabeled_sub = subsample_datasets(df_labeled, df_unlabeled, target_size, seed)

    print("\n--- STEP 5: RESOLVING DOUBLONS & SAVING FILES ---")
    df_labeled_sub, df_unlabeled_sub, _, _ = deduplicate_and_save(
        df_labeled_sub, df_unlabeled_sub, output_dir, target_size
    )

    return df_labeled_sub, df_unlabeled_sub


if __name__ == "__main__":
    # Bloc de test : ne s'exécute que si ce fichier est lancé directement,
    # jamais quand il est importé depuis pipeline.py
    RACINE = Path(__file__).resolve().parent.parent
    raw_dataset_path = RACINE / "data" / "raw" / "dataset-openfoodfacts.csv"
    processed_dir = RACINE / "data" / "processed"
    generate_clean_subsamples(str(raw_dataset_path), output_dir=str(processed_dir))