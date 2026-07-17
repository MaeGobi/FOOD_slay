from pathlib import Path

from functions.charge_visualize_dataset import charge_visualize_dataset
from functions.NaN_visualization import NaN_visualization
from functions.generate_clean_subsamples import generate_clean_subsamples
from functions.Divide_train_test_dataset import divide_train_test
from functions.remove_physical_outliers import remove_physical_outliers
from functions.Missing_data_imputation import imputation

# --- Gestion des chemins ---
RACINE = Path(__file__).resolve().parent.parent
CHEMIN_DATA_RAW = RACINE / "data" / "raw" / "dataset-openfoodfacts.csv"  # à adapter selon le nom réel
CHEMIN_DATA_PROCESSED = RACINE / "data" / "processed"

def main():
    # Étape 1 : chargement et visualisation
    df = charge_visualize_dataset(CHEMIN_DATA_RAW)

    # Étape 2 : visualisation des valeurs manquantes
    NaN_visualization(df)

    # Étape 3 : suppression des outliers physiques
    df = remove_physical_outliers(df)

    # Étape 4 : imputation des données manquantes
    df = imputation(df)

    # Étape 5 : génération de sous-échantillons propres
    df = generate_clean_subsamples(df)

    # Étape 6 : division train/test
    train_df, test_df = divide_train_test(df)

    # Sauvegarde
    train_df.to_csv(CHEMIN_DATA_PROCESSED / "train.csv", index=False)
    test_df.to_csv(CHEMIN_DATA_PROCESSED / "test.csv", index=False)

    print("Pipeline terminée avec succès.")

if __name__ == "__main__":
    main()