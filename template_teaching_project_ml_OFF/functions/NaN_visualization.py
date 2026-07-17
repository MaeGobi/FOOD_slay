import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

def NaN_visualization(path, colonnes_cibles=None, threshold_view=0.5):
    # Si aucune colonne n'est précisée, on les récupère depuis l'en-tête du fichier
    if colonnes_cibles is None:
        colonnes_cibles = pd.read_csv(path, sep='\t', nrows=0).columns.tolist()

    # Dictionnaire pour accumuler les comptes
    missing_counts = {col: 0 for col in colonnes_cibles}
    total_rows = 0

    chunksize = 10_000  # ajustez selon votre RAM

    for chunk in pd.read_csv(
        path,
        sep='\t',              # OFF utilise des tabulations
        usecols=colonnes_cibles,
        chunksize=chunksize,
        low_memory=False,
        on_bad_lines='skip'
    ):
        total_rows += len(chunk)
        for col in colonnes_cibles:
            missing_counts[col] += chunk[col].isna().sum()

    print(f"Total lignes traitées : {total_rows}")
    for col, n in missing_counts.items():
        pct = 100 * n / total_rows
        print(f"{col}: {n} manquants ({pct:.2f}%)")

    # Générer graphique
    percent_missing = pd.Series({col: 100 * n / total_rows for col, n in missing_counts.items()})

    filtered = percent_missing[percent_missing.values > threshold_view].sort_values(ascending=True)

    if filtered.empty:
        print(f"Aucune colonne au-dessus du seuil de {threshold_view}%.")
        return

    # --- Normalisation des valeurs entre 0 et 1 ---
    norm = mcolors.Normalize(vmin=filtered.min(), vmax=filtered.max())
    colors = [cm.rainbow(norm(val)) for val in filtered.values]

    plt.figure(figsize=(10, max(6, 0.4 * len(filtered))))
    ax = sns.barplot(x=filtered, y=filtered.index, orient='h', palette=colors)
    ax.set_title(f"Répartition du pourcentage de valeurs manquantes supérieures au seuil de {threshold_view}%")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path_test = r"C:\DATA\Projets\2026_07_DESU_Data_sicences\Projets\Devoir_Machine_learning\FOOD_slay\template_teaching_project_ml_OFF\data\raw\dataset-openfoodfacts.csv"
    NaN_visualization(path_test, threshold_view=0.5)
    