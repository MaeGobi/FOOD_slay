import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def charge_visualize_dataset(path, nrows=None):
    # Charger le dataframe
    df = pd.read_csv(path, nrows=nrows, sep='\t')
    print(df.head())

    # Description dataset
    print(len(df.columns))   # Nombre de colonnes
    print(len(df.index))     # Nombre de lignes

    # Extraire les noms des variables
    print(list(df.columns.values))

    return df

if __name__ == "__main__":

# Test rapide en local avec un chemin en dur (uniquement pour tester ce fichier seul)
    path_test = r"C:\DATA\Projets\2026_07_DESU_Data_sicences\Projets\Devoir_Machine_learning\FOOD_slay\template_teaching_project_ml_OFF\data\raw\dataset-openfoodfacts.csv"
    charge_visualize_dataset(path_test, nrows=1000)