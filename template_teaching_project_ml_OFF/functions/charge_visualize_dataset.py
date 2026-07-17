import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sklearn as sk
import seaborn as sns

def charge_visualize_dataset() :
    # Charger et visualiser dataframe
    path = r"C:\DATA\Projets\2026_07_DESU_Data_sicences\Projets\Devoir_Machine_learning\dataset-openfoodfacts.csv"
    df = pd.read_csv(path, nrows = 1000, sep='\t')
    print(df.head())

    # Description dataset
    print(len(df.columns))   # Nombre de colonnes (équivalent de ncol sous R)
    print(len(df.index))   # Nombre de lignes (équivalent de nrow sous R)

    # Extraire les noms des variables

    list(df.columns.values)

 if __name__ == "__main__":
        charge_visualize_dataset()