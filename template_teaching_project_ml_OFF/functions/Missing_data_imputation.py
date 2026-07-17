import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sklearn as sk
import seaborn as sns
from sklearn.impute import SimpleImputerImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def imputation():
    # Imputation des variables numériques par KNN
    numeric_features = df.select_dtypes(include=['float','int'])
    numeric_features.shape

    from sklearn.impute import KNNImputer
    imputation = KNNImputer(missing_values=np.nan)
    imputed = imputation.fit_transform(numeric_features)
    imputed.shape

    df.loc[:,numeric_features.columns] = imputed

    def compare_dist(feature):
        fig, axes = plt.subplots(1,2,figsize=(12,3))
        sns.histplot(raw_df.loc[:,feature],kde=True, ax=axes[0])
        axes[0].set_title(f"Raw {feature}");

        sns.histplot(df.loc[:,feature],kde=True, ax=axes[1])
        axes[1].set_title(f"Imputed {feature}");
    compare_dist("LotFrontage")

    # Imputation des variables catégorielles par le mode
    categorical_features = df.select_dtypes(include=['object','category'])
    categorical_features.shape

    from sklearn.impute import SimpleImputer
    imputation_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputed_cat = imputation_cat.fit_transform(categorical_features)
    imputed_cat.shape

    df.loc[:, categorical_features.columns] = imputed_cat

    def compare_dist_cat(feature):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        sns.countplot(x=raw_df.loc[:, feature], ax=axes[0])
        axes[0].set_title(f"Raw {feature}")
        axes[0].tick_params(axis='x', rotation=45)

        sns.countplot(x=df.loc[:, feature], ax=axes[1])
        axes[1].set_title(f"Imputed {feature}")
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()

    compare_dist_cat("nom_de_votre_variable_categorielle")
    
    

if __name__ == "__main__":
        imputation()