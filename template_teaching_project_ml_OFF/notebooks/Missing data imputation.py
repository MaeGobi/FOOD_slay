import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def imputation():
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


    if __name__ == "__main__":
        imputation()