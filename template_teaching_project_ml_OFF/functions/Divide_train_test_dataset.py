def divide_test_train() :
    path = r"C:\DATA\Projets\2026_07_DESU_Data_sicences\Projets\Devoir_Machine_learning\dataset-openfoodfacts.csv"
    path_train = r"C:\DATA\Projets\2026_07_DESU_Data_sicences\Projets\Devoir_Machine_learning\dataset_train.csv"
    path_test = r"C:\DATA\Projets\2026_07_DESU_Data_sicences\Projets\Devoir_Machine_learning\dataset_test.csv"

    chunksize = 100_000
    first_chunk = True

    for chunk in pd.read_csv(
        path,
        sep='\t',
        low_memory=False,
        on_bad_lines='skip',
        chunksize=chunksize
    ):
        # Lignes où nutriscore_score est renseigné -> dataset d'entraînement
        avec_score = chunk[chunk['nutriscore_score'].notna()]
        
        # Lignes où nutriscore_score est manquant -> dataset de test
        sans_score = chunk[chunk['nutriscore_score'].isna()]
        
        # Écriture incrémentale (mode 'a' = append), header seulement au premier chunk
        avec_score.to_csv(path_train, mode='a', index=False, header=first_chunk)
        sans_score.to_csv(path_test, mode='a', index=False, header=first_chunk)
        
        first_chunk = False

    print("Séparation terminée.")

    # Appel de cahque dataset pour vérifier l'application de la procédure de séparation
    col = ['nutriscore_score']
    df_train = pd.read_csv(path_train, usecols=col, nrows=1000)
    df_test = pd.read_csv(path_test, usecols=col, nrows=1000)

        
    # Vérification du dataset train
    df_train.head(500)
    # Vérifier s'il y a des NAs dans le dataset de train
    def count_empty(df_train):
        return df_train.apply(lambda col: col.isna() | (col.astype(str).str.strip() == '')).sum()

    print(count_empty(df_train))

    # Vérification du dataset test 
    df_test.head(500)


if __name__ == "__main__":
    divide_test_train()