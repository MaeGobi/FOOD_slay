import pandas as pd
import numpy as np
import matplotlib as plt
import sklearn as sk

# Exemple d'utilisation avec un DataFrame
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=50, sep='\t', encoding="utf-8")
print(df.head())
