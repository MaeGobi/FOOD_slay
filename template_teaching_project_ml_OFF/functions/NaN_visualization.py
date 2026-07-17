# Exploration des NAs

path = r"C:\DATA\Projets\2026_07_DESU_Data_sicences\Projets\Devoir_Machine_learning\dataset-openfoodfacts.csv"

def NaN_visualization() :
    colonnes_cibles = [
    'categories_tags','nutriscore_score', 
    'nutriscore_grade',
    'energy-kj_100g',
    'energy-kcal_100g',
    'energy_100g',
    'energy-from-fat_100g',
    'fat_100g',
    'saturated-fat_100g',
    'butyric-acid_100g',
    'caproic-acid_100g',
    'caprylic-acid_100g',
    'capric-acid_100g',
    'lauric-acid_100g',
    'myristic-acid_100g',
    'palmitic-acid_100g',
    'stearic-acid_100g',
    'arachidic-acid_100g',
    'behenic-acid_100g',
    'lignoceric-acid_100g',
    'cerotic-acid_100g',
    'montanic-acid_100g',
    'melissic-acid_100g',
    'unsaturated-fat_100g',
    'monounsaturated-fat_100g',
    'omega-9-fat_100g',
    'polyunsaturated-fat_100g',
    'omega-3-fat_100g',
    'omega-6-fat_100g',
    'alpha-linolenic-acid_100g',
    'eicosapentaenoic-acid_100g',
    'docosahexaenoic-acid_100g',
    'linoleic-acid_100g',
    'arachidonic-acid_100g',
    'gamma-linolenic-acid_100g',
    'dihomo-gamma-linolenic-acid_100g',
    'oleic-acid_100g',
    'elaidic-acid_100g',
    'gondoic-acid_100g',
    'mead-acid_100g',
    'erucic-acid_100g',
    'nervonic-acid_100g',
    'trans-fat_100g',
    'cholesterol_100g',
    'carbohydrates_100g',
    'sugars_100g',
    'added-sugars_100g',
    'sucrose_100g',
    'glucose_100g',
    'fructose_100g',
    'galactose_100g',
    'lactose_100g',
    'maltose_100g',
    'maltodextrins_100g',
    'psicose_100g',
    'starch_100g',
    'polyols_100g',
    'erythritol_100g',
    'isomalt_100g',
    'maltitol_100g',
    'sorbitol_100g',
    'fiber_100g',
    'soluble-fiber_100g',
    'polydextrose_100g',
    'insoluble-fiber_100g',
    'proteins_100g',
    'casein_100g',
    'serum-proteins_100g',
    'nucleotides_100g',
    'salt_100g',
    'added-salt_100g',
    'sodium_100g',
    'alcohol_100g',
    'vitamin-a_100g',
    'beta-carotene_100g',
    'vitamin-d_100g',
    'vitamin-e_100g',
    'vitamin-k_100g',
    'vitamin-c_100g',
    'vitamin-b1_100g',
    'vitamin-b2_100g',
    'vitamin-pp_100g',
    'vitamin-b6_100g',
    'vitamin-b9_100g',
    'folates_100g',
    'vitamin-b12_100g',
    'biotin_100g',
    'pantothenic-acid_100g',
    'silica_100g',
    'bicarbonate_100g',
    'potassium_100g',
    'chloride_100g',
    'calcium_100g',
    'phosphorus_100g',
    'iron_100g',
    'magnesium_100g',
    'zinc_100g',
    'copper_100g',
    'manganese_100g',
    'fluoride_100g',
    'selenium_100g',
    'chromium_100g',
    'molybdenum_100g',
    'iodine_100g',
    'caffeine_100g',
    'taurine_100g',
    'methylsulfonylmethane_100g',
    'hydroxymethylbutyrate_100g',
    'ph_100g',
    'fruits-vegetables-legumes_100g',
    'collagen-meat-protein-ratio_100g',
    'cocoa_100g',
    'chlorophyl_100g',
    'carbon-footprint_100g',
    'glycemic-index_100g',
    'water-hardness_100g',
    'choline_100g',
    'phylloquinone_100g',
    'beta-glucan_100g',
    'inositol_100g',
    'carnitine_100g',
    'sulphate_100g',
    'nitrate_100g',
    'acidity_100g',
    'carbohydrates-total_100g',
    'water_100g']  # à adapter

    # Dictionnaire pour accumuler les comptes
    missing_counts = {col: 0 for col in colonnes_cibles}
    total_rows = 0

    chunksize = 10_000  # ajustez selon votre RAM

    for chunk in pd.read_csv(
        path,
        sep='\t',              # OFF utilise des tabulations
        usecols=colonnes_cibles,  # ne charge QUE les colonnes utiles
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


    # --- Votre code, inchangé ---
    threshold_view = 0.5


    filtered = percent_missing[percent_missing.values > threshold_view].sort_values(ascending=True)

    # --- Normalisation des valeurs entre 0 et 1 ---
    norm = mcolors.Normalize(vmin=filtered.min(), vmax=filtered.max())
    colors = [cm.rainbow(norm(val)) for val in filtered.values]

    plt.figure(figsize=(10, max(6, 0.4 * len(filtered))))

    ax = sns.barplot(x = filtered, y = filtered.index, orient='h', palette=colors);
    ax.set_title(f"Répartition du pourcentage de valeurs manquantes supérieures au seuil de {threshold_view}%");
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    NaN_visualization()