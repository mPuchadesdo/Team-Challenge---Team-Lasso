import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, pearsonr

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

def plot_features_cat_regression(dataframe, target_col = "", columns = [], pvalue = 0.05, with_individual_plot = False):

    """
        Pinta los histogramas agrupados de la variable target_col para cada uno de los valores de columns, siempre y cuando el test de significación sea 1-pvalue. 
        Si columns no tiene valores, se pintarán los histogramas de las variables numéricas teniendo en cuenta lo mismo.

        Argumentos:
        dataframe (DataFrame): dataframe a estudiar.
        target_col (str): nombre de la columna con los datos target.
        columns (list): lista con el nombre de las columnas a comparar con target_col.
        pvalue (float64): valor p.
        with_individual_plot (bool): si es True pinta cada histograma por separado.

        Retorna:
        tipo: Descripción de lo que retorna la función. COMPLETAR
        """

    sig_num_col = []
    sig_cat_col = []

    if columns == []: # Si no introducimos columnas categóricas, cogeremos las columnas numéricas continuas:
        num_col = []
        for col in dataframe.describe().columns.to_list():
            if len(dataframe[col].unique()) > 12: # Verificamos que no sea una categórica codificada numéricamente
                num_col.append(col)
        for col in num_col: # Usamos la correlación de pearson para ver si están relacionadas:
            p = pearsonr(dataframe[target_col], dataframe[col]).pvalue
            if p < pvalue: # Si la relación entre variables entra en la significación seleccionada, nos guardamos esa columna:
                sig_num_col.append(col)

    else:
        for col in columns:
            cat = dataframe[col].unique()
            if len(cat) < 2:
                continue

            if len(cat) == 2: # Si la categoría es binaria, utilizamos el test T de Student
                group0 = dataframe.loc[dataframe[col] == cat[0], target_col]
                group1 = dataframe.loc[dataframe[col] == cat[1], target_col]
                p = ttest_ind(group0, group1).pvalue
            else: # Si la columna tiene más de dos categorías, utilizamos ANOVA:
                groups = [dataframe.loc[dataframe[col] == c, target_col] for c in cat]
                p = f_oneway(*groups).pvalue

            if p < pvalue: # Si la relación entre variables entra en la significación seleccionada, nos guardamos esa columna:
                sig_cat_col.append(col)
    
    if sig_cat_col != []:
        for cat in sig_cat_col:
                # Crea el gráfico
                plt.figure(figsize=(15, 8))
                sns.histplot(x = target_col, hue = cat, data = dataframe, kde = True)
                plt.show()