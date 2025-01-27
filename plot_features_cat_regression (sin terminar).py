import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import variables as var
from scipy.stats import ttest_ind, f_oneway, pearsonr

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

def plot_features_cat_regression(dataframe, target_col = "", columns = [], pvalue = 0.05, with_individual_plot = False, size_group = 4): # Cardinalidad numéricas categóricas.

    """
    Pinta los histogramas agrupados de la variable target_col para cada uno de los valores de columns, siempre y cuando el test de significación sea 1-pvalue. 
    Si columns no tiene valores, se pintarán los histogramas de las variables numéricas teniendo en cuenta lo mismo.

    Argumentos:
    dataframe (DataFrame): dataframe a estudiar.
    target_col (str): nombre de la columna con los datos target.
    columns (list): lista con el nombre de las columnas a comparar con target_col.
    pvalue (float64): valor p.
    with_individual_plot (bool): si es True pinta cada histograma por separado.
    size_group (int): por defecto 4. Si las columnas categóricas tienen más categorías que ese argumento, se dividirán sus plots.

    Retorna:
    object: figura o figuras con uno o varios histogramas.
    list: lista con las columnas que se hayan elegido (que tengan significación estadística).
    None: si se produce algún error, se devuelve None y un print con la explicación del error.
    """

    sns.set_style(var.SNS_STYLE)
    
    # Comprobamos utilizando la función is_valid_params que las columnas cumplen con su tipo elegido (target numérico y columns categóricas):
    numeric_types = [var.TIPO_NUM_CONTINUA, var.TIPO_NUM_DISCRETA]
    categoric_types = [var.TIPO_BINARIA, var.TIPO_CATEGORICA]
    if not fnc.is_valid_params(dataframe, target_col, columns, numeric_types, categoric_types):
        return None
    
    if len(columns) == 0: # Al no introducir el argumento columns, seleccionamos las variables numéricas:
        df_types = fnc.tipifica_variables(dataframe, var.UMBRAL_CATEGORIA, var.UMBRAL_CONTINUA)
        num_col = df_types[df_types[var.COLUMN_TIPO].isin(numeric_types)][var.COLUMN_NOMBRE].to_list()
    
    sig_num_col = []
    sig_cat_col = []

    if columns == []: # Si no introducimos columnas categóricas, cogeremos las columnas numéricas:
        if dataframe[num_col].isna().sum().sum() != 0: # Verificamos si existen nulos para avisar al usuario de que existen:
             print(f"Existen nulos o NaN presentes en las variables numéricas de estudio, tenga en cuenta que el análisis de correlación se realizará eliminando esos nulos.")

        for col in num_col: # Usamos la correlación de pearson para ver si están relacionadas (eliminando nulos en caso de que haya):
            p = pearsonr(dataframe.dropna()[target_col], dataframe.dropna()[col]).pvalue
            if p < pvalue: # Si la relación entre variables entra en la significación seleccionada, nos guardamos esa columna:
                sig_num_col.append(col)
        if target_col in sig_num_col: # Como nuestro target es una variable numérica, la quitamos de la lista
            sig_num_col.remove(target_col)
        if sig_num_col != []:
            print(f"Las columnas numéricas elegidas son: {sig_num_col}")
        else:
             print("No se ha seleccionado ninguna columna numérica.")

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
            print(f"Las columnas categóricas elegidas son: {sig_cat_col}")
        else:
             print("No se ha seleccionado ninguna columna categórica.")
    
    if sig_num_col != []:
        if with_individual_plot:
            for col in sig_num_col:
                    # Crea el gráfico
                    plt.figure(figsize=(12, 8))
                    sns.scatterplot(x = target_col, y = col, data = dataframe)
                    plt.title(f"Relación entre {col} y {target_col}")
                    plt.xlabel(col)
                    plt.ylabel(f"{col}")
                    plt.show();

        else:
            subplots = len(sig_num_col) # Guardamos las filas que va a tener nuestra figura
            plt.figure(figsize=(20,15))
            for i, col in enumerate(sig_num_col):
                plt.subplot(subplots, 1, i+1) # Colocamos cada columna en una de las filas de nuestra figura
                sns.scatterplot(x = target_col, y = col, data = dataframe)
                plt.title(f"Relación entre {col} y {target_col}")
                plt.xlabel(col)
                plt.ylabel(f"{col}")
            plt.tight_layout()
            plt.show();
    
    elif sig_cat_col != []:
        if with_individual_plot:
            for col in sig_cat_col:
                unique_categories = dataframe[col].unique()
                if len(unique_categories) > size_group:
                    num_plots = int(np.ceil(len(unique_categories) / size_group))
                    for i in range(num_plots):
                        cat_subset = unique_categories[i * size_group:(i + 1) * size_group]
                        data_subset = dataframe.loc[dataframe[col].isin(cat_subset), [col, target_col]]
                        plt.figure(figsize=(12, 8))
                        sns.histplot(x = target_col, hue = col, data = data_subset, kde = True)
                        plt.xlabel(target_col)
                        plt.ylabel("")
                        plt.show();
                        
                else:
                # Crea el gráfico
                    plt.figure(figsize=(12, 8))
                    sns.histplot(x = target_col, hue = col, data = dataframe, kde = True)
                    plt.title(f"Relación entre {col} y {target_col}")
                    plt.xlabel(target_col)
                    plt.ylabel("")
                    plt.show();

        else:
            subplots = len(sig_cat_col) # Guardamos las filas que va a tener nuestra figura
            plt.figure(figsize=(20,15))
            for i, col in enumerate(sig_cat_col):
                plt.subplot(subplots, 1, i+1) # Colocamos cada columna en una de las filas de nuestra figura
                sns.histplot(x = target_col, hue = col, data = dataframe, kde = True)
                plt.title(f"Relación entre {col} y {target_col}")
                plt.xlabel(cat)
                plt.ylabel("")
            plt.tight_layout()
            plt.show();