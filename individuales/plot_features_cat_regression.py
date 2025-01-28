import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import variables as var
import functions as fnc
from scipy.stats import ttest_ind, f_oneway, pearsonr

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

def plot_features_cat_regression(dataframe, target_col = "", columns = [], pvalue = 0.05, with_individual_plot = False, size_group = 3): # Cardinalidad numéricas categóricas.

    """
    Pinta los histogramas agrupados de la variable target_col para cada uno de los valores de columns, siempre y cuando el test de significación sea 1-pvalue. 
    Si columns no tiene valores, se pintarán los histogramas de todas las variables categóricas que cumplan con la significación.

    Argumentos:
    dataframe (DataFrame): dataframe a estudiar.
    target_col (str): nombre de la columna con los datos target.
    columns (list): lista con el nombre de las columnas a comparar con target_col.
    pvalue (float64): valor p.
    with_individual_plot (bool): si es True pinta cada histograma por separado.
    size_group (int): por defecto 3. Si las columnas categóricas tienen más categorías que ese argumento, se dividirán sus plots.

    Retorna:
    list: lista con las columnas que se hayan elegido (que tengan significación estadística).
    object: figura o figuras con uno o varios histogramas.
    None: si se produce algún error, se devuelve None y un print con la explicación del error.
    """

    sns.set_style = var.SNS_STYLE

    # Validación inicial de parámetros
    numeric_types = [var.TIPO_NUM_CONTINUA, var.TIPO_NUM_DISCRETA]
    categoric_types = [var.TIPO_BINARIA, var.TIPO_CATEGORICA]
    if not fnc.is_valid_params(dataframe, target_col, columns, numeric_types, categoric_types):
        return None
    if len(columns) == 0:
        df_types = fnc.tipifica_variables(dataframe, var.UMBRAL_CATEGORIA, var.UMBRAL_CONTINUA)
        columns = df_types[df_types[var.COLUMN_TIPO].isin(categoric_types)][var.COLUMN_NOMBRE].to_list()

    sig_cat_col = []

    # Obtenemos el pvalue de las columnas categóricas mediante T de Student y ANOVA
    for col in columns:
        cat = dataframe[col].unique()
        if len(cat) < 2:
            continue
        if len(cat) == 2:
            group0 = dataframe.loc[dataframe[col] == cat[0], target_col]
            group1 = dataframe.loc[dataframe[col] == cat[1], target_col]
            p = ttest_ind(group0, group1).pvalue
        else:
            groups = [dataframe.loc[dataframe[col] == c, target_col] for c in cat]
            groups = [g for g in groups if len(g) > 1]  # Validar que haya datos suficientes
            if len(groups) < 2:
                continue
            p = f_oneway(*groups).pvalue
        if p < pvalue:
            sig_cat_col.append(col)

    if sig_cat_col:
        print(f"Las columnas categóricas elegidas son: {sig_cat_col}")
    else:
        print("No se ha seleccionado ninguna columna categórica.")
        return

    if with_individual_plot:
        # Generamos gráficos individuales
        for col in sig_cat_col:
            unique_categories = dataframe[col].unique()
            # Dividimos por grupos en caso de que nuestra variable categórica tenga muchas categorías únicas
            if len(unique_categories) > size_group:
                num_plots = int(np.ceil(len(unique_categories) / size_group))
                for i in range(num_plots):
                    cat_subset = unique_categories[i * size_group:(i + 1) * size_group]
                    data_subset = dataframe.loc[dataframe[col].isin(cat_subset), [col, target_col]]
                    if not data_subset.empty:
                        plt.figure(figsize=(12, 8))
                        sns.histplot(x=target_col, hue=col, data=data_subset, kde=len(data_subset) > 1)
                        plt.xlabel(target_col)
                        plt.ylabel("")
                        plt.show();
            else:
                if not dataframe.empty:
                    plt.figure(figsize=(12, 8))
                    sns.histplot(x=target_col, hue=col, data=dataframe, kde=len(dataframe) > 1)
                    plt.title(f"Relación entre {col} y {target_col}")
                    plt.xlabel(target_col)
                    plt.ylabel("")
                    plt.show();
    
    else:
        # Obtenemos el número total de subplots
        subplots = 0
        columns_groups = {}
        for col in sig_cat_col:
            unique_categories = dataframe[col].unique()
            if len(unique_categories) > size_group:
                num_plots = int(np.ceil(len(unique_categories) / size_group))
                subplots += num_plots
                columns_groups[col] = np.array_split(unique_categories, num_plots)
            else:
                subplots += 1
                columns_groups[col] = [unique_categories]
        # Creamos una figura con subplots
        fig, axes = plt.subplots(nrows=subplots, ncols=1, figsize=(20, 5 * subplots))

        if subplots == 1:
            axes = [axes]  # Asegurarse de que sea una lista si hay un único subplot
            
        subplot_idx = 0
        for col, grupos in columns_groups.items():
            for grupo in grupos:
                # Filtrar datos por el grupo actual
                data_filtrada = dataframe[dataframe[col].isin(grupo)]
                # Ploteamos asegurándonos de que el subdataframe que hemos creado no está vacío
                if not data_filtrada.empty:
                    sns.histplot(data = data_filtrada, x = target_col, hue = col, ax = axes[subplot_idx], kde = len(data_filtrada) > 1)
                    axes[subplot_idx].set_title(f'{col} - Categorías: {list(grupo)}')
                    axes[subplot_idx].set_xlabel(target_col)
                    axes[subplot_idx].set_ylabel("")
                    subplot_idx += 1

        # Ajustamos el diseño y mostramos la figura completa
        plt.tight_layout()
        plt.show();