import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

def get_features_cat_regression(df, target_col, columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Analiza columnas categóricas para determinar cuáles se asocian significativamente
    con una variable objetivo continua, utilizando pruebas estadísticas (T-Test para
    dos categorías y ANOVA para más de dos). Retorna las columnas categóricas
    significativas según un umbral de p-valor determinado.
    -----
    Parametros
    -----
    df : pandas.DataFrame
        El DataFrame que contiene la variable objetivo y las columnas categóricas
        a analizar.

    target_col : str
        Nombre de la columna del DataFrame que se usará como variable objetivo.
        Debe ser de tipo numérico y con distribución continua.

    columns : list, opcional
        Lista de columnas categóricas a probar. Si no se proporciona, la función
        seleccionará automáticamente las columnas no numéricas de "df".
    
    pvalue : float, opcional
        Nivel de significancia para determinar si hay diferencias estadísticamente
        significativas en la variable objetivo según cada columna categórica. El
        valor por defecto es 0.05.
    
    with_individual_plot : bool, opcional
        Si se establece en True, se generarán diagramas histograma con `sns.histplot`
        para observar la distribución de la variable objetivo separada por las
        categorías de la columna en cuestión.
    """
     
    significant_columns = []
    
    # Validaciones iniciales
    if target_col not in df.columns:
        print(f"La columna '{target_col}' no está presente en el DataFrame.")
        return None

    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"La columna '{target_col}' no es numérica continua.")
        return None

    if not columns:  # Si no se especifican columnas, selecciona categóricas por defecto
        columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if not columns:
        print("No hay columnas categóricas en el DataFrame.")
        return None

    # Probar cada columna categórica
    for col in columns:
        unique_values = df[col].dropna().unique()
        if len(unique_values) < 2:
            continue  # Omitir columnas sin categorías válidas

        # Selección del test estadístico
        if len(unique_values) == 2:
            group1 = df[df[col] == unique_values[0]][target_col]
            group2 = df[df[col] == unique_values[1]][target_col]
            stat, p = ttest_ind(group1, group2, nan_policy='omit')
        else:
            groups = [df[df[col] == val][target_col] for val in unique_values]
            stat, p = f_oneway(*groups)

        # Verificar si el p-valor es significativo
        if p < pvalue:
            significant_columns.append(col)

            # Visualización opcional
            if with_individual_plot:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=target_col, hue=col, multiple="stack", kde=True)
                plt.title(f"Histograma de {target_col} agrupado por {col}")
                plt.xlabel(target_col)
                plt.ylabel("Frecuencia")
                plt.show()

    # Retornar columnas significativas
    if not significant_columns:
        print("No se encontraron columnas categóricas significativas.")
        return None

    return significant_columns

