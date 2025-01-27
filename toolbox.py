import functions as fnc
import variables as var

import pandas as pd
import numpy as np
import seaborn as sns

from scipy.stats import pearsonr, stats


def tipifica_variables(df, umbral_categoria= var.UMBRAL_CATEGORIA, umbral_continua= var.UMBRAL_CONTINUA):
    """
    Asigna un tipo a las variables de un dataframe en base a su cardinalidad y porcentaje de cardinalidad.

    Argumentos: 
        df: el dataframe a analizar
        umbral_categoria (int): número de veces max. que tiene que aparecer una variable para ser categórica
        bral_continua (float): porcentaje mínimo de cardinalidad que tiene que tener una variable para ser numérica continua

    Retorna: 
        Un dataframe con los resultados con dos columnas: 
        - El nombre de la variable 
        - El tipo sugerido para la variable 
        
    """

    resultados = [] #se crea una lista vacía para meter los resultados

    for columna in df.columns: #coge cada columna en el dataframe
        cardinalidad = df[columna].nunique() #calcula la cardinalidad 
        porcentaje_cardinalidad = (cardinalidad / len(df))*100 #calcula el porcentaje 

        if cardinalidad == 2:
            tipo = var.TIPO_BINARIA
            #tipo = "Binaria"

        elif (cardinalidad < umbral_categoria) and (cardinalidad != 2):
            tipo = var.TIPO_CATEGORICA
            #"Categórica"

        elif porcentaje_cardinalidad >= umbral_continua: #mayor que umbral categoria, mayor o igual que umbral continua
            tipo = var.TIPO_NUM_CONTINUA
            #"Numérica Continua"

        else:
            tipo = var.TIPO_NUM_DISCRETA #el porcentaje de cardinalidad es menor que umbral continua 
            #"Numérica Discreta"
        
        resultados.append({"variable": columna, "tipo": tipo}) #mete en la lista de resultados la columna y el tipo que se le asigna 

    return pd.DataFrame(resultados) #crea un dataframe con la lista de resultados 

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    '''
    Selecciona features numéricas basadas en su correlación con la variable target.
    La variable target debe ser numerica con alta cardinalidad.
    
    Args:
        df (pandas.DataFrame): DataFrame de entrada
        target_col (str): Nombre de la columna target
        umbral_corr (float): Umbral de correlación (valor absoluto) entre 0 y 1
        pvalue (float, optional): Nivel de significación para el test de hipótesis
        
    Returns:
        Lista de columnas que cumplen los criterios o None si hay error
    '''
    ## Validaciones de entrada
    # Verifica que target_col exista en el DataFrame y sea una cadena
    if target_col not in df.columns:
        print(f"Error: no encuentro {target_col} en el dataframe.")
        return None
    if not isinstance(target_col, str):
        print(f"Error: {target_col} debe ser una cadena de texto")
        return None
    # Verifica que la columna target_col sea numérica y con alta cardinalidad
    if not np.issubdtype(df[target_col].dtype, np.number):  #np.issubdtype comprueba si el tipo de datos de la columna es un subtipo de np.number (incluyendo enteros y float).
        print(f"Error: La columna '{target_col}' debe ser numérica.")
        return None
    n_unique = df[target_col].nunique()
    if n_unique < var.UMBRAL_CONTINUA:  # umbral arbitrario para considerar alta cardinalidad
        print(f"Error: La columna {target_col} debe tener alta cardinalidad")
        return None
    
    # Verifica que umbral_corr está entre 0 y 1
    if not (0 <= umbral_corr <= 1):
        print("Error: El umbral de correlación debe estar entre 0 y 1.")
        return None
    
    # Validación de pvalue si está presente
    if pvalue is not None and not 0 <= pvalue <= 1:
        print("El valor p debe estar entre 0 y 1.")
        return None
    

    # Lista para almacenar las columnas que cumplen con los criterios
    features_num = []

    # Iterar sobre todas las columnas numéricas del dataframe 
    # excluiendo la target y las numericas con cardinalidad baja que pueden ser consideradas categoricas 
    for col in df.select_dtypes(include=np.number).columns:
        if col != target_col and df[col].nunique() >= var.UMBRAL_CONTINUA:
            corr, p_val = stats.pearsonr(df[target_col].dropna(), df[col].dropna())
            # Verifica que la correlación supera el umbral
            if abs(corr) > umbral_corr:
                # Si pvalue es None, añade la columna
                if pvalue is None:
                    features_num.append(col)
                # Si pvalue no es None, verificar también la significación estadística
                elif p_val <= pvalue:
                    features_num.append(col)
    

    return features_num

def plot_features_num_regression(dataframe, target_col="", columns=[], umbral_corr=0, pvalue=None, max_pairplot_column=5):
    """
        Función que analiza la correlación de variables numéricas con la variable target. En el caso de que haya variables correladas
        pintará un pairplot con la comparativa de cada una de ellas.

        Argumentos:
            > dataframe: Dataframe con los datos
            > target_col: Columna target a analizar
            > columns: Columnas con las que buscar la correlación con la columna 'target'. En caso de no especificar
                        nada, se revisrán las variables numéricas que hay en el dataframe
            > umbral_corr: Umbral a partir del cual una columna se va a comparar con el target. Por defecto es 0
            > pvalue: Nivel de significación para el test de hipótesis
            > max_pairplot_column: Número de columnas a pintar. Debe ser mayor o igual a 2. Se define 5 como valor por defecto

        Retorna:
            > Parametro 1: Lista de las columnas que tienen correlación por encima de 'umbral_corr' con la variable target. En el caso de que 
                        haya algún error, se devuelve 'None'
            > Parametro 2: Matriz de correlación con las variables seleccionadas

    """
    final_columns = columns
    # Si no se pasa parámetro 'columns', extraemos todas las columnas numéricas
    if len(final_columns) == 0:
        final_columns = fnc.get_num_colums(dataframe, dataframe.columns)
    
    if not fnc.is_valid_numeric(dataframe, target_col, final_columns):
        return None
    
    # Borramos la columna target de la lista, en el caso de que exista
    if target_col in final_columns:
        final_columns.remove(target_col)

    # Comprobamos si finalmente hay columnas a analizar
    if len(final_columns) == 0:
        print("No se han especificado columnas en el parámetro 'columns' y el set de datos no contiene ninguna columna numérica (diferente al target)")
        return None

    # Verificamos que el número máximo de columnas a pintar es mayor que 2
    if max_pairplot_column < 2:
        print("El valor de la variable 'max_pairplot_column' debe ser mayor o igual a 2")
        return None
    
    corr_columns = fnc.get_corr_columns_num(dataframe, target_col, final_columns, umbral_corr, pvalue)

    # Comprobamos si hay columnas a analizar que correlan con el umbral especificado
    if len(corr_columns) == 0:
        print("No se han encontrado columnas de correlación con los criterios especificados")
        return None
    else:
        #Pintamos el pairplot
        sns.set_style = var.SNS_STYLE
        paint_columns = corr_columns
        while len(paint_columns) > 0:
            sns.pairplot(dataframe[[target_col] + paint_columns[0:max_pairplot_column-1]])
            paint_columns = paint_columns[max_pairplot_column-1:]

    return corr_columns