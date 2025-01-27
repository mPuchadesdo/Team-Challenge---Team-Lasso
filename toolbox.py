import functions as fnc
import variables as var

import pandas as pd
import seaborn as sns

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
            > pvalue: TODO: DescribirNone por defecto
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