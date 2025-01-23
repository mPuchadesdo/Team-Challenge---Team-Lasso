# OK: Esta función recibe un dataframe, una argumento "target_col" con valor por defecto "", una lista de strings ("columns") 
# cuyo valor por defecto es la lista vacía, un valor de correlación ("umbral_corr", con valor 0 por defecto) y 
# un argumento ("pvalue") con valor "None" por defecto.

# OK: Si la lista no está vacía, la función pintará una pairplot del dataframe considerando la columna designada por "target_col" 
# y aquellas incluidas en "column" que cumplan que su correlación con "target_col" es superior en valor absoluto a "umbral_corr", 

# PTE: y que, en el caso de ser pvalue diferente de "None", además cumplan el test de correlación para el nivel 1-pvalue de 
# significación estadística. 

# OK: La función devolverá los valores de "columns" que cumplan con las condiciones anteriores. 

# EXTRA: Se valorará adicionalmente el hecho de que si la lista de columnas a pintar es grande se pinten varios pairplot con un 
# máximo de cinco columnas en cada pairplot (siendo siempre una de ellas la indicada por "target_col")

# OK: Si la lista está vacía, entonces la función igualará "columns" a las variables numéricas del dataframe y se comportará como
#  se describe en el párrafo anterior.

# OK: De igual manera que en la función descrita anteriormente deberá hacer un check de los valores de entrada y comportarse como
#  se describe en el último párrafo de la función `get_features_num_regresion`

import functions as fnc
import variables as var

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Comprobamos si los parámetros son correcttos
    numeric_types = [var.TIPO_NUM_CONTINUA, var.TIPO_NUM_DISCRETA]
    if not fnc.is_valid_params(dataframe, target_col, columns, numeric_types, numeric_types):
        return None

    final_columns = columns
    # Si no se pasa parámetro 'columns', extraemos todas las columnas numéricas
    if len(final_columns) == 0:
        df_types = fnc.tipifica_variables(dataframe, var.UMBRAL_CATEGORIA, var.UMBRAL_CONTINUA)
        final_columns = df_types[df_types[var.COLUMN_TIPO].isin(numeric_types)][var.COLUMN_NOMBRE].to_list()
    
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

    
    df_corr_matrix = dataframe[final_columns + [target_col]].corr(numeric_only=True) 
    df_corr_matrix = df_corr_matrix.loc[df_corr_matrix[target_col] >= umbral_corr]
    corr_columns = df_corr_matrix[target_col].index.to_list()    

    # Borramos la columna target de la lista, en el caso de que exista
    if target_col in corr_columns:
        corr_columns.remove(target_col)

    # Comprobamos si hay columnas a analizar que correlan con el umbral especificado
    if len(corr_columns) == 0:
        print("No se han encontrado columnas de correlación con los criterios especificados")
        return None
    else:
        # Pintamos la matriz de correlación resultante
        print("Tabla de correlacion:")
        print(df_corr_matrix[target_col])

        #Pintamos el pairplot
        sns.set_style = var.SNS_STYLE
        paint_columns = corr_columns
        while len(paint_columns) > 0:
            sns.pairplot(dataframe[[target_col] + paint_columns[0:max_pairplot_column-1]])
            paint_columns = paint_columns[max_pairplot_column-1:]

    return corr_columns, df_corr_matrix[target_col]
    
