# Esta función recibe un dataframe, una argumento "target_col" con valor por defecto "", una lista de strings ("columns") 
# cuyo valor por defecto es la lista vacía, un valor de correlación ("umbral_corr", con valor 0 por defecto) y 
# un argumento ("pvalue") con valor "None" por defecto.

# Si la lista no está vacía, la función pintará una pairplot del dataframe considerando la columna designada por "target_col" 
# y aquellas incluidas en "column" que cumplan que su correlación con "target_col" es superior en valor absoluto a "umbral_corr", 
# y que, en el caso de ser pvalue diferente de "None", además cumplan el test de correlación para el nivel 1-pvalue de 
# significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores. 

# EXTRA: Se valorará adicionalmente el hecho de que si la lista de columnas a pintar es grande se pinten varios pairplot con un 
# máximo de cinco columnas en cada pairplot (siendo siempre una de ellas la indicada por "target_col")


# OK: Si la lista está vacía, entonces la función igualará "columns" a las variables numéricas del dataframe y se comportará como
#  se describe en el párrafo anterior.

# OK: De igual manera que en la función descrita anteriormente deberá hacer un check de los valores de entrada y comportarse como
#  se describe en el último párrafo de la función `get_features_num_regresion`

def plot_features_num_regression(dataframe, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
        <Descripción>

        Argumentos:
            > dataframe: Dataframe con los datos
            > target_col: Columna target a analizar
            > columns: Columnas con las que buscar la correlación con la columna 'target'. En caso de no especificar
                        nada, se revisrán las variables numéricas que hay en el dataframe
            > umbral_corr: Umbral a partir del cual una columna se va a comparar con el target
            > pvalue:

        Retorna:
        <return>

    """
    # Comprobamos si los parámetros son correcttos
    numeric_types = [var.TIPO_NUM_CONTINUA, var.TIPO_NUM_DISCRETA]
    if not fnc.is_valid_params(dataframe, target_col, columns, numeric_types, numeric_types):
        return None
    
    final_columns = columns
    # Si no se pasa parámetro 'columns', extraemos todas las columnas numéricas
    if len(final_columns) == 0:
        df_types = fnc.tipifica_variables(df, var.UMBRAL_CATEGORIA, var.UMBRAL_CONTINUA)
        final_columns = df_types[df_types[var.COLUMN_TIPO].isin(numeric_types)][var.COLUMN_NOMBRE].to_list()
    
    # Borramos la columna target de la lista, en el caso de que exista
    if target_col in final_columns:
        final_columns.remove(target_col)

    if len(final_columns) == 0:
        print("No se han especificado columnas en el parámetro 'columns' y el set de datos no contiene ninguna columna numérica (diferente al target)")
        return None

    #print(df[[target_col] +  final_columns].corr(numeric_only=True)[target_col])

    df_corr_matrix = dataframe[final_columns + [target_col]].corr(numeric_only=True)    
    corr_columns = df_corr_matrix.loc[df_corr_matrix[target_col] >= umbral_corr][target_col].index.to_list()    

    # Borramos la columna target de la lista, en el caso de que exista
    if target_col in corr_columns:
        corr_columns.remove(target_col)

    if len(corr_columns) == 0:
        print("No se han encontrado columnas de correlación con los criterios especificados")
        return None
    else:
        sns.set_style = var.SNS_STYLE
        sns.pairplot(dataframe[[target_col] +  corr_columns])

    return corr_columns