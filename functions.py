import pandas as pd
import variables as var

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


def is_valid_params(dataframe, target_col, columns, target_type=[], columns_type=[]):
    mensajes = []

    df_types = tipifica_variables(dataframe, var.UMBRAL_CATEGORIA, var.UMBRAL_CONTINUA)

    # Analisis variable target_col
    if target_col not in dataframe.columns: # Control para ver si 'target_col' existe en el dataframe
        mensajes.append(f"La columna target '{target_col}' no existe en el dataframe")
    else:
        if len(target_type) > 0: # Control para ver si 'target_col' es una variable del tipo especificado
            target_type_list = df_types[df_types[var.COLUMN_TIPO].isin(target_type)][var.COLUMN_NOMBRE].to_list() #Columnas del dataframe que son del tipo 'target_type'
            if not target_col in target_type_list:
                mensajes.append(f"La columna '{target_col}' no es una variable de tipo {target_type}")

    # Análisis de las columnas
    if len(columns_type) > 0:
        col_not_exist_list = []
        col_not_type_list = []

        column_type_list = df_types[df_types[var.COLUMN_TIPO].isin(columns_type)][var.COLUMN_NOMBRE].to_list() #Columnas del dataframe que son del tipo 'columns_type'

        for col in columns:
            if col not in dataframe.columns: # Control para ver si las columnas 'columns' existen en el dataframe
                col_not_exist_list.append(col)
            elif col not in column_type_list: # Control para ver si las columnas 'columns' son del tipo especificado 'columns_type'
                col_not_type_list.append(col)
        
        if len(col_not_exist_list) > 0:
            mensajes.append(f"Las siguientes columnas no existen en el dataframe: {col_not_exist_list}")
        if len(col_not_type_list) > 0:
            mensajes.append(f"Las siguientes columnas no son del tipo {columns_type}: {col_not_type_list}")

    for m in mensajes:
        print(m)

    return len(mensajes) == 0