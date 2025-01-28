import functions as fnc
import variables as var

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, ttest_ind, f_oneway, stats

def describe_df(df):
    '''
    Devuelve el df con la descripción de tipo de dato por columna, 
    el tanto por ciento de valores nulos o missings, los valores 
    únicos y el porcentaje de cardinalidad.
    
    Argumentos:
    df (pd.DataFrame): Dataset del que se quiere extraer la descripción.

    Retorna:
    pd.DataFrame: Retorna en el mismo formato el información del argumento df.    
    '''
    df_resultado = pd.DataFrame([df.dtypes, df.isna().sum()*100, df.nunique(), round(df.nunique()/len(df) * 100, 2)]) # Cardinaliad y porcentaje de variación de cardinalidad
    df_resultado = df_resultado.rename(index= {0: "DATA_TYPE", 1: "MISSINGS (%)", 2: "UNIQUE_VALUES", 3: "CARDIN (%)"})
    return df_resultado.T


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
    -----
    Retorna:
    -----
    list
        Lista con las columnas categóricas significativas.
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
        df_types = tipifica_variables(dataframe, var.UMBRAL_CATEGORIA, var.UMBRAL_CONTINUA)
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