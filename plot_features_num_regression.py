

# get_features_num_regresion
# Esta función recibe como argumentos un dataframe, el nombre de una de las columnas del mismo (argumento 'target_col'), que 
# debería ser el target de un hipotético modelo de regresión, es decir debe ser una variable numérica continua o discreta 
# pero con alta cardinalidad, además de un argumento 'umbral_corr', de tipo float que debe estar entre 0 y 1 y una variable 
# float "pvalue" cuyo valor debe ser por defecto "None".
# La función debe devolver una lista con las columnas numéricas del dataframe cuya correlación con la columna designada por "target_col" 
# sea superior en valor absoluto al valor dado por "umbral_corr". Además si la variable "pvalue" es distinta de None, sólo devolvera 
# las columnas numéricas cuya correlación supere el valor indicado y además supere el test de hipótesis con significación mayor o igual a 1-pvalue.
# La función debe hacer todas las comprobaciones necesarias para no dar error como consecuecia de los valores de entrada. Es decir 
# hará un check de los valores asignados a los argumentos de entrada y si estos no son adecuados debe retornar None y printar por pantalla 
# la razón de este comportamiento. Ojo entre las comprobaciones debe estar que "target_col" hace referencia a una variable numérica continua del dataframe.


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Esta función recibe un dataframe, una argumento "target_col" con valor por defecto "", una lista de strings ("columns") 
# cuyo valor por defecto es la lista vacía, un valor de correlación ("umbral_corr", con valor 0 por defecto) 
# y un argumento ("pvalue") con valor "None" por defecto.

# Si la lista no está vacía, la función pintará una pairplot del dataframe considerando la columna designada por "target_col" 
# y aquellas incluidas en "column" que cumplan que su correlación con "target_col" es superior en valor absoluto a "umbral_corr", 
# y que, en el caso de ser pvalue diferente de "None", además cumplan el test de correlación para el nivel 1-pvalue de 
# significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores. 

# EXTRA: Se valorará adicionalmente el hecho de que si la lista de columnas a pintar es grande se pinten varios pairplot con un 
# máximo de cinco columnas en cada pairplot (siendo siempre una de ellas la indicada por "target_col")
# Si la lista está vacía, entonces la función igualará "columns" a las variables numéricas del dataframe y se comportará como
#  se describe en el párrafo anterior.
# De igual manera que en la función descrita anteriormente deberá hacer un check de los valores de entrada y comportarse como
#  se describe en el último párrafo de la función `get_features_num_regresion`

def plot_features_num_regression(dataframe, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
        <Descripción>

        Argumentos:
            > dataframe:
            > target_col:
            > columns:
            > umbral_corr:
            > pvalue:

        Retorna:
        <return>
    """



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

    sns.set_style("whitegrid")
    
    if target_col not in dataframe.columns: # Control para ver si la target_col existe en el dataframe
        print(f"La columna '{target_col}' no existe en el dataframe")
        return None
    
    if dataframe[target_col].dtype != "float64": # Control para ver si la target_col es una variable numérica continua
         print(f"La columna '{target_col}' no es una variable numérica continua")
         return None
    
    for col in columns:
         if col not in dataframe.columns: # Control para ver si las columnas introducidas en columns existen en el dataframe
              print(f"La columna '{col}' no existe en el dataframe")
              return None
    
    sig_num_col = []
    sig_cat_col = []

    if columns == []: # Si no introducimos columnas categóricas, cogeremos las columnas numéricas continuas:
        num_col = []
        for col in dataframe.describe().columns.to_list():
            if len(dataframe[col].unique()) > 12: # Verificamos que no sea una categórica codificada numéricamente
                num_col.append(col)
        if dataframe[num_col].isna().sum().sum() != 0: # Verificamos si existen nulos para evitar errores al utilizar la correlación de pearson
             print(f"Error: existen nulos o NaN presentes en las variables numéricas de estudio")
             return None

        for col in num_col: # Usamos la correlación de pearson para ver si están relacionadas:
            p = pearsonr(dataframe[target_col], dataframe[col]).pvalue
            if p < pvalue: # Si la relación entre variables entra en la significación seleccionada, nos guardamos esa columna:
                sig_num_col.append(col)
        if target_col in sig_num_col: # Como nuestro target es una variable numérica, la quitamos de la lista
            sig_num_col.remove(target_col)
        print(f"Las columnas numéricas elegidas son: {sig_num_col}")

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
        print(f"Las columnas categóricas elegidas son: {sig_cat_col}")
    
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
            for cat in sig_cat_col:
                    # Crea el gráfico
                    plt.figure(figsize=(12, 8))
                    sns.histplot(x = target_col, hue = cat, data = dataframe, kde = True)
                    plt.title(f"Relación entre {cat} y {target_col}")
                    plt.xlabel(target_col)
                    plt.ylabel("")
                    plt.show();

        else:
            subplots = len(sig_cat_col) # Guardamos las filas que va a tener nuestra figura
            plt.figure(figsize=(20,15))
            for i, cat in enumerate(sig_cat_col):
                plt.subplot(subplots, 1, i+1) # Colocamos cada columna en una de las filas de nuestra figura
                sns.histplot(x = target_col, hue = cat, data = dataframe, kde = True)
                plt.title(f"Relación entre {cat} y {target_col}")
                plt.xlabel(cat)
                plt.ylabel("")
            plt.tight_layout()
            plt.show();