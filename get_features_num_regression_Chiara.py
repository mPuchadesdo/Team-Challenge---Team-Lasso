import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    '''
    Selecciona features numéricas basadas en su correlación con la variable target.
    
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
    if not isinstance(df[target_col].dtype, np.number):
        print(f"Error: La columna {target_col} debe ser numérica")
        return None
    n_unique = df[target_col].nunique()
    if n_unique < 15:  # umbral arbitrario para considerar alta cardinalidad
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
        if col != target_col and df[col].nunique() >= 15:
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
