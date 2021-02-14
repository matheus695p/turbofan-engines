import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score


def ls(path):
    """
    lista archivos en un directorio
    Parameters
    ----------
    path : string
        path del directorio, desde el de ejecución.
    Returns
    -------
    list
        lista con los archivos en el escritorio.
    """
    return os.listdir(path)


def add_linear_remaining_useful_life(df):
    """
    Obtener RUL: remaining useful life de cada motor de manera lineal
    decreciente como una primera aproximación
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    result_frame : TYPE
        DESCRIPTION.

    """
    # obtener el total de numeros de ciclos de cada unidad
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    # hacer el merge con el dataset original
    result_frame = df.merge(max_cycle.to_frame(
        name='max_cycle'), left_on='unit_nr', right_index=True)
    # calcular remaining useful life para cada fila
    result_frame["RUL"] = result_frame["max_cycle"] - \
        result_frame["time_cycles"]
    # sacar la columna que ya no es más necesaria
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def evaluate(y_true, y_hat, label='test'):
    """
    Evaluar según las predicciones del modelo entrenado
    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    y_hat : TYPE
        DESCRIPTION.
    label : TYPE, optional
        DESCRIPTION. The default is 'test'.

    Returns
    -------
    None.

    """
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print(f'{label} set RMSE:{rmse}, R2:{variance}')


def read_turbofan_dataset(path, prefix):
    """
    Función para leer los archivos turbofan como .txt desde un directorio dado
    Parameters
    ----------
    path : string
        path al directorio.
    prefix : string
        prefix del archivo buscado.
    Returns
    -------
    data : dataframe
        data cargada.

    """
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f's_{i}' for i in range(1, 22)]
    space = '\s+'
    col_names = index_names + setting_names + sensor_names
    data = pd.DataFrame()
    # read data
    files = ls(path)
    for file in files:
        if prefix in file:
            dir_path = path + "/" + file
            print(dir_path)
            if "RUL" in file:
                state = pd.read_csv((dir_path),
                                    sep=space, header=None, names=['RUL'])
            else:
                state = pd.read_csv((dir_path),
                                    sep=space, header=None, names=col_names)
            data = pd.concat([data, state])
    data.reset_index(drop=True, inplace=True)
    return data


def plot_historgram_rul(df):
    """
    Plot de la vida remanente
    Parameters
    ----------
    df : df
        dataframe de la vida remanente.
    Returns
    -------
    Gráfica de la vida remanente
    """
    df_max_rul = df[['unit_nr', 'RUL']].groupby('unit_nr').max().reset_index()
    df_max_rul['RUL'].hist(bins=15, figsize=(15, 7))
    plt.title('Remining useful life: TurboFan')
    plt.xlabel('RUL')
    plt.ylabel('frequency')
    plt.show()


def plot_sensor(df, sensor_name):
    """
    Plot de la evolución de valores de un sensor en función de la vida
    remanente del activo
    Parameters
    ----------
    df : df
        dataframe con los datos de los sensores.
    sensor_name : string
        nombre del sensor.
    Returns
    -------
    Gráfica de la evolución del sensor
    """
    plt.figure(figsize=(13, 5))
    for i in df['unit_nr'].unique():
        if (i % 10 == 0):
            plt.plot('RUL', sensor_name, data=df[df['unit_nr'] == i])
    # plt.xlim(250, 0)
    plt.xticks(np.arange(0, 275, 25))
    plt.title(f'Remining useful life: {sensor_name}')
    plt.ylabel(sensor_name)
    plt.xlabel('Remaining Use fulLife')
    plt.show()


def plot_RUL(df, sensor_name, constant=125):
    """
    Agregar la parte constante de una vida remanente.
    Parameters
    ----------
    df : df
        dataframe de trabajo.
    sensor_name : string
        nombre del sensor en el cual se quiere ver la evolución.
    constant : int
        ¿dónde termina la parte lineal y empieza la degradación severa del
        activo?
    Returns
    -------
    Gráfica de la evolución de RUL con la parte constante al principio
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(13, 5))
    for i in df['unit_nr'].unique():
        if (i % 10 == 0):
            signal = ax1.plot('RUL', sensor_name,
                              data=df.loc[df['unit_nr'] == i])
    # plt.xlim(250, 0)
    plt.xticks(np.arange(0, 275, 25))
    ax1.set_ylabel(sensor_name, labelpad=20)
    ax1.set_xlabel('RUL', labelpad=20)
    ax2 = ax1.twinx()
    rul_line = ax2.plot('RUL', 'RUL', 'k', linewidth=4,
                        data=df.loc[df['unit_nr'] == 20])
    data = df.loc[df['unit_nr'] == 20]
    rul = data["RUL"]
    rul_line2 = ax2.plot(rul, rul.where(rul <= constant, constant),
                         '--g', linewidth=4, label='clipped_rul')
    ax2.set_ylabel('RUL', labelpad=20)
    # ax2.set_ylim(0, 250)
    ax2.set_yticks(
        np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 6))
    ax1.set_yticks(
        np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 6))
    ax1.set_title(f'Remining useful life: {sensor_name}')
    lines = signal+rul_line+rul_line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc=0)
    plt.show()


def add_lagged_variables(df_input, nr_of_lags, columns):
    """
    Agregar variables pasadas del dataframe con el que se esta trabajando
    Parameters
    ----------
    df_input : df
        dataframe de test o train.
    nr_of_lags : int
        número de frames de la evolución que quirees ir hacia atrás.
    columns : list
        lista de columas a las cuales agregar variables lagged.
    Returns
    -------
    df : df
        dataframe con las variables agregadas.

    """
    df = df_input.copy()
    for i in range(nr_of_lags):
        lagged_columns = [col + f'_lag_{i+1}' for col in columns]
        df[lagged_columns] = df.groupby('unit_nr')[columns].shift(i+1)
    df.dropna(inplace=True)
    return df


def add_specific_lags(df_input, list_of_lags, columns):
    """
    Agregar variables pasadas del dataframe con el que se esta trabajando
    Parameters
    ----------
    list_of_lags : list
        columnas que lagear.
    df_input : df
        dataframe de test o train.
    columns : list
        lista de columas totales
    Returns
    -------
    df : df
        dataframe con las variables agregadas.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df = df_input.copy()
    for i in list_of_lags:
        lagged_columns = [col + f'_lag_{i}' for col in columns]
        df[lagged_columns] = df.groupby('unit_nr')[columns].shift(i)
    df.dropna(inplace=True)
    return df


def find_max_diff(series):
    """
    código de prueba para encontrar los tiempos máximos de diferenciación
    de una serie temporal subconjunto a la unidad nr 1 y sensores de interés.

    Parameters
    ----------
    series : series
        serie del sensor a analizar.
    Returns
    -------
    maxdiff : TYPE
        DESCRIPTION.

    """
    maxdiff = 0
    do = True
    adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(
        series, maxlag=1)
    if pvalue < 0.05:
        do = False

    while do:
        maxdiff += 1
        adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(
            series.diff(maxdiff).dropna(), maxlag=1)
        # if significant, stop differencing and testing for stationarity
        if pvalue < 0.05:
            do = False
    return maxdiff


def make_stationary(df_input, columns):
    """
    código de prueba para encontrar los tiempos máximos de diferenciación
    de una serie temporal subconjunto a la unidad nr 1 y sensores de interés.
    Parameters
    ----------
    df_input : df
        dataframe de entrada.
    columns : list
        lista de columnas de sensores.

    Returns
    -------
    df : df
        dataframe estacionario.
    """
    df = df_input.copy()
    for unit_nr in range(1, df['unit_nr'].max()+1):
        for col in columns:
            maxdiff = find_max_diff(df.loc[df['unit_nr'] == unit_nr, col])
            if maxdiff > 0:
                df.loc[df['unit_nr'] == unit_nr,
                       col] = df.loc[
                           df['unit_nr'] == unit_nr, col].diff(maxdiff)
    df.dropna(inplace=True)
    return df


def search_nr_lags(intermediate_df, x_train, remaining_sensors, index_names,
                   nr_of_lags=30):
    """
    Encontrar el número correcto de lags que son necesarios
    Parameters
    ----------
    intermediate_df : TYPE
        DESCRIPTION.
    x_train : TYPE
        DESCRIPTION.
    remaining_sensors : TYPE
        DESCRIPTION.
    index_names : TYPE
        DESCRIPTION.
    nr_of_lags : TYPE, optional
        DESCRIPTION. The default is 30.

    Returns
    -------
    metrics : TYPE
        DESCRIPTION.
    result : TYPE
        DESCRIPTION.

    """
    # buscando el número correcto de lags para agregar como variables
    metrics = pd.DataFrame(columns=['rmse', 'AIC', 'BIC'])
    for i in range(0, nr_of_lags+1):
        x_train = add_lagged_variables(intermediate_df, i, remaining_sensors)
        x_train = x_train.drop(index_names, axis=1)
        y_train = x_train.pop('RUL')
        model = sm.OLS(y_train, sm.add_constant(x_train.values))
        result = model.fit()
        metrics = metrics.append(pd.DataFrame(
            data=[[np.sqrt(result.mse_resid), round(
                result.aic, 2), round(result.bic, 2)]],
            columns=['rmse', 'AIC', 'BIC']),
            ignore_index=True)
    print(metrics)
    print(result.summary())
    metrics["AIC_diff"] = np.abs(metrics["AIC"].diff())
    metrics["AIC_diff"] = np.abs(metrics["AIC_diff"].diff())

    plt.figure(figsize=(15, 5))
    # plot the difference to see where it flattens out
    plt.plot(metrics['AIC'].diff(), marker='.')
    plt.plot(14, metrics['AIC'].diff()[14], '.r')
    plt.xlabel("Nr de lags")
    plt.ylabel("AIC tasa de cambio")
    plt.title("AIC: grid search")
    plt.show()
    plt.close()
