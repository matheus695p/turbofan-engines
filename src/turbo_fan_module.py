import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def ls(path):
    return os.listdir(path)


def add_remaining_useful_life(df):
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
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print(f'{label} set RMSE:{rmse}, R2:{variance}')


def read_turbofan_dataset(path, prefix):
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
