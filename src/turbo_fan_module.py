import os
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def ls(path):
    return os.listdir(path)


def add_linear_remaining_useful_life(df):
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


def plot_historgram_rul(df):
    df_max_rul = df[['unit_nr', 'RUL']].groupby('unit_nr').max().reset_index()
    df_max_rul['RUL'].hist(bins=15, figsize=(15, 7))
    plt.title('Remining useful life: TurboFan')
    plt.xlabel('RUL')
    plt.ylabel('frequency')
    plt.show()


def plot_sensor(df, sensor_name):
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


def plot_RUL(df, sensor_name):
    fig, ax1 = plt.subplots(1, 1, figsize=(13, 5))
    for i in df['unit_nr'].unique():
        if (i % 10 == 0):
            signal = ax1.plot('RUL', sensor_name,
                              data=df.loc[df['unit_nr'] == i])
    plt.xlim(250, 0)
    plt.xticks(np.arange(0, 275, 25))
    ax1.set_ylabel(sensor_name, labelpad=20)
    ax1.set_xlabel('RUL', labelpad=20)
    ax2 = ax1.twinx()
    rul_line = ax2.plot('RUL', 'RUL', 'k', linewidth=4,
                        data=df.loc[df['unit_nr'] == 20])
    data = df.loc[df['unit_nr'] == 20]
    rul = data["RUL"]
    rul_line2 = ax2.plot(rul, rul.where(rul <= 125, 125),
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
