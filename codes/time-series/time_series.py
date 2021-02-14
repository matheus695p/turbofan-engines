# import numpy as np
# from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LinearRegression
from src.turbo_fan_module import (read_turbofan_dataset, plot_historgram_rul,
                                  plot_RUL, add_linear_remaining_useful_life,
                                  evaluate)
# dirección local
dir_path =\
    r'C:\Users\Matheus\Desktop\turbofan-engines-predictive-mantenaince\data'
# nombres de indices y de columnas
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f's_{i}' for i in range(1, 22)]
# leer data
train_name = 'train_FD001'
test_name = 'test_FD001'
rule_name = 'RUL_FD001'
path = "./data"
# leer la data de entrenamiento
train = read_turbofan_dataset(path, train_name)
# leer la data de testeo
test = read_turbofan_dataset(path, test_name)
# label test
y_test = read_turbofan_dataset(path, rule_name)
# agregar RUL a el dataframe
train = add_linear_remaining_useful_life(train)
