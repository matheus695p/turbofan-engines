# import numpy as np
# from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
# from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from src.turbo_fan_module import (read_turbofan_dataset,
                                  add_linear_remaining_useful_life,
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
# sensores que se habia determinado eliminar
drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
drop_labels = setting_names + drop_sensors
# data de entrenamiento
x_train = train.drop(drop_labels, axis=1)
x_test_interim = test.drop(drop_labels, axis=1)
# creamos varibles precedentes para ver diferencias temporales
remaining_sensors = x_train.columns.difference(index_names+['RUL'])
lag1 = [col + '_lag_1' for col in remaining_sensors]
# variables lageadas
x_train[lag1] = x_train.groupby('unit_nr')[remaining_sensors].shift(1)
x_train.dropna(inplace=True)
# intermin con lag
x_test_interim[lag1] = x_test_interim.groupby(
    'unit_nr')[remaining_sensors].shift(1)
x_test_interim.dropna(inplace=True)
# label
y_train = x_train.pop('RUL')
x_test = x_test_interim.groupby('unit_nr').last().reset_index()

# encuadramiento del baseline con lag
lm = LinearRegression()
lm.fit(x_train.drop(index_names, axis=1), y_train)
# evaluar
y_hat_train = lm.predict(x_train.drop(index_names, axis=1))
evaluate(y_train, y_hat_train, 'train')
y_hat_test = lm.predict(x_test.drop(index_names, axis=1))
evaluate(y_test, y_hat_test)
