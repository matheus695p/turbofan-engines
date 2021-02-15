import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxTimeVaryingFitter
from src.turbo_fan_module import (read_turbofan_dataset, kaplan_meier_curve,
                                  add_linear_remaining_useful_life)
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
# agregar una linea base de RUL
train['RUL'].clip(upper=125, inplace=True)

# sensores que se habia determinado eliminar
drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']

drop_labels = setting_names + drop_sensors
train.drop(labels=drop_labels, axis=1, inplace=True)
# sensores que quedaron
remaining_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12',
                     's_13', 's_14', 's_15', 's_17', 's_20', 's_21']


# El CoxTimeVaryingFitter que se usa más adelante requiere una columna de
# desglose. agreguemos esa columna que indica si los motores se averiaron (1)
# o aún están funcionando (0)
train['breakdown'] = 0
idx_last_record = train.reset_index().groupby(
    by='unit_nr')['index'].last()
train.at[idx_last_record, 'breakdown'] = 1

train['start'] = train['time_cycles'] - 1
tail = train.tail()


# introducir la detección correcta en el conjunto de entrenamiento
# Dado que todos los motores del conjunto de entrenamiento funcionan hasta
# fallar, introducimos artificialmente la censura correcta para permitir
# jugar con los datos en un entorno un poco más realista.

cut_off = 200
train_censored = train[train['time_cycles'] <= cut_off].copy()


# create kaplan meier curve
data = train_censored[index_names+['breakdown']].groupby('unit_nr').last()
kaplan_meier_curve(data)

# CoxTimeVaryingFitter
train_cols = index_names + remaining_sensors + ['start', 'breakdown']
predict_cols = ['time_cycles'] + remaining_sensors + ['start', 'breakdown']

ctv = CoxTimeVaryingFitter()
ctv.fit(train_censored[train_cols], id_col="unit_nr", event_col='breakdown',
        start_col='start', stop_col='time_cycles', show_progress=True,
        step_size=1)
ctv.print_summary()

plt.figure(figsize=(10, 5))
ctv.plot()
plt.show()
plt.close()

# Hacer predicciones y evaluar
df = train_censored.groupby("unit_nr").last()
df = df[df['breakdown'] == 0]
df_to_predict = df[df['breakdown'] == 0].copy()
# obtener motores del conjunto de datos que todavía están funcionando para
# que podamos predecir su RUL
predictions = ctv.predict_log_partial_hazard(df_to_predict[predict_cols])
predictions = pd.DataFrame(predictions, columns=["predictions"])
# predictions = predictions["predictions"]

df_last = train.groupby('unit_nr').last()
predictions['RUL'] = df_to_predict['RUL']


plt.figure(figsize=(15, 5))
plt.plot(predictions['RUL'], predictions['predictions'], '.b')
xlim = plt.gca().get_xlim()
plt.xlim(xlim[1], xlim[0])
plt.xlabel('RUL')
plt.ylabel('log_partial_hazard')
plt.show()


# now lets look at some hazard trajectories
x = train.loc[train['unit_nr'].isin(df_to_predict.index)]
x_unique = len(x['unit_nr'].unique())

plt.figure(figsize=(15, 5))
for i in range(1, x_unique, 2):
    x_sub = x.loc[x['unit_nr'] == i]
    predictions = ctv.predict_partial_hazard(x_sub)[0].values
    plt.plot(x_sub['time_cycles'].values, np.log(predictions))
plt.xlabel('time_cycles')
plt.ylabel('log_partial_hazard')
plt.show()
