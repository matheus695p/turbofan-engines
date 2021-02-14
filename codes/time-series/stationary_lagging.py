import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif)
from src.turbo_fan_module import (read_turbofan_dataset, make_stationary,
                                  add_linear_remaining_useful_life, evaluate,
                                  search_nr_lags, add_lagged_variables,
                                  add_specific_lags)
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


adf, pvalue, usedlag, n_obs, critical_values, icbest = adfuller(
    train['s_7'], maxlag=1)
print('todas las unidades, una sola columna')
print(f'adf: {adf} \npvalue: {pvalue}')
print(f'Significant: {pvalue < 0.05}')


test_series = train.loc[train['unit_nr'] == 1, 's_7']
adf, pvalue, usedlag, n_obs, critical_values, icbest = adfuller(
    test_series, maxlag=1)
print('una sola unidad, una sola columna')
print(f'adf: {adf} \npvalue: {pvalue}')
print(f'Significant: {pvalue < 0.05}')

test_series = test_series.diff(1).dropna()
adf, pvalue, usedlag, n_obs, critical_values, icbest = adfuller(
    test_series, maxlag=1)
print('una sola unidad, una sola columna después de hacer la diferencia')
print(f'adf: {adf} \npvalue: {pvalue}')
print(f'Significant: {pvalue < 0.05}')


plt.subplots(3, 1, figsize=(15, 8))
plt.subplot(3, 1, 1)
plt.plot(train['s_7'])
plt.title('todas las columnas parecen ser estacionarias \n\
adf=-32.41, pvalue=0.0')
plt.subplot(3, 1, 2)
plt.plot(train.loc[train['unit_nr'] == 1, 's_7'])
plt.title('motores individuales no son estacionarios (engine unit_nr 1) \n\
adf=-2.28, pvalue=0.18')
plt.subplot(3, 1, 3)
plt.plot(train.loc[train['unit_nr'] == 1, 's_7'].diff(1).dropna())
plt.title(
    'differenced motores parecen ser estacionarios (engine unit_nr 1) \n\
        adf=-14.82, pvalue<0.01')
plt.tight_layout()
plt.show()


# código de prueba para encontrar los tiempos máximos de diferenciación
# de una serie temporal subconjunto a la unidad nr 1 y sensores de interés.
unit = train.loc[train['unit_nr'] == 1].drop(
    drop_labels + ['unit_nr', 'time_cycles', 'RUL'], axis=1)
for col in unit.columns:
    maxdiff = 0
    do = True
    adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(
        unit[col], maxlag=1)
    if pvalue < 0.05:
        do = False

    while do:
        maxdiff += 1
        adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(
            unit[col].diff(maxdiff).dropna(), maxlag=1)
        # if significant, stop differencing and testing for stationarity
        if pvalue < 0.05:
            do = False

    print(f"{col}: pvalue = {pvalue}, maxdiff = {maxdiff}")


intermediate_df = train.drop(drop_labels, axis=1)
intermediate_df = make_stationary(intermediate_df, remaining_sensors)

# buscar el número de lags con una función
search_nr_lags(intermediate_df, x_train, remaining_sensors, index_names,
               nr_of_lags=30)

# multicolinealidad
x_train = add_lagged_variables(intermediate_df, 14, remaining_sensors)
x_train = x_train.drop(index_names, axis=1)
vifs = {x_train.columns[i]: round(vif(x_train.values, i), 2)
        for i in range(len(x_train.columns))}
print(vifs)

# redefinir el dataframe
intermediate_df = train.drop(drop_labels, axis=1)
# estandarizar
scaler = StandardScaler()
scaler.fit(intermediate_df[remaining_sensors])
intermediate_df[
    remaining_sensors] = scaler.transform(intermediate_df[remaining_sensors])
# hacer el dataframe estacionario
intermediate_df = make_stationary(intermediate_df, remaining_sensors)
# agregar variables retardadas
x_train = add_lagged_variables(intermediate_df, 14, remaining_sensors)
x_train = x_train.drop(index_names, axis=1)
# verificar multicolinealidad
vifs = {x_train.columns[i]: round(vif(x_train.values, i), 2)
        for i in range(len(x_train.columns))}
print(vifs)

# buscar nuevamente el número correcto de lags
search_nr_lags(intermediate_df, x_train, remaining_sensors, index_names,
               nr_of_lags=30)


# dado que el resultado del proceso anterior fueron 9 lags
lags = 9
# prep data
x_train_interim = train.drop(drop_labels, axis=1)
x_train_interim[remaining_sensors] = scaler.transform(
    x_train_interim[remaining_sensors])
x_train_interim = make_stationary(x_train_interim, remaining_sensors)
x_train_interim = add_lagged_variables(
    x_train_interim, lags, remaining_sensors)
x_train_interim = sm.add_constant(x_train_interim)
x_train = x_train_interim.drop(index_names, axis=1)
y_train = x_train.pop("RUL")
x_test_interim = test.drop(drop_labels, axis=1)
x_test_interim[remaining_sensors] = scaler.transform(
    x_test_interim[remaining_sensors])
x_test_interim = make_stationary(x_test_interim, remaining_sensors)
x_test_interim = add_lagged_variables(x_test_interim, lags, remaining_sensors)
x_test_interim = x_test_interim.groupby('unit_nr').last().reset_index()
x_test_interim = sm.add_constant(x_test_interim)
x_test = x_test_interim.drop(index_names, axis=1)

# apply clipped RUL
model = sm.OLS(y_train.clip(upper=125), x_train)
model_fitted = model.fit()
# predict
y_hat_train = model_fitted.predict(x_train)
y_hat = model_fitted.predict(x_test)
# evaluate
evaluate(y_train.clip(upper=125), y_hat_train, 'train')
evaluate(y_test, y_hat)


# entrenemos y evaluemos con lags en especifico
specific_lags = [1, 2, 3, 4, 5, 10, 20]

# prep data
x_train_interim = train.drop(drop_labels, axis=1)
x_train_interim[remaining_sensors] = scaler.transform(
    x_train_interim[remaining_sensors])
x_train_interim = make_stationary(x_train_interim, remaining_sensors)
x_train_interim = add_specific_lags(
    x_train_interim, specific_lags, remaining_sensors)
x_train_interim = sm.add_constant(x_train_interim)
X_train = x_train_interim.drop(index_names, axis=1)
y_train = X_train.pop("RUL")

x_test_interim = test.drop(drop_labels, axis=1)
x_test_interim[remaining_sensors] = scaler.transform(
    x_test_interim[remaining_sensors])
x_test_interim = make_stationary(x_test_interim, remaining_sensors)
x_test_interim = add_specific_lags(
    x_test_interim, specific_lags, remaining_sensors)
x_test_interim = x_test_interim.groupby('unit_nr').last().reset_index()
x_test_interim = sm.add_constant(x_test_interim)
X_test = x_test_interim.drop(index_names, axis=1)

# fit model
model = sm.OLS(y_train.clip(upper=125), X_train)
model_fitted = model.fit()

# predict
y_hat_train = model_fitted.predict(X_train)
y_hat = model_fitted.predict(X_test)

# evaluate
evaluate(y_train.clip(upper=125), y_hat_train, 'train')
evaluate(y_test, y_hat)
