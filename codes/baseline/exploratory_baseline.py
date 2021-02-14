import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score
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
# describe dataset
describe = train.describe().T
# visualizar gráficos
plot_historgram_rul(train)
for sensor_name in sensor_names:
    # plot_sensor(train, sensor_name)
    plot_RUL(train, sensor_name)

# sacar sensores que no sirven dado que no aportan información a determinar RUL
drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
drop_labels = index_names+setting_names+drop_sensors
# data de entrenemaiento
x_train = train.drop(drop_labels, axis=1)
y_train = x_train.pop('RUL')
# Dado que los valores reales de RUL para el conjunto de prueba
# solo se proporcionan para el último ciclo de tiempo de cada motor,
# el conjunto de prueba se subdivide para representar el mismo
x_test = test.groupby('unit_nr').last().reset_index().drop(drop_labels, axis=1)
# clipped RUL: esta es constante en primera instancia posterior baja lineal
clip = 125
y_train_clipped = y_train.clip(upper=clip)
all(y_train.where(y_train <= clip, clip) == y_train_clipped)

# scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# SVM regression + clipped RUL + scaled
svr = SVR(kernel='linear')
svr.fit(x_train_scaled, y_train_clipped)

# predecir y evaluar
y_hat_train = svr.predict(x_train_scaled)
evaluate(y_train_clipped, y_hat_train, 'train')

y_hat_test = svr.predict(x_test_scaled)
evaluate(y_test, y_hat_test)

# Polinomio de segundo grado Las características de
# [a, b] se convierten en [1, a, b, a ^ 2, ab, b ^ 2]
poly = PolynomialFeatures(2)
x_train_transformed = poly.fit_transform(x_train_scaled)
x_test_transformed = poly.fit_transform(x_test_scaled)

print(x_train_scaled.shape)
print(x_train_transformed.shape)

# SVM regression + clipped RUL + engineered features
svr_f = SVR(kernel='linear')
svr_f.fit(x_train_transformed, y_train_clipped)

# predict and evaluate
y_hat_train = svr_f.predict(x_train_transformed)
evaluate(y_train_clipped, y_hat_train, 'train')

y_hat_test = svr_f.predict(x_test_transformed)
evaluate(y_test, y_hat_test)

# feature engineering + selection
select_features = SelectFromModel(svr_f, threshold='mean', prefit=True)
select_features.get_support()
feature_names = poly.get_feature_names()
print('features originales:\n', x_train.columns)
print('mejores features:\n',
      np.array(feature_names)[select_features.get_support()])
np.array(feature_names)[select_features.get_support()].shape

# hacer un support vector
svr = SVR(kernel='linear')
svr.fit(x_train_transformed[:, select_features.get_support()], y_train_clipped)

# predict and evaluate
y_hat_train = svr.predict(
    x_train_transformed[:, select_features.get_support()])
evaluate(y_train_clipped, y_hat_train, 'train')
# sacar label
y_hat_test = svr.predict(x_test_transformed[:, select_features.get_support()])
evaluate(y_test, y_hat_test)

# selección del mejor modelo
epsilon = [0.4, 0.3, 0.2, 0.1, 0.05]
for e in epsilon:
    svr = SVR(kernel='linear', epsilon=e)
    svr.fit(
        x_train_transformed[:, select_features.get_support()], y_train_clipped)
    # predict and evaluate
    y_hat = svr.predict(x_train_transformed[:, select_features.get_support()])
    mse = mean_squared_error(y_train_clipped, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_train_clipped, y_hat)
    print("epsilon:", e, "RMSE:", rmse, "R2:", variance)

# encuadramiento
svr = SVR(kernel='linear', epsilon=0.2)
svr.fit(x_train_transformed[:, select_features.get_support()], y_train_clipped)

# predict and evaluate
y_hat_train = svr.predict(
    x_train_transformed[:, select_features.get_support()])
evaluate(y_train_clipped, y_hat_train, 'train')

y_hat_test = svr.predict(x_test_transformed[:, select_features.get_support()])
evaluate(y_test, y_hat_test)
evaluate(y_test, y_hat_test)
