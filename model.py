import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from preprocessing import x_train, y_train, x_test, y_test

losses = []


class StopAtLossValue(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(StopAtLossValue, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        self.model.stop_training = logs["loss"] <= self.threshold
        losses.append(logs['loss'])


stopAtLossValue = StopAtLossValue(2000)

model = Sequential()

init = RandomUniform(minval=-0.001, maxval=0.001)

model.add(Dense(41))
model.add(Dense(128, kernel_initializer=init))
model.add(Dense(64, kernel_initializer=init))
model.add(Dense(41, kernel_initializer=init))
model.add(Dense(6, kernel_initializer=init))
model.add(Dense(2, activation='linear', kernel_initializer=init))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['rmse', 'acc', 'mae'])

early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=25)

for group in x_train.groups.keys():
    x_traindf = x_train.get_group(group).drop('ID', axis=1)
    y_traindf = y_train.get_group(group).drop('ID', axis=1)
    model.fit(x_traindf, y_traindf, batch_size=105, epochs=64, callbacks=[early_stop])

for group in x_test.groups.keys():
    x_testdf = x_test.get_group(group).drop('ID', axis=1)
    y_testdf = y_test.get_group(group).drop('ID', axis=1)
    model.fit(x_testdf, y_testdf, batch_size=105, epochs=64, callbacks=[early_stop])

preds = pd.DataFrame(model.predict(x_train.get_group('AL011852').drop('ID', axis=1)))
yhat = y_train.get_group('AL011852')

pd.DataFrame(preds).to_csv('preds.csv')
yhat.to_csv('yhat.csv')

pd.DataFrame(losses).plot()
plt.show()
