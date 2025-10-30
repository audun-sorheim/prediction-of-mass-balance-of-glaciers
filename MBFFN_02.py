import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# load the Silvretta data
train_data = pd.read_pickle('Silvretta_train.pkl')
val_data = pd.read_pickle('Silvretta_val.pkl')
test_data = pd.read_pickle('Silvretta_test.pkl')

# in tesorflow the dependent variables (that's what we want to predict) is called "labels"
# and the independent variables (that's what we predict with) are called "features"

# we start with picking the yearly sums to predict
train_features = train_data.drop(['RYear', 'SYear', 'MBW', 'MBS', 'TMPP', 'Snow'], axis=1)
train_labels = train_features.pop('MBA')

# we do the same for the validation and test datasets
val_features = val_data.drop(['RYear', 'SYear', 'MBW', 'MBS', 'TMPP', 'Snow'], axis=1)
val_labels = val_features.pop('MBA')
test_features = test_data.drop(['RYear', 'SYear', 'MBW', 'MBS', 'TMPP', 'Snow'], axis=1)
test_labels = test_features.pop('MBA')

# don't forget, we need to normalize the input data to make the most of the activation function!
normalizer = tf.keras.layers.Normalization(axis=1)
normalizer.adapt(np.array(train_features))

# let's design the network architecture
fnn_model = keras.Sequential([
    normalizer,
    layers.Dense(units=36, activation='sigmoid'),
    layers.Dense(units=36, activation='sigmoid'),
    layers.Dense(units=36, activation='sigmoid'),
    layers.Dense(units=1)
])

# compile the model, or the network
# loss is what the network uses in the learning, metrics would be an alterantive evaluation metric.
fnn_model.compile(loss='mean_absolute_error',
                  # this is basically backpropagation
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=['mean_absolute_error'])

# we need to beat about MAE=500 mm from the statsmodels

# let's train the model
history = fnn_model.fit(
    train_features,
    train_labels,
    validation_data=(val_features, val_labels),
    epochs=250
)

# save the Fnn model to a file
fnn_model.save('FNN_MB_3L_sigmoid_03')

# define a convenient history plotting function
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MAE [Mass Balance mm]')
    plt.legend()
    plt.grid(True)

plt.figure(figsize=(10,10))
plt.title('FNN: 36(sigmoid)-36(sigmoid)-36(sigmoid)')
plot_loss(history)

plt.show()