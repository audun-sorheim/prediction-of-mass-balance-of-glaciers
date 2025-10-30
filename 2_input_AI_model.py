import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import pandas as pd
from pathlib import Path

print(f"TensorFlow version: {tf.__version__}")

# Function to load pickle data
def load_data(pickle_file):
    """Load data from a pickle file."""
    return pd.read_pickle(pickle_file)

# Define feature and label columns
FEATURE_COLUMNS = ['TMPP', 'Annual_SF']
LABEL_COLUMN = 'MB_Year'

def split_features_labels(data, feature_columns, label_column):
    """Split data into features and labels."""
    features = data[feature_columns].values
    labels = data[label_column].values
    return features, labels

# Build the neural network model
def create_fnn_model(normalizer):
    """Create and compile the feedforward neural network model."""
    model = Sequential([
        normalizer,
        Dense(units=256, activation='sigmoid'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(units=128, activation='sigmoid'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(units=64, activation='sigmoid'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(units=32, activation='sigmoid'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(units=16, activation='sigmoid'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(
        loss='mean_absolute_error',
        optimizer=Adam(learning_rate=0.001),
        metrics=['mean_absolute_error']
    )
    return model

# Function to plot the training and validation loss
def plot_loss(history, title):
    """Plot training and validation loss."""
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('MAE [Mass Balance mm]')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to train and save the model for a specific dataset
def train_model_for_dataset(data_dir):
    base_dir = Path("C:/Users/audso/Documents/Fysikk og matematikk/8. semester/numerical methods in Glaciology/project")
    train_data_path = base_dir / data_dir / f"{data_dir.split('_')[-1]}_train_data.pkl"
    val_data_path = base_dir / data_dir / f"{data_dir.split('_')[-1]}_val_data.pkl"
    test_data_path = base_dir / data_dir / f"{data_dir.split('_')[-1]}_test_data.pkl"

    # Load the data
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)
    test_data = load_data(test_data_path)

    # Split the data
    train_features, train_labels = split_features_labels(train_data, FEATURE_COLUMNS, LABEL_COLUMN)
    val_features, val_labels = split_features_labels(val_data, FEATURE_COLUMNS, LABEL_COLUMN)
    test_features, test_labels = split_features_labels(test_data, FEATURE_COLUMNS, LABEL_COLUMN)
    
    # Normalize the features
    normalizer = Normalization(axis=1)
    normalizer.adapt(np.array(train_features))

    # Create the model
    fnn_model = create_fnn_model(normalizer)

    # Train the model
    history = fnn_model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=500,
        batch_size=32
    )

    # Save the model
    model_save_path = base_dir / data_dir / f"FNN_MB_{data_dir.split('_')[-1]}.keras"
    fnn_model.save(model_save_path)

    # Plot the loss
    plot_loss(history, title=f"Training and Validation Loss for {data_dir}")

# Example usage:
data_dir = ["data_EU", "data_NA", "data_SA", "data_HMA"]
train_model_for_dataset(data_dir[3])
