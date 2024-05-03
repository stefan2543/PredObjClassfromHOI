import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


def split_data(data, labels, test_size=0.2):
    train_data, eval_data, train_labels, eval_labels = train_test_split(data, labels, test_size=test_size, random_state=42)
    return train_data, eval_data, train_labels, eval_labels

def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load preprocessed data and labels
with open('s1_data.pkl', 'rb') as f:
    train_data, train_labels_encoded, classes = pickle.load(f)


# One-hot encoding of labels
all_labels_categorical = to_categorical(all_labels_encoded)

# Split data
train_data, eval_data, train_labels, eval_labels = split_data(all_data, all_labels_categorical)

# Model building
model = build_model(train_data[0].shape[0], train_labels.shape[1])

# Train the model
model.fit(np.array(train_data), np.array(train_labels), epochs=10, batch_size=32, validation_data=(np.array(eval_data), np.array(eval_labels)))

# Evaluate the model
_, accuracy = model.evaluate(np.array(eval_data), np.array(eval_labels))
print(f"Evaluation Accuracy: {accuracy}")
