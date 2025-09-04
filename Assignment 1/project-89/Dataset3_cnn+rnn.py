# Incase of any rare unexpected behaviour of the program please run the code again 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Embedding, LSTM
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_data = pd.read_csv('datasets/train/train_text_seq.csv')
val_data = pd.read_csv('datasets/valid/valid_text_seq.csv')

def preprocess_input(data):
    inputs = np.array([list(map(int, list(seq))) for seq in data['input_str']])
    return inputs

X_train = preprocess_input(train_data)
X_val = preprocess_input(val_data)

# Label encoding
y_train = train_data['label'].values
y_val = val_data['label'].values

# CNN+RNN model definition for sequence data (with ~10000 parameters)
def build_cnn_rnn_model(input_shape):
    model = Sequential()
    

    model.add(Embedding(input_dim=10, output_dim=8, input_length=input_shape))
    model.add(Conv1D(filters=32, kernel_size=4, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(LSTM(32, return_sequences=False))
    
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


subset_percentages = [20, 40, 60, 80, 100]
input_shape = X_train.shape[1] 
total_train_size = X_train.shape[0]
accuracies = []

for percentage in subset_percentages:
    subset_size = int(total_train_size * (percentage / 100))
    
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train[:subset_size]
    
    # Build iur classification model
    model = build_cnn_rnn_model(input_shape)
    
    # Aditional manual build before summary evaluation
    model.build(input_shape=(None, input_shape))
    model.summary()
    total_params = model.count_params()
    print(f"Total number of trainable parameters: {total_params}")
    
    model.fit(X_train_subset, y_train_subset, epochs=40, batch_size=32, verbose=0)
    
    y_val_pred = (model.predict(X_val) > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_val_pred)
    
    accuracies.append(accuracy)
    
    print(f"Accuracy after training on {percentage}% of the training data: {accuracy:.4f}") 

# Plot accuracy on each percentage of training data
plt.figure(figsize=(8, 6))
plt.plot(subset_percentages, accuracies, marker='o', linestyle='--', color='b')
plt.title("Model Accuracy vs. Training Data Percentage")
plt.xlabel("Training Data Percentage")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(subset_percentages)
plt.ylim(0.7, 0.9)
plt.show()