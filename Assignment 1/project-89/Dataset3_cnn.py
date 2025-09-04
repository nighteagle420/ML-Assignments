import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Embedding
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_data = pd.read_csv('datasets/train/train_text_seq.csv')
val_data = pd.read_csv('datasets/valid/valid_text_seq.csv')

def preprocess_input(data):
    # Convert input_str to a 50x1 matrix (digit sequence as input to CNN)
    inputs = np.array([list(map(int, list(seq))) for seq in data['input_str']])
    return inputs

X_train = preprocess_input(train_data)
X_val = preprocess_input(val_data)

y_train = train_data['label'].values
y_val = val_data['label'].values

def build_cnn_model(input_shape):
    model = Sequential()
    
    model.add(Embedding(input_dim=10, output_dim=20, input_length=input_shape))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))  
    
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    
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
    
    model = build_cnn_model(input_shape)
    if percentage == 100:
    
        model.build(input_shape=(None, input_shape))  
        model.summary()
        total_params = model.count_params()  
        print(f"Total number of trainable parameters: {total_params}")
    
    model.fit(X_train_subset, y_train_subset, epochs=40, batch_size=32, verbose=0)
    
    y_val_pred = (model.predict(X_val) > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_val_pred)
    accuracies.append(accuracy)
    print(f"Accuracy after training on {percentage}% of the training data: {accuracy:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(subset_percentages, accuracies, marker='o', linestyle='--', color='b')
plt.title("Model Accuracy vs. Training Data Percentage")
plt.xlabel("Training Data Percentage")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(subset_percentages)
plt.ylim(0.6, 0.9)
plt.show()