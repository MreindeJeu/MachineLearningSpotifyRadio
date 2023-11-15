# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:41:17 2023

@author: Danille
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow import keras

# Load your data into a DataFrame
# Change the names to match your data file and structure
df = pd.read_csv('C:/Users/Danille/Documents/minor/Block2/Tech/Test_machineLearning_audioclasivication/Data/features_3_sec.csv')

# Drop the "filename" column
df = df.drop(labels="filename", axis=1)

# Feature Extraction
class_list = df.iloc[:, -1]
convertor = LabelEncoder()
y = convertor.fit_transform(class_list)
print(y)

# Feature Scaling
fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype=float))

# Deviding data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(len(y_train))
print(len(y_test))

def trainmodel(model, epochs, optimizer):
    batch_size = 128
    # callback = myCallback()
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"]
                  )
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size)
    return history

def plotValidate(history):
    print("Validation Accuracy", max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12, 6))
    plt.show()

model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(10, activation='softmax'),
])

# Training the Model
model_history = trainmodel(model=model, epochs=600, optimizer='adam')

# Model Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
print("The Test Loss is:", test_loss)
print("The Best Test Accuracy is:", test_acc * 100)

# Save the trained model
model.save("audio_classification_model.h5")
