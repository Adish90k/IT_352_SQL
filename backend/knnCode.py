# # from newCode import KNN
# import numpy  as np
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import MaxAbsScaler
# def maxScalerNormalization(X):
# 		scaler = MaxAbsScaler()
# 		scaler.fit(X)
# 		X_scaled = scaler.transform(X)
# 		return X_scaled


# import joblib
# from sklearn.preprocessing import MinMaxScaler

# loaded_model = joblib.load('knn_model.pkl')
# input_data = np.array([1625739415,3064463620,2117344770,6273,6273,80,39030,6,0,27,0])
# input_data_reshaped = input_data.reshape(1, -1)
# scaler = MinMaxScaler()

# # Fit the scaler on the input data and transform it
# normalized_input = scaler.fit_transform(input_data_reshaped)

# # Reshape the array into a 2D array with a single row and 11 columns

# # input_data = maxScalerNormalization(input_data)
# # Now you can use input_data_reshaped for prediction
# predictions = loaded_model.predict(normalized_input)
# print(predictions)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
import joblib

def maxScalerNormalization(X):
    scaler = MaxAbsScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled

# Load the trained model
loaded_model = joblib.load('knn_model.pkl')

# Prompt the user to enter input values from the terminal
input_str = input("The output is:")
input_data = np.array([int(x) for x in input_str.split(',')])

# Reshape the input data into a 2D array with a single row and multiple columns
input_data_reshaped = input_data.reshape(1, -1)

# Normalize the input data using MinMaxScaler
scaler = MinMaxScaler()
normalized_input = scaler.fit_transform(input_data_reshaped)

# Make predictions using the loaded model
predictions = loaded_model.predict(normalized_input)
print(predictions)
