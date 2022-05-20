import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('stockframe5min.csv')
print(df)

# print("number of rows and columns in csv: " + str(df.shape))
# print("")

# Create a new dataframe with only the Close column
data = df.filter(['Close'])
print(data)
print(df)
# Convert the dataframe to a numpy array
dataset = data.values

# Get the number of rows to train the model on and round up
training_data_len = math.ceil(len(dataset) * 0.8)
#print(training_data_len)

# Scale the data for normalization (advantagious to preprocess the data for input data to a neural network)
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(dataset)
scaled_data = dataset
# print("Scaled_data looks like: ")
# print("")
#print(scaled_data)

# Create the training dataset 
# Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
# hist variable is used to hold how many intervals of time will be used to predict the next interval
hist = 60

for i in range(hist, len(train_data)):
    x_train.append(train_data[i-hist:i, 0])
    y_train.append(train_data[i, 0])
    # To visualize the first instance
    # if i <= hist:
    #     print(x_train)
    #     print(y_train)
    #     print("")

# Convert the x_rtain and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the x_train and y_train data since it will be 3 dimensional
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print("")
# print("x_train shape: " + str(x_train.shape))
# print ("")
# print("x_train looks like: ")
# print(x_train)