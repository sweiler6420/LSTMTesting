#This program uses and artifical recurrent neural network called Long Short Term Memory LSTM to try to predict a 5 minute interval in the stock market using the past 2 hours of data

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# # Get the stock qoute for testing
# df = web.DataReader('SPY', data_source = 'yahoo', start='2012-01-01', end='2022-05-11')

df = pd.read_csv('stockframe5min.csv')
print(df)

# Get the number of rows and columns in the dataframe
#print(df.shape)

#################### Visualize the closing price history #########################
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Time', fontsize=18)
# plt.ylabel('Close Price USD $', fontsize=18)
# plt.show()

# Create a new dataframe with only the Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on and round up
training_data_len = math.ceil(len(dataset) * 0.8)
#print(training_data_len)

# Scale the data for normalization (advantagious to preprocess the data for input data to a neural network)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
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
#print(x_train.shape)

# Import a trained model
#model = load_model('C:/Users/sweil/OneDrive/Documents/Trading Bots/Machine Learning/LSTMTesting/Models/Model_version_6.9791049310135405.h5')

# Build the LSTM Model
model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(x_train.shape[1],1)))  #200 for both LSTM and 100 Dense gave with 10 epochs gave 6.9
model.add(LSTM(200, return_sequences=False))
model.add(Dense(100))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=10)

# Create the testing data set
# Create a new array containing scaled values from the remaining csv
test_data = scaled_data[training_data_len - hist: , :]
# Create the data sets for x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(hist, len(test_data)):
    x_test.append(test_data[i-hist:i, 0])

# Convert these to numpy arrays
x_test = np.array(x_test)

# Reshape the x_test to 3 dimensional
x_test= np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values########################################
predictions = model.predict(x_test)

# Inverse scale the output predictions
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error or RMSE
rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

# # Save the model for persistence
model_name = 'C:/Users/sweil/OneDrive/Documents/Trading Bots/Machine Learning/LSTMTesting/5M_Models/Model_version_' + str(rmse) + '.h5'
model.save(model_name)


# Visualize the two datasets
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD $', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Show the actual price and predicted prices
print(valid)



