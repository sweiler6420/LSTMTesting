import math
from typing import List

import pandas_datareader as web

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

class NeuralNetwork():

    def __init__(self, save_model: bool = True, max_accuracy: int = 8, model_file_name: str = None, input_csv: str = None) -> None:
        """
        Initalizes a new instance of the neuralnetwork and saves options.
        Arguments:
        ----
        savemodel {bool} -- variable that is used to determine to save the model or not
        max_accuracy {int} -- maximum amount for the RSME value to save a model if specified to save
        model_file_name {str} -- The path to the model to load. Loads the model without creating a model if using for predictions and not training a new model
        input_csv {str} -- the path to the csv used to train/test the model.
        """
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scale = None
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.training_data_len = 0
        self.lookback = 0
        self.test_predictions = None
        self.y_data = None
        self.y_column_name = None
        self.y_column_num = 0
        self.x_column_num = 0
        self.history = None

        # Set variables
        self.save_model = save_model
        self.max_accuracy = max_accuracy
        self.model_file_name = model_file_name
        self.input_csv = input_csv

        if input_csv is not None:
            self.df = pd.read_csv(self.input_csv)
            self.df.set_index("Date", inplace=True, drop=True)

        if model_file_name is None:
            self._createmodel()
        else:
            self.model = load_model(self.model_file_name)


    def _createmodel(self):
        """
        Creates a new sequential model and saves it.
        """
        self.model = Sequential()

    def add_layer(self, layer_type: str, neurons: int, return_sequence: bool = False, input_shape: bool = False, dropout: float = 0.0):
        """
        Adds a new layer to the model
        Arguments:
        ----
        layer_type: {str} -- variable that determines what type of layer to add, LSTM, DENSE...ETC
        neurons {int} -- how many neurons for this layer
        return_sequence {bool} -- do we want this layer to return the training sequence
        input_shape {Tuple(int, int)} -- The dimensions of the layer usually: (x_train.shape[1], and the # of output params)
        """
        #print(self.x_train.shape)
        #if the layer type is lstm create an lstm layer, same with Dense. Possible to add more options later
        if layer_type == "LSTM" or layer_type == "lstm":
            if input_shape:
                self.model.add(LSTM(neurons, return_sequences=return_sequence, input_shape=(self.x_train.shape[1],self.x_column_num), dropout=dropout))
            else:
                self.model.add(LSTM(neurons, return_sequences=return_sequence, dropout=dropout))
        elif layer_type == "DENSE" or layer_type == "dense":
            self.model.add(Dense(neurons))

    def train_model(self, batch_size: int = 1, epochs: int = 10):
        """
        Uses x_train and y_train arrays and trains the model
        Arguments:
        ----
        batch_size: {int} -- The batch size to be used for the model compiler
        epochs {int} -- how many times the training data set should be traversed in training
        """
        #print(self.x_train)
        # Compile the model before training
        self.model.compile(optimizer='adam', loss='mse', metrics='accuracy')
        # Train the model
        self.history = self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)

    def normalize_data(self, y_column_name: List[str], lookback: int = 50, training_percent: float = 0.8, scale: bool = True):
        """
        Create the dataframes and normalizes the data to the correct, type, size, and breaks them up into x_train, y_train, x_test, y_test arrays
        Arguments:
        ----
        y_column_name: {str} -- the output column name in the csv used to make the y_training, y_test. (What the model should predict)
        y_column_num: {int} -- the number of output columns we want
        lookback {int} -- how many rows in the past is used as input to guess the future
        training_percent {float} -- value between 0-1 that is used to decide how to split the training and testing up from the one input file
        """
        # Set column names and get the length of the list
        self.scale = scale
        self.y_column_name = y_column_name
        self.y_column_num = len(y_column_name)
        # Save the lookback number
        self.lookback = lookback

        # Create a new dataframe with only the specified output column
        if self.y_column_num == 1:
            self.y_data = self.df.filter([y_column_name[0]])
        elif self.y_column_num == 2:
            self.y_data = self.df.filter([[y_column_name[0], y_column_name[1]]])
        elif self.y_column_num == 3:
            self.y_data = self.df.filter([[y_column_name[0], y_column_name[1], y_column_name[2]]])
        else:
            self.y_data = self.df.filter([[y_column_name[0], y_column_name[1], y_column_name[2], y_column_name[3]]])
        
        x_train_dataset = self.df.values
        y_train_dataset = self.y_data.values

        # Get the number of rows to train the model on and round up
        self.training_data_len = math.ceil(len(x_train_dataset) * training_percent)
        # Scale the data for normalization (advantagious to preprocess the data for input data to a neural network)
        if scale:
            x_scaled_data = self.scaler.fit_transform(x_train_dataset)
            y_scaled_data = self.scaler.fit_transform(y_train_dataset)
        else:
            x_scaled_data = x_train_dataset
            y_scaled_data = y_train_dataset

        # Create the training dataset 
        # Create the scaled training data set
        x_train_data = x_scaled_data[0:self.training_data_len, :]
        y_train_data = y_scaled_data[0:self.training_data_len, :]

        self.x_column_num = len(x_train_data[0])

        # lookback variable is used to hold how many intervals of time will be used to predict the next interval
        for i in range(lookback, len(x_train_data)):
            self.x_train.append(x_train_data[i-lookback:i, :self.x_column_num])
            self.y_train.append(y_train_data[i, :self.x_column_num])
            # if i <= lookback:
            #     print(self.x_train)
            #     print(self.y_train)

        #print(str(len(self.x_train[0])))

        # Convert the x_train and y_train to numpy arrays
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        # Reshape the x_train data since it will be 3 dimensional
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], self.x_column_num))

        # Create the testing data set
        # Create a new array containing scaled values from the remaining csv
        test_data = x_scaled_data[self.training_data_len - lookback: , :]
        # Create the data sets for x_test and y_test
        self.y_test = y_train_dataset[self.training_data_len:, :]
        for i in range(lookback, len(test_data)):
            self.x_test.append(test_data[i-lookback:i, :self.x_column_num])

        # Convert these to numpy arrays
        self.x_test = np.array(self.x_test)

        # Reshape the x_test to 3 dimensional
        self.x_test= np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], self.x_column_num))
        # print(self.x_test)
        # print(self.y_test)

    def prediction_test(self):
        """ 
        input the prediction test data and save the predictions after inverse scaling them
        """
        # Get the models predicted price values
        self.test_predictions = self.model.predict(self.x_test)
        # Inverse scale the output predictions
        if self.scale:
            self.test_predictions = self.scaler.inverse_transform(self.test_predictions)
        return self.test_predictions

    def get_rmse(self) -> float:
        """ 
        calculate the Root Mean Squared Error and return that value
        """
        # Get the root mean squared error or RMSE
        rmse = np.sqrt(np.mean(((self.test_predictions- self.y_test)**2)))
        if self.save_model:
            model_name = 'C:/Users/sweil/OneDrive/Documents/Trading Bots/Machine Learning/LSTMTesting/Models/Model_version_' + str(rmse) + '.h5'
            self.model.save(model_name)
        return rmse

    def plot_loss(self):
        loss_per_epoch = self.history.history['loss']
        plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

    def plot_fit(self):
        h1 = self.history.history
        plt.figure(figsize=(20,6))
        plt.subplot(1,3,1)
        ax = sns.lineplot(y=h1['val_acc'], x = range(len(h1['val_acc'])),label="val_acc",palette="binary")
        ax.set(xlabel='epochs', ylabel="val acc")
        ax = sns.lineplot(y=h1['acc'], x = range(len(h1['val_acc'])),label="acc",palette="flare")
        ax.set(xlabel='epochs', ylabel="acc")
        plt.title('Epochs vs Accuracy (Acc-'+str(self.test_accuracy)+')')

    def plot_predictions_test(self):
        """ 
        plot the predictions compared to the actual values
        """
        # Visualize the two datasets
        #print(self.y_data)
        train = self.y_data[:self.training_data_len]
        valid = self.y_data[self.training_data_len:]
        valid['Predictions'] = self.test_predictions

        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD $', fontsize=18)
        plt.gcf().autofmt_xdate()
        plt.plot(train[self.y_column_name[0]])
        plt.plot(valid[[self.y_column_name[0], 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()

    def plot_chart(self):
        """ 
        plot the chart with the first y_column_name index since we assume that is the most important name for guessing
        """
        plt.figure(figsize=(16,8))
        plt.title('Close Price History')
        plt.plot(self.df[self.y_column_name[0]])
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Close Price USD $', fontsize=18)
        plt.show()