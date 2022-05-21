import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
import math
from typing import List

import pandas_datareader as web

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
warnings.filterwarnings('error')
plt.style.use('fivethirtyeight')
pd.options.mode.chained_assignment = None

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
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.rmse_history = pd.DataFrame()
        self.training_data_len = 0
        self.lookback = 0
        self.test_predictions = None
        self.y_dataset = None
        self.y_column_name = None
        self.y_column_num = 0
        self.x_column_num = 0
        self.history = None
        self.actual_vs_preds = None

        # Set variables
        self.save_model = save_model
        self.max_accuracy = max_accuracy
        self.model_file_name = model_file_name
        self.input_csv = input_csv

        if input_csv is not None:
            self.df = pd.read_csv(self.input_csv)
            self.df_no_index = pd.read_csv(self.input_csv)
            #self.df.set_index("Date", inplace=True, drop=True)
            #self.df = self.df.filter(['Close'])

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
                self.model.add(LSTM(neurons, return_sequences=return_sequence, input_shape=(self.x_train.shape[1], self.x_train.shape[2]), dropout=dropout))
            else:
                self.model.add(LSTM(neurons, return_sequences=return_sequence, dropout=dropout))
        elif layer_type == "DENSE" or layer_type == "dense":
            self.model.add(Dense(neurons))

    def train_model_with_metrics(self, batch_size: int = 1, epochs: int = 10, verbose: int = 0):
        """
        Uses x_train and y_train arrays and trains the model
        Arguments:
        ----
        batch_size: {int} -- The batch size to be used for the model compiler
        epochs {int} -- how many times the training data set should be traversed in training
        """
        #print(self.x_train)
        # Compile the model before training
        self.model.compile(optimizer='adam', loss='mse')
        # Train the model
        train_rmse = []
        test_rmse = []
        for i in range(epochs):
            self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=batch_size, verbose=verbose)
            #self.model.reset_states()
            # evaluate model on train data
            x_full_data = self.df_no_index.filter([self.y_column_name[0]])
            x_train_data = x_full_data[:self.training_data_len]
            x_train_data = x_train_data[self.lookback:]
            x_test_data = x_full_data[self.training_data_len:]
            #print(x_test_data.shape)
            train = self.get_rmse(x_train_data, self.test_prediction(self.x_train))
            train_rmse.append(train[0])
            #self.model.reset_states()
            # evaluate model on test data
            train = self.get_rmse(x_test_data, self.test_prediction(self.x_test))
            test_rmse.append(train[0])
            #self.model.reset_states()
        self.rmse_history['train'], self.rmse_history['test'] = train_rmse, test_rmse
        return self.rmse_history
        

    def train_model(self, batch_size: int = 1, epochs: int = 10, verbose: int = 0):
        """
        Uses x_train and y_train arrays and trains the model
        Arguments:
        ----
        batch_size: {int} -- The batch size to be used for the model compiler
        epochs {int} -- how many times the training data set should be traversed in training
        """
        #print(self.x_train)
        # Compile the model before training
        self.model.compile(optimizer='adam', loss='mse')
        # Train the model
        #print(self.x_train)
        self.history = self.model.fit(self.x_train, self.y_train, validation_data = (self.x_test, self.y_test), batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=True)

    def normalize_data(self, y_column_name: List[str], lookback: int = 50, training_percent: float = 0.8):
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
        self.y_column_name = y_column_name
        self.y_column_num = len(y_column_name)
        self.x_column_num = len(self.df)
        self.lookback = lookback

        self.output_df = self._get_output_df(y_column_name=y_column_name)

        #convert train datasets to numpy arrays
        x_dataset = self.df.values
        y_dataset = self.output_df.values

        # Get the number of rows to train the model on and round up
        self.training_data_len = math.ceil(len(x_dataset) * training_percent)

        # Scale the data for normalization (advantagious to preprocess the data for input data to a neural network)
        x_scaled_data = self.scaler.fit_transform(x_dataset)
        y_scaled_data = self.scaler.fit_transform(y_dataset)

        # Create the training dataset 
        # Create the scaled training data set
        x_train_data = x_scaled_data[0:self.training_data_len, :]
        y_train_data = y_scaled_data[0:self.training_data_len, :]

        # Transform the train data sets into normalized sets with sequences and the expected future y value
        self.x_train, self.y_train = self._create_train_arrays(x_train_data=x_train_data, y_train_data=y_train_data, lookback=self.lookback)       

        ############################### Create the testing data set#########################
        # Create a new array containing scaled values from the remaining input csv values
        x_test_data = x_scaled_data[self.training_data_len - lookback: , :]

        # Create the data sets for x_test and y_test, convert them to numpy array and reshape them (same logic as _parse_sequence_array but for output)
        self.x_test, self.y_test = self._create_test_arrays(x_test_data=x_test_data, y_test_data=y_dataset, lookback=self.lookback)

    def test_prediction(self, input = None):
        """ 
        input the prediction test data and save the predictions after inverse scaling them
        """
        if input is None:
            # Get the models predicted price values
            self.test_predictions = self.model.predict(self.x_test)
            # Inverse scale the output predictions
            self.test_predictions = self.scaler.inverse_transform(self.test_predictions)
        else:
            # Get the models predicted price values
            self.test_predictions = self.model.predict(input)
            # Inverse scale the output predictions
            self.test_predictions = self.scaler.inverse_transform(self.test_predictions)

        return self.test_predictions

    def get_rmse(self, valid = None, preds = None) -> float:
        """ 
        calculate the Root Mean Squared Error and return that value
        """
        # Get the root mean squared error or RMSE
        if valid is None or preds is None:
            rmse = np.sqrt(np.mean(((self.test_predictions- self.y_test)**2)))
        else:
            rmse = np.sqrt(np.mean(((preds- valid)**2)))
        if self.save_model:
            model_name = 'C:/Users/sweil/OneDrive/Documents/Trading Bots/Machine Learning/LSTMTesting/Models/Model_version_' + str(rmse) + '.h5'
            self.model.save(model_name)
        return rmse

    def plot_loss(self):
        loss_per_epoch = self.history.history['loss']
        plt.plot(range(len(loss_per_epoch)),loss_per_epoch)

    def plot_fit(self):
        h1 = self.history.history
        #print(h1)
        plt.figure(figsize=(20,6))
        plt.subplot(1,3,1)
        ax = sns.lineplot(y=h1['val_accuracy'], x = range(len(h1['val_accuracy'])),label="val_acc",palette="binary")
        ax.set(xlabel='epochs', ylabel="val acc")
        ax = sns.lineplot(y=h1['accuracy'], x = range(len(h1['val_accuracy'])),label="acc",palette="flare")
        ax.set(xlabel='epochs', ylabel="acc")
        plt.title('Epochs vs Accuracy (Acc = '+str(self.test_accuracy)+')')

        plt.subplot(1,3,2)
        ax = sns.lineplot(x=h1['accuracy'], y = h1['val_accuracy'],label="acc vs val_acc",palette="binary",color="green",sort=False)
        ax.set(xlabel='acc', ylabel="val acc")
        plt.legend()
        plt.title('Validate vs Train Accuracy (Acc = '+str(self.test_accuracy)+')')

        plt.subplot(1,3,3)
        pdtmp1 = abs(pd.DataFrame(h1['loss'])-pd.DataFrame(h1['val_loss']))
        pdtmp1.fillna(0,inplace=True)
        ax = sns.lineplot(y=pdtmp1[0],x=range(0,len(h1['val_loss']),1),label="Loss Convergence",color="red")
        ax.set(xlabel='epochs',ylabel='abs(val loss - loss)')
        plt.legend()
        plt.title('Loss (Acc = '+str(self.test_accuracy)+')')
        plt.show()

    def plot_predictions_test(self):
        """ 
        plot the predictions compared to the actual values
        """
        # Visualize the two datasets
        #print(self.y_data)
        train = self.output_df[:self.training_data_len]
        actual_vs_preds = self.output_df[self.training_data_len:]

        actual_vs_preds['Predictions'] = self.test_predictions

        self.actual_vs_preds = actual_vs_preds

        plt.figure(figsize=(16,8))
        plt.title('Model')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD $', fontsize=18)
        plt.plot(train[self.y_column_name[0]])
        plt.plot(actual_vs_preds[[self.y_column_name[0], 'Predictions']])
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

    def get_actual_vs_preds(self):
        return self.actual_vs_preds

    def _create_train_arrays(self, x_train_data, y_train_data, lookback: int):
        # lookback variable is used to hold how many intervals of time will be used to predict the next interval
        x_train = []
        y_train = []

        for i in range(lookback, len(x_train_data)):
            x_train.append(x_train_data[i-lookback:i, :self.x_column_num])
            y_train.append(y_train_data[i, :self.x_column_num])
            # if i <= lookback:
            #     print(x_train)
            #     print(y_train)

        # Convert the arrays to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the x_train data since it will be 3 dimensional (3rd dimension is key for multifeature inputs)
        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], self.x_column_num))
        return x_train, y_train

    def _create_test_arrays(self, x_test_data, y_test_data, lookback: int):
        # lookback variable is used to hold how many intervals of time will be used to predict the next interval
        x_test = []
        y_test = []

        y_test = y_test_data[self.training_data_len:, :]
        for i in range(lookback, len(x_test_data)):
            x_test.append(x_test_data[i-lookback:i, :self.x_column_num])
        
        # Convert the arrays to numpy arrays
        x_test, y_test = np.array(x_test), np.array(y_test)

        return x_test, y_test

    def _get_output_df(self, y_column_name: List[str]):
        # lookback variable is used to hold how many intervals of time will be used to predict the next interval
        # Create a new dataframe with only the specified output column
        if self.y_column_num == 1:
            #output_df = self.df_no_index.filter([y_column_name[0]])
            output_df = self.df.filter([y_column_name[0]])
        elif self.y_column_num == 2:
            output_df = self.df.filter([[y_column_name[0], y_column_name[1]]])
        elif self.y_column_num == 3:
            output_df = self.df.filter([[y_column_name[0], y_column_name[1], y_column_name[2]]])
        else:
            output_df = self.df.filter([[y_column_name[0], y_column_name[1], y_column_name[2], y_column_name[3]]])
        
        return output_df
