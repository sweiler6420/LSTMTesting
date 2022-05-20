from MLM.neuralnetwork import NeuralNetwork

neural_network = NeuralNetwork(
    save_model = True,
    max_accuracy = 10,
    input_csv = "stockframe.csv"
)

y_columns = ["Close"]

neural_network.normalize_data(y_column_name=y_columns, lookback=60, training_percent=0.8)

neural_network.add_layer(layer_type='lstm', neurons=100, return_sequence=True, input_shape=True, dropout=0.1)
neural_network.add_layer(layer_type='lstm', neurons=50, return_sequence=False, dropout=0.1)
neural_network.add_layer(layer_type='dense', neurons=25)
neural_network.add_layer(layer_type='dense', neurons=1)

neural_network.train_model(batch_size=1, epochs=30)

predictions = neural_network.prediction_test()

print(predictions)

rmse = neural_network.get_rmse()
print(rmse)

neural_network.plot_loss()

neural_network.plot_predictions_test()