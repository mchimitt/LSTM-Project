## Matthew Chimitt
## mmc200005
## Project
## Recurrent Neural Networks on Time Series Prediction

# imports
import pandas as pd
import numpy as np
from datetime import datetime as dt
from matplotlib import pyplot as plt, dates as mdates
from sklearn.metrics import mean_squared_error
from tabulate import tabulate


## Sigmoid Activation Function
def sigmoid(x):
  return 1/(1+np.exp(-x))



## Preprocessing
def preprocess(url, timesteps):
  df = pd.read_csv(url)

  # get the dataframe of the adjusted close price
  dfclose = df.filter(["Adj Close"])
  dfdate = df.filter(["Date"])

  # convert to numpy array
  data = dfclose.values

  # get the date values, convert to datetime object
  dfdate = pd.to_datetime(dfdate["Date"])

  # convert to numpy array
  dfdate = dfdate.values

  

  # split into train and test data
  train_limit = round(len(data)*.85)
  training_data = data[0:train_limit]
  training_dates = dfdate[0:train_limit]

  testing_data = data[train_limit:]
  testing_dates = dfdate[train_limit:]

  # create the X_train and Y_train nparrays
  X_train = []
  Y_train = []

  X_test = []
  Y_test = []

  # create the training and testing data by making the x train values be
  #   timesteps of the values before the y train
  for i in range(timesteps, len(training_data)):
    X_train.append(training_data[i-timesteps:i, 0])
    Y_train.append(training_data[i, 0])
  # do same for testing
  for i in range(timesteps, (len(testing_data))):
    X_test.append(testing_data[i-timesteps:i, 0])
    Y_test.append(testing_data[i, 0])

  # convert to np array
  X_train = np.array(X_train)
  Y_train = np.array(Y_train)
  # convert to np array
  X_test = np.array(X_test)
  Y_test = np.array(Y_test)


  # reshape the X_train and Y_train to be of proper shape for inputs
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  Y_train = np.reshape(Y_train, (Y_train.shape[0], 1))
  
  # do the same for the test X and y
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))


  return X_train, Y_train, X_test, Y_test, training_dates[timesteps:], testing_dates[timesteps:]



## LSTM CLASS!!
class LSTM:
  def __init__(self, epochs, hidden_units, timesteps, learning_rate, id):
    print("\nBeginning Trial " + str(id))
    print("Epochs: " + str(epochs) + ", Timesteps: " + str(timesteps) + ", Hidden Units: " + str(hidden_units) + ", Learning Rate: " + str(learning_rate))
    self.hidden_units = hidden_units
    self.timesteps = timesteps
    self.id = id
    self.epochs = epochs

    # call prepropcessing
    X_train, Y_train, X_test, Y_test, self.training_dates, self.testing_dates = preprocess('https://mchimitt.github.io/DELL_STOCKS.csv', self.timesteps)
    self.X = X_train
    self.y = Y_train


    # weights for the hidden state
    self.Wh = np.random.randn(self.hidden_units, self.hidden_units + self.X.shape[2])
    
    # weights for the output gate
    self.Wo = np.random.randn(self.hidden_units, self.hidden_units + self.X.shape[2])

    # weights for the forget gate
    self.Wf = np.random.randn(self.hidden_units, self.hidden_units + self.X.shape[2])

    # weights for the input gate
    self.Wi = np.random.randn(self.hidden_units, self.hidden_units + self.X.shape[2])

    # weights for the candidate value
    self.Wc = np.random.randn(self.hidden_units, self.hidden_units + self.X.shape[2])

    # weights for the predicted value
    self.Wy = np.random.randn(self.y.shape[1], self.hidden_units)

    # Biases
    self.bf = np.zeros((self.hidden_units, 1))
    self.bi = np.zeros((self.hidden_units, 1))
    self.bo = np.zeros((self.hidden_units, 1))
    self.bc = np.zeros((self.hidden_units, 1))




    # ANAYLSIS

    # CALLING THE TRAIN AND TESTING FUNCTIONS
    self.train(self.epochs, learning_rate)

    print("Showing the Prediction Stock Curve for the training data.\nClose the Graph in order to proceed with the code.")

    # showing the prediction graph on training data
    self.show_predicted_curve_training(self.predicted_outputs, self.training_outputs)

    print("Showing the Prediction Stock Curve for the testing data.\nClose the Graph in order to proceed with the code.")
    # Testing function
    outputs, testing_mse = self.predict(X_test, Y_test)
    self.show_predicted_curve_test(outputs, Y_test)

    print("Showing the Error Curve for the training data at each epoch.\nClose the Graph in order to proceed with the code.")
    # show the error curve
    self.show_error_curve()

    print("Showing the MSE Curve for the training data at each epoch.\nClose the Graph in order to proceed with the code.")
    # show the mse curve
    self.show_mse_curve()
    
    # print the results
    self.print_results()


  #########################
  ## FORWARD PROPAGATION ##
  #########################

  # called for every timestep, computes a forward pass and returns the 
  #   cell state, hidden state and predicted output at the given timestep
  def lstm_forward_cell(self, xt, ht_1, ct_1):
    # concatenate the previous hidden state and the current input vector 
    # (because common weights)
    self.input_and_prev = np.concatenate((ht_1, xt.reshape(1,1)), axis=0)

    # calculating the gates
    self.forget_gate = sigmoid(np.dot(self.Wf, self.input_and_prev) + self.bf)
    self.input_gate = sigmoid(np.dot(self.Wi, self.input_and_prev) + self.bi)
    self.output_gate = sigmoid(np.dot(self.Wo, self.input_and_prev) + self.bo)
    self.candidate_value = np.tanh(np.dot(self.Wc, self.input_and_prev) + self.bc)

    # calculate the cell state
    ct = self.forget_gate * ct_1 + self.input_gate * self.candidate_value

    # calculate the hidden state
    ht = self.output_gate * np.tanh(ct)

    # calculate y (predicted!!)
    yt = np.dot(self.Wy, ht)

    return ct, ht, yt

  def forward(self, input):
    # getting an input from the parameter
    x_data = self.X[input] 
    y_data = self.y[input]
    
    self.inputs = []

    # initializing the cell state and hidden state (ct and ht respectively)
    self.ht = np.zeros((self.hidden_units, 1)) 
    self.ct = np.zeros((self.hidden_units, 1)) 

    # for every timestep, call the forward pass
    for timestep in range(len(x_data)):
        self.ct, self.ht, yt = self.lstm_forward_cell(x_data[timestep], self.ht, self.ct)
        # pass in the data at the timestep to an inputs array (used in backpropagation)
        self.inputs.append(x_data[timestep].reshape(1,1))
    
    # calculate the error
    self.error = yt - y_data
    # add the error to the errors list
    # self.errors.append(self.error)
    
    # add the predicted and actual outputs to respective lists (for getting the mse)
    self.yt = yt
    self.predicted_outputs.append(yt.item())
    self.training_outputs.append(y_data)




  ##########################
  ## BACKWARD PROPAGATION ## - optimizing the weights
  ##########################

  def backward(self):
    # creating zeroed np arrays for every derivative that are the same dimensions as the original
    dWh = np.zeros_like(self.Wh)
    dWo = np.zeros_like(self.Wo)
    dWf = np.zeros_like(self.Wf)
    dWi = np.zeros_like(self.Wi)
    dWc = np.zeros_like(self.Wc)
    dWy = np.zeros_like(self.Wy)
    # creating the zeroed np arrays for the derivatives of the biases (same dimensions as original)
    dbf = np.zeros_like(self.bf)
    dbi = np.zeros_like(self.bi)
    dbo = np.zeros_like(self.bo)
    dbc = np.zeros_like(self.bc)

    # essentially ht_1 and ct_1, (or ht-1 and ct-1) but calling them next helped me understand what I was doing
    # previous is technically the next because in back propagation we go in reverse.
    dht_next = np.zeros_like(self.ht)
    dct_next = np.zeros_like(self.ct)

    # going backward (reversed range) through the timesteps
    for timestep in reversed(range(len(self.inputs))):
      xt = self.inputs[timestep]
      ht_1 = self.ht
      ct_1 = self.ct

      # Calculating the error gradient at the output layer
      dy = self.error
      dWy += np.dot(dy, self.ht.T)

      # Calculate the gradient with respect to ht (hidden state)
      dht = np.dot(self.Wy.T, dy) + dht_next

      # Calculate the gradient with respect to ct (cell state)
      dct = dht * self.output_gate * (1 - np.tanh(self.ct)**2) + dct_next

      # Calculate the gradient for each gate
      # gradient with respect to the forget gate
      dforget_gate = dct * ct_1 * self.forget_gate * (1 - self.forget_gate)
      # gradient with respect to the input gate
      dinput_gate = dct * self.candidate_value * self.input_gate * (1 - self.input_gate)
      # gradient with respect to the output gate
      doutput_gate = dht * np.tanh(self.ct) * self.output_gate * (1 - self.output_gate)
      # gradient with respect to the output gate
      dcandidate_value = dct * self.input_gate * (1 - np.tanh(self.candidate_value)**2)

      # Calculate the gradients for the weights
      dWf += np.dot(dforget_gate, self.input_and_prev.T)
      dWi += np.dot(dinput_gate, self.input_and_prev.T)
      dWo += np.dot(doutput_gate, self.input_and_prev.T)
      dWc += np.dot(dcandidate_value, self.input_and_prev.T)
      dWh[:, :-1] += np.dot(dforget_gate, ht_1.T)

      # do the same for the biases!
      dbf += np.sum(dforget_gate, axis=1, keepdims=True)
      dbi += np.sum(dinput_gate, axis=1, keepdims=True)
      dbo += np.sum(doutput_gate, axis=1, keepdims=True)
      dbc += np.sum(dcandidate_value, axis=1, keepdims=True)

      # Update the next cell and hidden state for the next iteration
      dht_next = np.dot(self.Wh[:, :-1].T, dforget_gate)
      dct_next = dct * self.forget_gate

    # gradient clipping (helps with exploding gradient problem!)
    dWf, dbf = np.clip(dWf, -1, 1), np.clip(dbf, -1, 1)
    dWi, dbi = np.clip(dWi, -1, 1), np.clip(dbi, -1, 1)
    dWc, dbc = np.clip(dWc, -1, 1), np.clip(dbc, -1, 1)
    dWo, dbo = np.clip(dWo, -1, 1), np.clip(dbo, -1, 1)

    # Update the weights using the weights equation
    self.Wh -= self.lr * dWh
    self.Wo -= self.lr * dWo
    self.Wf -= self.lr * dWf
    self.Wi -= self.lr * dWi
    self.Wc -= self.lr * dWc
    self.Wy -= self.lr * dWy


    # do the same with the biases using the weights equation
    self.bf -= self.lr * dbf
    self.bi -= self.lr * dbi
    self.bo -= self.lr * dbo
    self.bc -= self.lr * dbc



  ##############
  ## TRAINING ##
  ##############

  def train(self, epochs, learning_rate):
    # used to print the data
    # self.epochs_trained_with = epochs

    # training errors instantiation
    self.training_errors = []

    # getting the learning rate
    self.lr = learning_rate

    # creating a history of the mse values for each epoch
    self.mse_history = []

    for epoch in range(epochs):
        print("Epoch -- " + str(epoch))

        # storing the outputs for every forward pass (for mse calculations!!)
        self.predicted_outputs = []
        self.training_outputs = []

        # forward and backward passes for every single input (self.X.shape[0])
        for i in range(self.X.shape[0]):
            # forward and backward passes through the network
            self.forward(i)
            self.backward()

        # store the error
        self.training_errors.append(np.squeeze(np.absolute(self.error)))

        # convert predicted and training output values to np array and then reshape
        po = np.array(self.predicted_outputs)
        po = np.reshape(po, (po.shape[0]))
        to = np.array(self.training_outputs)
        to = np.reshape(to, (to.shape[0]))

        # now get the mse on the predicted outputs and training outputs(actual)
        self.training_mse = mean_squared_error(po, to)
        self.mse_history.append(self.training_mse)
        # print the MSE at each epoch
        print("MSE: " + str(self.training_mse))
  
  #############
  ## TESTING ##
  #############

  def predict(self, X, y):
    # set the training X and Y temp variables to be the X and y inputs 
    train_X = self.X
    train_Y = self.y

    # instantiate the testing_errors list
    self.testing_errors = []
    self.testing_mse_history = []
    # set X and y to be the new inputs and outputs for propagation
    self.X = X
    self.y = y
    
    # Instantiate the output list
    output = []
    # forward propagation for every timestep
    for i in range(len(X)):
      self.forward(i)
      # add the result to the outputs list
      output.append(self.yt.item())

    # get the errors
    self.testing_mse = mean_squared_error(output, y)
    self.testing_errors.append(np.squeeze(self.error))
    
    self.testing_mse_history.append(self.testing_mse)

    # set X and y to be the former temp values
    self.X = train_X
    self.y = train_Y

    # return!
    return output, self.testing_mse


  ##############
  ## ANALYSIS ##
  ##############

  def show_error_curve(self):
    plt.figure(figsize=(8,3))
    plt.title('Error Curve')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Training Error', fontsize=18)
    plt.plot(self.training_errors)
    plt.show()

  def show_mse_curve(self):
    plt.figure(figsize=(8,3))
    plt.title('MSE Curve')
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Training MSE', fontsize=18)
    plt.plot(self.mse_history)
    plt.show()

  def show_predicted_curve_training(self, predicted_outputs, training_outputs):
    plt.figure(figsize=(12,3))
    plt.title('Model on Training Data')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    # plt.xticks(self.training_dates)
    plt.plot(self.training_dates, predicted_outputs, label='Predicted')
    plt.plot(self.training_dates, training_outputs, label='Actual')
    plt.legend(labels = ('Predicted', 'Actual'))
    plt.show()


  def show_predicted_curve_test(self, outputs, Y_test):
    # showing the prediction graph on test data
    plt.figure(figsize=(8,3))
    plt.title('Model on Testing Data')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(self.testing_dates, outputs, label='Predicted')
    plt.plot(self.testing_dates, Y_test, label='Actual')
    plt.legend(labels = ('Predicted', 'Actual'))
    plt.show()

  def table_data(self):
    # if we have columns: Trial, Timesteps, Epochs, Learning Rate, Hidden Units, Training MSE, Testing MSE
    row_in_table = [str(self.id), str(self.timesteps), str(self.epochs), str(self.lr), str(self.hidden_units), str(self.training_mse), str(self.testing_mse)]
    return row_in_table

  def print_results(self):
    print("\n\nTrial " + str(self.id) + " Results:")
    print("Training MSE: " + str(self.training_mse))
    print("Testing MSE: " + str(self.testing_mse))
    print("\n\n\n")




## Main Method
if __name__ == "__main__":
  np.random.seed(42)
  
  # creating the table
  table = [["Trial", "Timesteps", "Epochs", "Learning Rate", "Hidden Units", "Training MSE", "Testing MSE"]]
  table_separator = ["----------------------", "----------------------", "----------------------", "----------------------", "----------------------", "----------------------", "----------------------"]

  ## THESE ARE THE EXPERIMENTS
  ## THE ONLY ONE UNCOMMENTED IS THE MODEL WITH THE BEST HYPER PARAMETERS
  ## UNCOMMENTING ALL OF THE EXPERIMENTS WILL TAKE ROUGHLY 5 MINUTES PER EXPERIMENT, SO I ONLY UNCOMMENTED THE BEST ONE

  # # TRIAL 1
  # # epochs=50, hidden_units=50, timesteps=25, learning_rate=0.005 
  # lstm = LSTM(50, 50, 25, 0.0005, 1)
  # table.append(lstm.table_data())
  # table.append(table_separator)

  # # TRIAL 2
  # # epochs=50, hidden_units=50, timesteps=15, learning_rate=0.005 
  # lstm2 = LSTM(50, 50, 15, 0.0005, 2)
  # table.append(lstm2.table_data())
  # table.append(table_separator)

  # # TRIAL 3
  # # epochs=50, hidden_units=50, timesteps=10, learning_rate=0.005
  # lstm3 = LSTM(50, 50, 10, 0.0005, 3)
  # table.append(lstm3.table_data())
  # table.append(table_separator)

  # # TRIAL 4
  # # epochs=50, hidden_units=50, timesteps=3, learning_rate=0.005
  # lstm4 = LSTM(50, 50, 3, 0.0005, 4)
  # table.append(lstm4.table_data())
  # table.append(table_separator)

  # # TRIAL 5
  # # epochs=50, hidden_units=50, timesteps=3, learning_rate=0.1
  # lstm5 = LSTM(50, 50, 3, 0.1, 5)
  # table.append(lstm5.table_data())
  # table.append(table_separator) 

  # # TRIAL 6
  # # epochs=50, hidden_units=50, timesteps=3, learning_rate=0.05
  # lstm6 = LSTM(50, 50, 3, 0.05, 6)
  # table.append(lstm6.table_data())
  # table.append(table_separator) 

  # # TRIAL 7
  # # epochs=50, hidden_units=50, timesteps=3, learning_rate=0.005
  # lstm7 = LSTM(50, 50, 3, 0.005, 7)
  # table.append(lstm7.table_data())
  # table.append(table_separator) 

  # # TRIAL 8
  # # epochs=50, hidden_units=50, timesteps=3, learning_rate=0.0005
  # lstm8 = LSTM(50, 50, 3, 0.0005, 8)
  # table.append(lstm8.table_data())
  # table.append(table_separator) 

  # # TRIAL 9
  # # epochs=50, hidden_units=50, timesteps=3, learning_rate=0.00005
  # lstm9 = LSTM(50, 50, 3, 0.00005, 9)
  # table.append(lstm9.table_data())
  # table.append(table_separator) 

  # # TRIAL 10
  # # epochs=50, hidden_units=200, timesteps=3, learning_rate=0.0005
  # lstm10 = LSTM(50, 200, 3, 0.0005, 10)
  # table.append(lstm10.table_data())
  # table.append(table_separator) 

  # # TRIAL 11
  # # epochs=50, hidden_units=100, timesteps=3, learning_rate=0.0005
  # lstm11 = LSTM(50, 100, 3, 0.0005, 11)
  # table.append(lstm11.table_data())
  # table.append(table_separator) 

  # # TRIAL 12
  # # epochs=50, hidden_units=50, timesteps=3, learning_rate=0.0005
  # lstm12 = LSTM(50, 50, 3, 0.0005, 12)
  # table.append(lstm12.table_data())
  # table.append(table_separator) 

  # # TRIAL 13
  # # epochs=50, hidden_units=25, timesteps=3, learning_rate=0.0005
  # lstm13 = LSTM(50, 25, 3, 0.0005, 13)
  # table.append(lstm13.table_data())
  # table.append(table_separator) 

  # # TRIAL 14
  # # epochs=50, hidden_units=10, timesteps=3, learning_rate=0.0005
  # lstm14 = LSTM(50, 10, 3, 0.0005, 14)
  # table.append(lstm14.table_data())
  # table.append(table_separator) 

  # # TRIAL 15
  # # epochs=50, hidden_units=5, timesteps=3, learning_rate=0.0005
  # lstm15 = LSTM(50, 5, 3, 0.0005, 15)
  # table.append(lstm15.table_data())
  # table.append(table_separator) 

  # # TRIAL 16
  # # epochs=500, hidden_units=25, timesteps=3, learning_rate=0.0005
  # lstm16 = LSTM(500, 25, 3, 0.0005, 16)
  # table.append(lstm16.table_data())
  # table.append(table_separator) 

  # # TRIAL 17
  # # epochs=400, hidden_units=25, timesteps=3, learning_rate=0.0005
  # lstm17 = LSTM(400, 25, 3, 0.0005, 17)
  # table.append(lstm17.table_data())
  # table.append(table_separator) 

  # # TRIAL 18
  # # epochs=200, hidden_units=25, timesteps=3, learning_rate=0.0005
  # lstm18 = LSTM(200, 25, 3, 0.0005, 18)
  # table.append(lstm18.table_data())
  # table.append(table_separator) 

  # # TRIAL 19
  # # epochs=100, hidden_units=25, timesteps=3, learning_rate=0.0005
  # lstm19 = LSTM(100, 25, 3, 0.0005, 19)
  # table.append(lstm19.table_data())
  # table.append(table_separator) 

  # # TRIAL 20
  # # epochs=50, hidden_units=25, timesteps=3, learning_rate=0.0005
  # lstm20 = LSTM(50, 25, 3, 0.0005, 20)
  # table.append(lstm20.table_data())
  # table.append(table_separator) 

  # TRIAL 21
  # epochs=200, hidden_units=50, timesteps=3, learning_rate=0.00005
  lstm21 = LSTM(200, 50, 3, 0.00005, 21)
  table.append(lstm21.table_data())
  table.append(table_separator) 

  # Print the table
  print("\n\n\nTABLE:\n")
  print(tabulate(table, headers='firstrow'))