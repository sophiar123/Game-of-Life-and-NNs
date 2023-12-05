import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import math

def initialize_board(size, density=0.5):
    """
        size: size of board in cells
        density: proportion of live cells
    Initializes game of life board 
    """
    return np.random.choice([0, 1], size=(size, size), p=[1 - density, density])

def game_of_life_step(board):
    """
    Calculates number of living neighbor cells,
    updates the board,and returns the new board
    as an integer array of 1 for live cells and 0 for dead
    """
    neighbors = sum(np.roll(np.roll(board, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))
    new_board = (neighbors == 3) | (board & (neighbors == 2))
    return new_board.astype(int)

def generate_sequence(size, density=0.5, num_steps=3):
    """
    Creates sequences of states by evolving them over
    3 generations, saves each state and its output as pairs
    """
    initial_state = initialize_board(size, density)
    sequence = [initial_state]

    for _ in range(num_steps):
        initial_state = game_of_life_step(initial_state)
        sequence.append(initial_state)

    return sequence[:-1], sequence[1:]

def generate_dataset(num_sequences, size=10, density=0.5, num_steps=3):
    """
    Generates datasets to be used for training over 
    a 10x10 board
    """
    X, y = [], []

    for _ in range(num_sequences):
        input_sequence, output_sequence = generate_sequence(size, density, num_steps)
        X.append(input_sequence)
        y.append(output_sequence)

    return np.array(X), np.array(y)

def preprocess_data():
    X_train, Y_train = generate_dataset(num_sequences=1000)

    # Generate Test Dataset
    X_test, Y_test = generate_dataset(num_sequences=200)

    # Reshape and Normalize Test Data
    X_test_reshaped = X_test.reshape(200, 3, 100)
    Y_test_reshaped = Y_test.reshape(200, 3, 100)

    X_test_normalized = X_test_reshaped.astype('float32') / 1.0
    Y_test_normalized = Y_test_reshaped.astype('float32') / 1.0


    # Reshape to (1000, 10, 400)
    X_train_reshaped = X_train.reshape(1000, 3, 100)
    Y_train_reshaped = Y_train.reshape(1000, 3, 100)

    # Normalize
    X_train_normalized = X_train_reshaped.astype('float32') / 1.0
    Y_train_normalized = Y_train_reshaped.astype('float32') / 1.0
    
    return (
        X_train_normalized, 
        Y_train_normalized, 
        X_test_normalized, 
        Y_test_normalized
    )

def generate_seed_seq():
    # Step 1: Generate an Initial Game of Life Board
    seed_sequence, _ = generate_sequence(size=10, density=0.5, num_steps=3)

    # Convert the sequence to a NumPy array
    seed_sequence_array = np.array(seed_sequence)

    # Reshape the sequence for LSTM input
    seed_sequence_reshaped = seed_sequence_array.reshape(1, 3, 100)
    seed_sequence_normalized = seed_sequence_reshaped.astype('float32') / 1.0
    return (seed_sequence, seed_sequence_normalized)

def visualize_predictions(initial_sequence, predicted_sequence, title):
    # Convert the sequences to NumPy arrays
    initial_sequence_array = np.array(initial_sequence)
    predicted_sequence_array = np.array(predicted_sequence)

    # Reshape the sequences for visualization
    initial_sequence_reshaped = initial_sequence_array.reshape(3, 10, 10)
    predicted_sequence_reshaped = predicted_sequence_array.reshape(3, 10, 10)
    
    initial_sequence_normalized = initial_sequence_reshaped.astype('float32') / 1.0
    predicted_sequence_normalized = predicted_sequence_reshaped.astype('float32') / 1.0

    # Plot the sequences
    plt.figure(figsize=(12, 6))

    # Plot the initial sequence
    plt.subplot(1, 2, 1)
    plt.imshow(initial_sequence_reshaped[0], cmap='binary')
    plt.title('Initial Sequence')
    plt.axis('off')

    # Plot the predicted sequence
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_sequence_reshaped[0], cmap='binary')
    plt.title('Predicted Sequence')
    plt.axis('off')

    plt.suptitle(title)
    plt.show()

"""
First architecture tested. 4 layer model
    layer 1: LSTM layer with 100 units
    layer 2: hidden layer with 50 units
    layer 3: hidden layer with 50 units
    layer 4: output layer with 100 units and sigmoid activation 
"""
def layer_4_model(): 
    def train_model():
        
        X_train_normalized, Y_train_normalized, X_test_normalized, Y_test_normalized = preprocess_data()
    
        # Define LSTM model
        model = Sequential()

        model.add(LSTM(units=100, return_sequences=True, input_shape=(3, 100)))

        # adds 2 more layers of 20 units each and stacks
        # since return sequence is true, output array is 3D
        model.add(LSTM(units=50, return_sequences=True))
        model.add(LSTM(units=50, return_sequences=True))

        # dense layer with 100 output units for each cell
        model.add(Dense(units=100, activation='sigmoid'))

        # will give an overview of our model 
        model.summary()

        """
        compiles model
        'binary_crossentropy': Used as a loss function for binary classification model.
            The binary_crossentropy function computes the cross-entropy loss between true
            labels and predicted labels.
        'adam':stochastic gradient descent method that is based
            on adaptive estimation of first-order and second-order moments.
        """
        model.compile(loss='binary_crossentropy', optimizer='adam')

        history = model.fit(X_train_normalized, Y_train_normalized, epochs=10)

        ## plot results in a graph
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        """
        calculates model accuracy
        """
        trainScore = model.evaluate(X_train_normalized, Y_train_normalized, verbose=0)
        print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
        testScore = model.evaluate(X_test_normalized, Y_test_normalized, verbose=0)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


        #save the model
        model.save('LSTM_4')
        
    train_model()
    model = tf.keras.models.load_model('LSTM_4')
    seed, seed_normalized = generate_seed_seq()
    predicted_sequence = model.predict(seed_normalized)
    print(predicted_sequence)

    # Threshold the predictions
    threshold = 0.27
    binary_predictions = (predicted_sequence > threshold).astype(int)

    # Visualize the Initial and Predicted Sequences
    visualize_predictions(seed, binary_predictions, title='Initial and Predicted Sequences')
    

"""
Second architecture tested. 2 layer model
    layer 1: LSTM layer with 100 units
    layer 2: output layer with 100 units and sigmoid activation 
"""
def layer_2_model(): 
    def train_model():
        
        X_train_normalized, Y_train_normalized, X_test_normalized, Y_test_normalized = preprocess_data()
    
        # Define LSTM model
        model = Sequential()

        model.add(LSTM(units=100, return_sequences=True, input_shape=(3, 100)))

        # dense layer with 100 output units for each cell
        model.add(Dense(units=100, activation='sigmoid'))

        # will give an overview of our model 
        model.summary()

        """
        compiles model
        'binary_crossentropy': Used as a loss function for binary classification model.
            The binary_crossentropy function computes the cross-entropy loss between true
            labels and predicted labels.
        'adam':stochastic gradient descent method that is based
            on adaptive estimation of first-order and second-order moments.
        """
        model.compile(loss='binary_crossentropy', optimizer='adam')

        history = model.fit(X_train_normalized, Y_train_normalized, epochs=10)

        ## plot results in a graph
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        """
        calculates model accuracy
        """
        trainScore = model.evaluate(X_train_normalized, Y_train_normalized, verbose=0)
        print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
        testScore = model.evaluate(X_test_normalized, Y_test_normalized, verbose=0)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


        #save the model
        model.save('LSTM_2')
        
    train_model()
    model = tf.keras.models.load_model('LSTM_2')
    seed, seed_normalized = generate_seed_seq()
    predicted_sequence = model.predict(seed_normalized)
    print(predicted_sequence)

    # Threshold the predictions
    threshold = 0.27
    binary_predictions = (predicted_sequence > threshold).astype(int)

    # Visualize the Initial and Predicted Sequences
    visualize_predictions(seed, binary_predictions, title='Initial and Predicted Sequences')
    
"""
Third architecture tested. 4 layer model with less units
    layer 1: LSTM layer with 10 units
    layer 2: hidden layer with 5 units
    layer 3: hidden layer with 5 units
    layer 4: output layer with 100 units and sigmoid activation 
"""
def layer_4v2_model(): 
    def train_model():
        
        X_train_normalized, Y_train_normalized, X_test_normalized, Y_test_normalized = preprocess_data()
    
        # Define LSTM model
        model = Sequential()
        
        model.add(LSTM(units=10, return_sequences=True, input_shape=(3, 100)))

        # since return sequence is true, output array is 3D
        model.add(LSTM(units=5, return_sequences=True))
        model.add(LSTM(units=5, return_sequences=True))

        # dense layer with 100 output units
        model.add(Dense(units=100, activation='sigmoid'))

        # will give an overview of our model 
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam')

        history = model.fit(X_train_normalized, Y_train_normalized, epochs=10)

        ## plot results in a graph
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        """
        calculates model accuracy
        """
        trainScore = model.evaluate(X_train_normalized, Y_train_normalized, verbose=0)
        print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
        testScore = model.evaluate(X_test_normalized, Y_test_normalized, verbose=0)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


        #save the model
        model.save('LSTM_4v2')
        
    train_model()
    model = tf.keras.models.load_model('LSTM_4v2')
    seed, seed_normalized = generate_seed_seq()
    predicted_sequence = model.predict(seed_normalized)
    print(predicted_sequence)

    # Threshold the predictions
    threshold = 0.2
    binary_predictions = (predicted_sequence > threshold).astype(int)
    print(binary_predictions)

    # Visualize the Initial and Predicted Sequences
    visualize_predictions(seed, binary_predictions, title='Initial and Predicted Sequences')

    
"""
Fourth architecture tested. 2 layer model with less units
    layer 1: LSTM layer with 10 units
    layer 2: output layer with 100 units and sigmoid activation 
"""
def layer_2v2_model(): 
    def train_model():
        
        X_train_normalized, Y_train_normalized, X_test_normalized, Y_test_normalized = preprocess_data()
    
        # Define LSTM model
        model = Sequential()
        
        model.add(LSTM(units=10, return_sequences=True, input_shape=(3, 100)))
        
        
        # dense layer with 10 output units
        model.add(Dense(units=100, activation='sigmoid'))

        # will give an overview of our model 
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam')

        history = model.fit(X_train_normalized, Y_train_normalized, epochs=10)

        ## plot results in a graph
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        """
        calculates model accuracy
        """
        trainScore = model.evaluate(X_train_normalized, Y_train_normalized, verbose=0)
        print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
        testScore = model.evaluate(X_test_normalized, Y_test_normalized, verbose=0)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


        #save the model
        model.save('LSTM_2v2')
        
    train_model()
    model = tf.keras.models.load_model('LSTM_2v2')
    seed, seed_normalized = generate_seed_seq()
    predicted_sequence = model.predict(seed_normalized)
    print(predicted_sequence)

    # Threshold the predictions
    threshold = 0.27
    binary_predictions = (predicted_sequence > threshold).astype(int)

    # Visualize the Initial and Predicted Sequences
    visualize_predictions(seed, binary_predictions, title='Initial and Predicted Sequences')
    
"""
Fifth architecture tested. 3 layer model
    layer 1: LSTM layer with 100 units
    layer 2: hidden layer with 50 units
    layer 3: output layer with 100 units and sigmoid activation 
"""
def layer_3_model(): 
    def train_model():
        
        X_train_normalized, Y_train_normalized, X_test_normalized, Y_test_normalized = preprocess_data()
    
        # Define LSTM model
        model = Sequential()
        
        model.add(LSTM(units=100, return_sequences=True, input_shape=(3, 100)))
        
        model.add(LSTM(units=50, return_sequences=True))
        
        # dense layer with 10 output units
        model.add(Dense(units=100, activation='sigmoid'))

        # will give an overview of our model 
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam')

        history = model.fit(X_train_normalized, Y_train_normalized, epochs=10)

        ## plot results in a graph
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        """
        calculates model accuracy
        """
        trainScore = model.evaluate(X_train_normalized, Y_train_normalized, verbose=0)
        print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
        testScore = model.evaluate(X_test_normalized, Y_test_normalized, verbose=0)
        print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


        #save the model
        model.save('LSTM_3')
        
    train_model()
    model = tf.keras.models.load_model('LSTM_3')
    seed, seed_normalized = generate_seed_seq()
    predicted_sequence = model.predict(seed_normalized)
    print(predicted_sequence)

    # Threshold the predictions
    threshold = 0.2
    binary_predictions = (predicted_sequence > threshold).astype(int)

    # Visualize the Initial and Predicted Sequences
    visualize_predictions(seed, binary_predictions, title='Initial and Predicted Sequences')

# layer_4_model()
# layer_2_model()
# layer_4v2_model()
# layer_2v2_model()
# layer_3_model()

