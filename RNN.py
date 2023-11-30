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

def generate_sequence(size, density=0.5, num_steps=10):
    """
    Creates sequences of states by evolving them over
    multiple generations, saves each state and its output as pairs
    """
    initial_state = initialize_board(size, density)
    sequence = [initial_state]

    for _ in range(num_steps):
        initial_state = game_of_life_step(initial_state)
        sequence.append(initial_state)

    return sequence[:-1], sequence[1:]

def generate_dataset(num_sequences, size=20, density=0.5, num_steps=10):
    """
    Generates datasets to be used for training
    """
    X, y = [], []

    for _ in range(num_sequences):
        input_sequence, output_sequence = generate_sequence(size, density, num_steps)
        X.append(input_sequence)
        y.append(output_sequence)

    return np.array(X), np.array(y)

X_train, Y_train = generate_dataset(num_sequences=1000)

# Generate Test Dataset
X_test, Y_test = generate_dataset(num_sequences=200)

# Reshape and Normalize Test Data
X_test_reshaped = X_test.reshape(200, 10, 400)
Y_test_reshaped = Y_test.reshape(200, 10, 400)

# X_test_normalized = X_test_reshaped.astype('float32') / 1.0
# Y_test_normalized = Y_test_reshaped.astype('float32') / 1.0


# Reshape to (1000, 10, 400)
X_train_reshaped = X_train.reshape(1000, 10, 400)
Y_train_reshaped = Y_train.reshape(1000, 10, 400)

# Normalize
# X_train_normalized = X_train_reshaped.astype('float32') / 1.0
# Y_train_normalized = Y_train_reshaped.astype('float32') / 1.0

# Define LSTM model
model = Sequential()

model.add(LSTM(units=400, return_sequences=True, input_shape=(10, 400)))

# adds 2 more layers of 20 units each and stacks
    # since return sequence is true, output array is 3D
model.add(LSTM(units=200, return_sequences=True))
model.add(LSTM(units=200, return_sequences=True))

# dense layer with 400 output units for each cell
model.add(Dense(units=400, activation='sigmoid'))

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

"""
actually begins training the model with 10 epochs
"""
history = model.fit(X_train_reshaped, Y_train_reshaped, epochs=10)

## plot results in a graph
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

"""
calculates model accuracy
"""
trainScore = model.evaluate(X_train_reshaped, Y_train_reshaped, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(X_test_reshaped, Y_test_reshaped, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))



"""
Eventually we will want to implement this to see how model
predicts output after training
"""

# Step 1: Generate an Initial Game of Life Board
seed_board = initialize_board(size=20, density=0.5)

# Step 2: Repeat the Board State Across 10 Time Steps
seed_sequence = np.tile(seed_board, (10, 1, 1))

# Step 3: Reshape and Normalize the Seed Sequence
seed_sequence_reshaped = seed_sequence.reshape(1, 10, 400)
# seed_sequence_normalized = seed_sequence_reshaped.astype('float32') / 1.0

# Step 3: Use the Model to Predict Future States
predicted_sequence = model.predict(seed_sequence_reshaped)
threshold = 0.5
binary_predictions = (predicted_sequence > threshold).astype(int)

def visualize_sequence(sequence, title):
    plt.figure(figsize=(20, 20))
    for i in range(sequence.shape[0]):
        plt.subplot(1, sequence.shape[0], i + 1)
        
        plt.imshow(reshaped_board, cmap='binary')
        plt.title(f'Time Step {i + 1}')
        plt.axis('off')
    
    plt.suptitle(title)
    plt.show()


