import numpy as np

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

X_train, y_train = generate_dataset(num_sequences=1000)

"""
converts to float and normalizes
"""
X_train_normalized = X_train.astype('float32') / 1.0
y_train_normalized = y_train.astype('float32') / 1.0

print("X_train_normalized:")
print(X_train_normalized)

print("\ny_train_normalized:")
print(y_train_normalized)
