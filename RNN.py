import tensorflow as tf
import game_of_life
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Reshape to (1000, 10, 400)
X_train_reshaped = X_train.reshape(1000, 10, 400)
Y_train_reshaped = Y_train.reshape(1000, 10, 400)

# Normalize
X_train_normalized = X_train_reshaped.astype('float32') / 1.0
Y_train_normalized = Y_train_reshaped.astype('float32') / 1.0

# Define LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(10, 400)))

# adds 2 more layers of 20 units each and stacks
    # since return sequence is true, output array is 3D
model.add(LSTM(units=20, return_sequences=True))
model.add(LSTM(units=20, return_sequences=False))

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
model.fit(X_train_normalized, Y_train_normalized, epochs=10)

## plot results in a graph
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

"""
calculates model accuracy
"""
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


"""
Eventually we will want to implement this to see how model
predicts output after training
"""
# # Make a prediction on a single input example
# example = ...
# prediction = model.predict(preprocess_data(example))
