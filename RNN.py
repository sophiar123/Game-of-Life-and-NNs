import tensorflow as tf
import game_of_life.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

###### need to look into if its normalized correctly
X_train = game_of_life.X_train_normalized.reshape(1000, 10, 400)# input data with shape (batch_size, 10, 1)
y_train = game_of_life.Y_train_normalized.reshape(1000, 10, 400)# target data with shape (batch_size, 1)


# Define LSTM model
model = Sequential()


## this model has an input layer, 2 hidden layers, and an ouptut layer
## an LSTM model takes in a 3D array of shape(samples, timesteps, features)
    # a single sequence is a sample

# first layer is LSTM layer with 50 units (number of units can be adjusted according to performance)
# represents 1000 samples, 10 timesteps, 400 features (places on board)
# return false to only view final prediction, set to true if you want to see how it changes
# for every time step
model.add(LSTM(units=50, return_sequences=False, input_shape=(10, 400)))

# adds 2 more layers of 20 units each and stacks
    # since return sequence is true, output array is 3D
model.add(LSTM(units=20, return_sequences=True))
model.add(LSTM(units=20, return_sequences=True))

# dense layer with single output unit
model.add(Dense(units=1))

# will give an overview of our model 
model.summary()

#compile
model.compile(loss='binary_crossentropy', optimizer='adam')


model.fit(X_train, y_train, epochs=10)

model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

## plot in a graph
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

## calculates accuracy
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# # Make a prediction on a single input example
# example = ...
# prediction = model.predict(preprocess_data(example))
