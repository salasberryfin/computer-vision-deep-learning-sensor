from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


# Create the Sequential model
model = Sequential()

# 1st layer - Add a flatten layer
model.add(Flatten(input_shape=(32, 32, 3)))

# 2nd layer - Add a fully connected layer
model.add(Dense(100))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('relu'))

# 4th layer - Add a fully connected layer
model.add(Dense(60))

# 5th Layer - Add a ReLU activation layer
model.add(Activation('relu'))
