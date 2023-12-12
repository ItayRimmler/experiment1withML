from tensorflow import keras
from keras import layers

# The function that creates and tests the neural network:
def LetsGo(train, test, leng):

    # Creating a sequential API because we have 1 output for each input (image):
    model = keras.Sequential(
        [
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(160, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3),
            layers.Dropout(0.3)
        ]
    )

    # Compiling:
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(lr=0.01),
        metrics=["accuracy"]
    )

    # Fitting and testing:
    model.fit(train[0], train[1], batch_size=int(leng/10 + 1), epochs=10, verbose=2)
    model.evaluate(test[0], test[1], batch_size=int(leng/10 + 1), verbose=2)