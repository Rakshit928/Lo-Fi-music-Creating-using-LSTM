import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH


OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model(output_units, num_units, loss, learning_rate):
    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"])

    model.summary()

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE, encoding_dim=128):
    # generate the training sequences
    input_dim = ... # Get the input dimension (e.g., number of unique notes/symbols)
    autoencoder = build_autoencoder(input_dim, encoding_dim)
    
    # Load and preprocess the songs
    songs = ...
    encoded_songs = encode_songs(songs, autoencoder, input_dim)
    
    # Split the encoded songs into inputs and targets
    inputs, targets = generate_training_sequences(encoded_songs, SEQUENCE_LENGTH)

    # build the network
    model = build_model(output_units, num_units, loss, learning_rate, encoding_dim)

    # train the model
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()