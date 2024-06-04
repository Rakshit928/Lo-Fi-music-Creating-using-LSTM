import numpy as np
import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers

KERN_DATASET_PATH = "deutschl/erk"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

# durations are expressed in quarter length
ACCEPTABLE_DURATIONS = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]


def load_songs_in_kern(dataset_path):
    songs = []

    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):

    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # get interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    tranposed_song = song.transpose(interval)
    return tranposed_song



def build_autoencoder(input_dim, encoding_dim):
    input_layer = layers.Input(shape=(None, input_dim))
    
    # Encoder
    encoder = layers.LSTM(encoding_dim, return_sequences=True)(input_layer)
    
    # Decoder
    decoder = layers.LSTM(encoding_dim, return_sequences=True, go_backwards=True)(encoder)
    decoder = layers.Dense(input_dim, activation='softmax')(decoder)
    
    autoencoder = keras.Model(input_layer, decoder)
    autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
    
    return autoencoder

def encode_songs(songs, autoencoder, input_dim):
    encoded_songs = []
    for song in songs:
        input_seq = tf.one_hot(song, input_dim)
        encoded_seq = autoencoder.encoder(input_seq)
        encoded_songs.append(encoded_seq)
    
    return np.array(encoded_songs)


def preprocess(dataset_path):

    # load folk songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        song = transpose(song)

        encoded_song = encode_song(song)
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    songs = songs[:-1]

    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    mappings = {}

    songs = songs.split()
    vocabulary = list(set(songs))

    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    int_songs = []

    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    songs = songs.split()

    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length):
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    vocabulary_size = len(set(int_songs))
    # inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets



def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)

    # Load the song sequences
    with open(SINGLE_FILE_DATASET, "r") as fp:
        songs = fp.read().split()

    # Convert songs to integers
    songs = convert_songs_to_int(songs)

    # Get the input dimension (e.g., number of unique notes/symbols)
    input_dim = len(set(songs))

    # Build the autoencoder
    autoencoder = build_autoencoder(input_dim, encoding_dim=128)

    # Encode the songs using the autoencoder
    encoded_songs = encode_songs(songs, autoencoder, input_dim)

    # Generate training sequences from encoded songs
    inputs, targets = generate_training_sequences(encoded_songs, SEQUENCE_LENGTH)

    # One-hot encode the sequences
    inputs = keras.utils.to_categorical(inputs, num_classes=input_dim)
    targets = np.array(targets)

    # Train the LSTM model using encoded sequences
    model = build_model(input_dim, NUM_UNITS, LOSS, LEARNING_RATE, encoding_dim=128)
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save the trained model
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    main()


