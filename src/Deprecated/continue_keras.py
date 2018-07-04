from keras.callbacks import ModelCheckpoint
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data

print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42)

sequence_length = x.shape[1]  # 56
vocabulary_size = len(vocabulary_inv)  # 18765
embedding_dim = 256

filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5
epochs = 100
batch_size = 30

new_model = load_model("weights.001-0.9113.hdf5")
# fit the model
filepath = 'keras_model/model_random_initialize.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
new_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training
