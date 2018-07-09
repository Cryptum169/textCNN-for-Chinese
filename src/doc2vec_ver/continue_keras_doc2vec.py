from keras.callbacks import ModelCheckpoint
from keras.models import Model
from sklearn.model_selection import train_test_split
# from data_helpers import load_data_doc2vec
import data_helpers_doc2vec as data_helpers
import keras

print('Loading data')
# sentences, labels = data_helpers.load_data_and_labels_doc2vec()
x, y = data_helpers.load_data_doc2vec()
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42)

filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5
epochs = 10
batch_size = 32

print('Loadong Model')
filepath = 'model/textCNN_with_doc2vec/model_doc2vec.hdf5'
model = keras.models.load_model(filepath)
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              callbacks=callbacks_list, validation_data=(X_test, y_test))  # starts training

# for i in range(10):
#     del x, y, X_test, X_train, y_train, y_test
#     x, y = load_data_doc2vec(sentences, labels)
#     X_train, X_test, y_train, y_test = train_test_split(
#         x, y, test_size=0.1, random_state=42)
#     model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
#                 callbacks=callbacks_list, validation_data=(X_test, y_test))  # starts training
