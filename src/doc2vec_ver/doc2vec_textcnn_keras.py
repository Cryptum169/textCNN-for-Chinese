from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data_doc2vec
import data_helpers
import json

print('Loading data')
sentences, labels = data_helpers.load_data_and_labels_doc2vec()
x, y = load_data_doc2vec(sentences, labels)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42)

# # with open('my_dict.json', 'w') as f:
# #     json.dump(vocabulary, f)

sequence_length = x.shape[1]  # 342 Used for Prediction
# # print(sequence_length)
# # vocabulary_size = max_length  # 81867 Used for Prediction
# # print(vocabulary_size)
embedding_dim = 100

filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5

epochs = 5
batch_size = 30

# this returns a tensor
print("Creating Model...")

inputs = Input(shape=(sequence_length,embedding_dim,1), dtype='float64')

conv_0 = Conv2D(num_filters, kernel_size=(
    filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(inputs)
conv_1 = Conv2D(num_filters, kernel_size=(
    filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(inputs)
conv_2 = Conv2D(num_filters, kernel_size=(
    filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(inputs)

maxpool_0 = MaxPool2D(pool_size=(
    sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(
    sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(
    sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=4, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)
filepath = 'keras_model/model_doc2vec.hdf5'
checkpoint = ModelCheckpoint(filepath,
                             monitor='acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")

model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, verbose=1 ,callbacks=[checkpoint],validation_data=(X_test, y_test))  # starts training

x, y = load_data_doc2vec(sentences, labels)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42)

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training

x, y = load_data_doc2vec(sentences, labels)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42)

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training

x, y = load_data_doc2vec(sentences, labels)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42)

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training
