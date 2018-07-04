from keras.layers import Input, Dense, Embedding, Conv1D, MaxPool1D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
import keras
from sklearn.model_selection import train_test_split
from data_helpers import load_data_doc2vec_1D
import json

print('Loading Data')
[x, y] = load_data_doc2vec_1D()
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42)

sequence_length = 75 # x.shape[0] # x.shape[1] (100, 1)
# vocab_size = len(vocab_inv)
embedding_size = 100

filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

epochs = 100
batch_size = 30

print("Creating Model...")
# print(x.shape)
inputs = Input(shape=(embedding_size,1), dtype='float64')

reshape = Reshape((embedding_size, 1))(inputs)

conv_0 = Conv1D(
    num_filters, kernel_size=filter_sizes[0], kernel_initializer='normal', activation='relu')(inputs)
conv_1 = Conv1D(
    num_filters, kernel_size=filter_sizes[1], kernel_initializer='normal', activation='relu')(inputs)
conv_2 = Conv1D(
    num_filters, kernel_size=filter_sizes[2], kernel_initializer='normal', activation='relu')(inputs)

maxpool_0 = MaxPool1D(sequence_length - filter_sizes[0] + 1)(conv_0)
maxpool_1 = MaxPool1D(sequence_length - filter_sizes[1] + 1)(conv_1)
maxpool_2 = MaxPool1D(sequence_length - filter_sizes[2] + 1)(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=4, activation='softmax')(dropout)

model = Model(inputs=inputs, outputs=output)
filepath = 'model/textCNN/model_doc2vec.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print("Traning Model...")
print(model.summary())

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training
# 