from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from data_helpers import load_data
import json

print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42) # The Magic number of universe
 
with open('vocabulary.json', 'w') as f:
    json.dump(vocabulary, f)

with open('vocabulary_inv.json','w') as f:
    json.dump(vocabulary_inv, f)

sequence_length = x.shape[1] # 342 Used for Prediction
vocabulary_size = len(vocabulary_inv)
embedding_dim = 256
num_classifier = 4

filter_sizes = [3, 4, 5]
num_filters = 512
drop = 0.5

epochs = 100
batch_size = 30

print("Creating Model...")

inputs = Input(shape=(sequence_length,), dtype='int32')

embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)

reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(
    filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(
    filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = Conv2D(num_filters, kernel_size=(
    filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = MaxPool2D(pool_size=(
    sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(
    sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(
    sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=num_classifier, activation='softmax')(dropout)

# this creates a model that includes
model = Model(inputs=inputs, outputs=output)
filepath = 'model/textCNN/classification.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
print("Traning Model...")

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=[checkpoint], validation_data=(X_test, y_test))  # starts training
