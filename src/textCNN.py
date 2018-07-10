from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
import keras
import os

class textCNN():
    def __init__(self, sequence_length=200,
                 vocabulary_size=-1,
                 embedding_dim=256,
                 num_filters=512,
                 filter_sizes=[3, 4, 5],
                 num_classifier=2,
                 drop=0.5,
                 directory=None):
        if directory == None:
            print('Initializing Parameters')
            self.sequence_length = sequence_length
            self.vocabulary_size = vocabulary_size
            self.embedding_dim = embedding_dim
            self.num_filters = num_filters
            self.num_classifier = num_classifier
            self.filter_sizes = filter_sizes
            self.drop = drop
            self.resume_training = False
            self.model = None
        else:
            print('Loading Model from {}'.format(directory))
            self.model = keras.models.load_model(directory)
            self.resume_training = True
            self.directory = directory
            self.sequence_length = int(self.model.layers[0].get_output_at(
                0).get_shape().as_list()[1])

    def construct_model(self):
        if not self.model == None:
            raise Exception('Trying to construct new model in the same object while we already have one')

        # Input Layer
        print('Creating Model')
        inputs = Input(shape=(self.sequence_length,), dtype='int32')
        # Embedding Layer
        embedding = Embedding(input_dim=self.vocabulary_size,
                              output_dim=self.embedding_dim, input_length=self.sequence_length)(inputs)

        # Check to make sure dimension matches
        reshape = Reshape(
            (self.sequence_length, self.embedding_dim, 1))(embedding)

        # Convolutional layers
        conv_0 = Conv2D(self.num_filters,
                        kernel_size=(
                            self.filter_sizes[0], self.embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(self.num_filters,
                        kernel_size=(
                            self.filter_sizes[1], self.embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(self.num_filters,
                        kernel_size=(
                            self.filter_sizes[2], self.embedding_dim), padding='valid',
                        kernel_initializer='normal', activation='relu')(reshape)

        # Pooling layers
        maxpool_0 = MaxPool2D(
            pool_size=(self.sequence_length - self.filter_sizes[0] + 1, 1),
            strides=(1, 1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(
            pool_size=(self.sequence_length - self.filter_sizes[1] + 1, 1),
            strides=(1, 1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(
            pool_size=(self.sequence_length - self.filter_sizes[2] + 1, 1),
            strides=(1, 1), padding='valid')(conv_2)

        # Concatenate Stuff
        concatenated_tensor = Concatenate(axis=1)(
            [maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(self.drop)(flatten)
        output = Dense(units=self.num_classifier,
                       activation='softmax')(dropout)
        self.model = Model(inputs=inputs, outputs=output)

        print('Optimizer Defaulted to be Adam')
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, decay=0.0)

        self.model.compile(optimizer=adam, loss='categorical_crossentropy',
                           metrics=['accuracy'])
        print(self.model.summary())

    def train(self, x_data, y_data, checkpoint_path=-1, batch_size=32, epochs=50):

        # Check for new
        if x_data.shape[1] > self.sequence_length:
            x_data = np.array([x[:int(self.sequence_length)] for x in x_data])

        X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.1, random_state=42)

        # Questionable Code block here, fix later
        if checkpoint_path == -1:
            print('No Checkpoint path directory found, creating new one')
            checkpoint_path = 'model/textCNN'
            os.mkdir(checkpoint_path)
            checkpoint_path += '/classification.hdf5'

        checkpoint = ModelCheckpoint(
            checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')


        print('Start Training Model')
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                       callbacks=[checkpoint], validation_data=(X_test, y_test))

    def predict(self, x_data, y_data):
        if self.model == None:
            raise Exception('Trying to predict with no model')