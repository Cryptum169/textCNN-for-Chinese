from src.textCNN import textCNN
from src.data_helpers import load_data

print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data('data/training_data/')

classifier = textCNN(
    sequence_length=x.shape[1],
    vocabulary_size=len(vocabulary_inv),
    num_classifier=y.shape[1]
)

classifier.construct_model()
classifier.train(x, y, checkpoint_path='model/textCNN/try.hdf5')
