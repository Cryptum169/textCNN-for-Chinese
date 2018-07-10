from src.textCNN import textCNN
from src.data_helpers import load_data
import json

x, y, vocabulary, vocabulary_inv = load_data('data/Sogou/training_data')

with open('model/textCNN/dict/vocabulary_spam.json', 'w') as f:
    json.dump(vocabulary, f)

print("Vocabulary Size is {}".format(len(vocabulary_inv)))

classifier = textCNN(
    sequence_length=x.shape[1],
    vocabulary_size=len(vocabulary_inv),
    num_classifier=y.shape[1]
)

del vocabulary
del vocabulary_inv

classifier.construct_model()
classifier.train(x, y, checkpoint_path='model/textCNN/classification.hdf5', epochs=20)

