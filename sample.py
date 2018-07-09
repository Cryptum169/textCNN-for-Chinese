from src.textCNN import textCNN
from src.data_helpers import load_data
from src.data_helpers import build_input_data
from sklearn.model_selection import train_test_split
import numpy as np
import jieba
import json

# Train a new model
print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data('data/Spam/')

with open('vocabulary.json', 'w') as f:
    json.dump(vocabulary, f)

classifier = textCNN(
    sequence_length=x.shape[1],
    vocabulary_size=len(vocabulary_inv),
    num_classifier=y.shape[1]
)

del vocabulary
del vocabulary_inv

classifier.construct_model()
classifier.train(x, y, checkpoint_path='model/textCNN/classification.hdf5', epochs = 10)

del classifier

# ## Load an existing Model
# old_model = textCNN(directory = 'model/textCNN/classification.hdf5')

# # Keep training old model
# old_model.train(x, y, checkpoint_path='model/textCNN/classification.hdf5', epochs = 1)

# # predict using model

def load_text_and_label():
    econ = 'data/evaluation_data/econ.txt'
    sports = 'data/evaluation_data/econ.txt'
    econList = list(open(econ, "r").readlines())
    econList = [jieba.lcut(s.strip()) for s in econList]
    sportsList = list(open(sports, "r").readlines())
    sportsList = [jieba.lcut(s.strip()) for s in sportsList]
    x_text = econList + sportsList
    econLabel = [[1, 0, 0, 0] for _ in econList]
    sportLabel = [[0, 0, 1, 0] for _ in sportsList]
    y = np.concatenate([econLabel, sportLabel], 0)
    return [x_text, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = 342  # as defined by model
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding > 0:
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences

with open('vocabulary.json', 'r') as fp:
    dictionary = json.load(fp)
    [corpus, labels] = load_text_and_label()
    data = pad_sentences(corpus)
    x, y = build_input_data(data, labels, dictionary)

result = classifier.model.predict(x)
result = np.where(result > 0.5, 1, 0)

counter = 0
for i in range(len(y)):
    if np.array_equal(y[i], result[i]):
        counter += 1
accuracy = counter/len(y)
print(accuracy)
