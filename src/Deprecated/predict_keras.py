from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import keras
import numpy as np
import jieba
import json
import data_helpers as dh

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = 342  # as defined by dictionary
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


def load_text_and_label():
    econ = 'eval_set/Econ_Eval/all.txt'
    sports = 'eval_set/Sports_Eval/all.txt'
    econList = list(open(econ, "r").readlines())
    econList = [jieba.lcut(s.strip()) for s in econList]
    sportsList = list(open(sports, "r").readlines())
    sportsList = [jieba.lcut(s.strip()) for s in sportsList]
    x_text = econList + sportsList

    econLabel = [[1, 0, 0, 0] for _ in econList]
    sportLabel = [[0, 0, 0, 1] for _ in sportsList]
    y = np.concatenate([econLabel, sportLabel], 0)
    return [x_text, y]


def load_eval_dataset():
    with open('my_dict.json', 'r') as fp:
        dictionary = json.load(fp)
    [corpus, labels] = load_text_and_label()
    data = pad_sentences(corpus)
    x, y = dh.build_input_data(data, labels, dictionary)
    return [x, y]


x, y = load_eval_dataset()
model = keras.models.load_model('classification.hdf5')
result = model.predict(x)
result = np.where(result > 0.5, 1, 0)
accuracy = np.sum(y == result)/4/100
print(accuracy)
