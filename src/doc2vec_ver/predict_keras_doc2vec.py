from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import progressbar
import keras
import numpy as np
import jieba
import json
from gensim.models.doc2vec import Doc2Vec
import data_helpers as dh


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = 262  # as defined by dictionary
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
    print('Loading Gensim Model')
    model = Doc2Vec.load('d2v.model')
    print('Generating Gensim Vector')

    totalDoc = []
    i = 0
    with progressbar.ProgressBar(max_value=len(data)) as bar:
        for eachDocument in data:
            currDoc = []
            for eachSentence in eachDocument:
                currDoc.append(model.infer_vector(eachSentence))
            currDoc = np.array(currDoc)
            totalDoc.append(currDoc)
            i += 1
            bar.update(i)
    x = np.array(totalDoc)
    # 262 as max length, 100 as embedding length
    x.resize(x.shape[0], 262, 100, 1)
    y = np.array(labels)
    return [x, y]


x, y = load_eval_dataset()
filepath = 'keras_model/model_doc2vec.hdf5'
model = keras.models.load_model(filepath)
result = model.predict(x)
result = np.where(result > 0.5, 1, 0)
accuracy = np.sum(y == result) / 4 / x.shape[0]
print(accuracy)
