import progressbar
import numpy as np
import re
import itertools
import jieba
from collections import Counter
from gensim.models.doc2vec import Doc2Vec
import random
import os
import re

def clean_s(string):
    string = string.replace('。','')
    string = string.replace('，','')
    return string

# Vanilla
def load_data_and_labels(econ='data/training_data/bizList.txt',
        dangshi = 'data/training_data/dangshiList.txt', ent = 'data/training_data/entList.txt',
                         sports='data/training_data/sportsList.txt'):

    econList = list(open(econ, "r").readlines())
    econList = [jieba.lcut(s.strip()) for s in econList]
    dangshiList = list(open(dangshi, "r").readlines())
    dangshiList = [jieba.lcut(s.strip()) for s in dangshiList]
    entList = list(open(ent, "r").readlines())
    entList = [jieba.lcut(s.strip()) for s in entList]
    sportsList = list(open(sports, "r").readlines())
    sportsList = [jieba.lcut(s.strip()) for s in sportsList]

    x_text = econList + dangshiList + sportsList + entList
    # print('x_test length is {}'.format(len(x_text)))

    econLabel = [[1, 0, 0, 0] for _ in econList]
    dangshiLabel = [[0, 1, 0, 0] for _ in dangshiList]
    sportLabel = [[0, 0, 1, 0] for _ in sportsList]
    entLabel = [[0, 0, 0, 1] for _ in entList]

    y = np.concatenate(
        [econLabel, dangshiLabel, sportLabel, entLabel], 0)
    return [x_text, y]

def load_multiple_data_and_labels(directory='data/training_data/'):
    if os.path.isdir(directory):
        data_set = [str(directory + x) for x in os.listdir(directory) if x[-4:] == '.txt' ]
    else:
        raise Exception('No training data directory')

    classifier = len(data_set)
    x_text = list()
    y = list()
    counter = 0

    for eachEntry in data_set:
        tempList = list(open(eachEntry, "r").readlines())
        tempList = [jieba.lcut(clean_s(s.strip())) for s in tempList]
        currentLabel = [0] * counter + [1] + [0] * (classifier - 1 - counter)
        print('Entry file:{} marked with label{}'.format(eachEntry, currentLabel))
        tempLabel = [currentLabel for _ in tempList]
        y += tempLabel
        x_text += tempList
        counter += 1
        
    y = np.array(y)
    return [x_text,y]
    
def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] if word in vocabulary else vocabulary["<PAD/>"] for word in sentence]
                  for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data(directory):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_multiple_data_and_labels(directory)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    print("x shape is {}".format(x.shape))
    print('y shape is {}'.format(y.shape))
    return [x, y, vocabulary, vocabulary_inv]