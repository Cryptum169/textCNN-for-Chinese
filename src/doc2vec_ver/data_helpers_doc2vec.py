import random
import jieba
import progressbar
from gensim.models import Doc2Vec
import numpy as np

def load_data_and_labels_doc2vec(econ='data/Sogou/training_data/bizList.txt',
                                 dangshi='data/Sogou/training_data/dangshiList.txt', ent='data/Sogou/training_data/entList.txt',
                                 sports='data/Sogou/training_data/sportsList.txt'):
    print('Function reserved for doc2vec')
    econList = list(open(econ, "r").readlines())
    econList = [s.split('。')[:-1] for s in econList]
    res = []
    for s in econList:
        currList = []
        for seg in s:
            currList.append(jieba.lcut(seg.strip()))
        res.append(currList)
    econList = res
    print('{} data from econ sec'.format(len(econList)))

    dangshiList = list(open(dangshi, "r").readlines())
    dangshiList = [s.split('。') for s in dangshiList]
    res = []
    for s in dangshiList:
        currList = []
        for seg in s:
            currList.append(jieba.lcut(seg.strip()))
        res.append(currList)
    dangshiList = res
    print('{} data from dangshi sec'.format(len(dangshiList)))

    entList = list(open(ent, "r").readlines())
    entList = [s.split('。') for s in entList]
    res = []
    for s in entList:
        currList = []
        for seg in s:
            currList.append(jieba.lcut(seg.strip()))
        res.append(currList)
    entList = res
    print('{} data from Entertainment sec'.format(len(entList)))

    sportsList = list(open(sports, "r").readlines())
    sportsList = [s.split('。') for s in sportsList]
    res = []
    for s in sportsList:
        currList = []
        for seg in s:
            currList.append(jieba.lcut(seg.strip()))
        res.append(currList)
    sportsList = res
    print('{} data from Sports sec'.format(len(sportsList)))

    x_text = econList + dangshiList + entList + sportsList
    # x_text = [cleanStr(text) for text in x_text]

    econLabel = [[1, 0, 0, 0] for _ in econList]
    dangshiLabel = [[0, 1, 0, 0] for _ in dangshiList]
    entLabel = [[0, 0, 1, 0] for _ in entList]
    sportLabel = [[0, 0, 0, 1] for _ in sportsList]

    print('x_text len: {}'.format(len(x_text)))

    y = np.concatenate(
        [econLabel, dangshiLabel, entLabel, sportLabel], 0)
    return [x_text, y]


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


def load_data_doc2vec(sentences=None, labels=None):
    """
    Loads data for the dataset, uses gensim doc2vec to generate embedding.
    Returns matrix of input text, each column column consists of vectors of sentences
    """
    # [[sentence],[sentence]],[[sentence],[sentence]],[[sentence],[sentence]]]
    try:
        if not (len(sentences) > 0 and len(labels) > 0):
            sentences, labels = load_data_and_labels_doc2vec()
    except:
        sentences, labels = load_data_and_labels_doc2vec()
    testCase = 5405
    sentences_padded = pad_sentences(sentences)
    print('There are {} samples'.format(len(sentences_padded)))
    c = list(zip(sentences_padded, labels))
    random.shuffle(c)
    sentences_padded, labels = zip(*c)
    sentences_padded = sentences_padded[:testCase]
    labels = labels[:testCase]
    print('Loading Gensim Model')
    model = Doc2Vec.load('d2v.model')
    print('Generating Gensim Vector')
    totalDoc = []
    i = 0
    with progressbar.ProgressBar(max_value=len(sentences_padded)) as bar:
        for eachDocument in sentences_padded:
            currDoc = []
            for eachSentence in eachDocument:
                model.random.seed(0)
                currDoc.append(model.infer_vector(eachSentence))
            currDoc = np.array(currDoc)
            totalDoc.append(currDoc)
            i += 1
            bar.update(i)
    x = np.array(totalDoc)
    print(x.shape)
    x.resize(x.shape[0], x.shape[1], x.shape[2], 1)
    print(x.shape)
    y = np.array(labels)
    return [x, y]


def load_data_doc2vec_1D(sentences=None, labels=None):
    """
    Loads data for the dataset, uses gensim doc2vec to generate embedding.
    Returns input vector and labels, document directly converted to vector
    """
    try:
        if not (len(sentences) > 0 and len(labels) > 0):
            sentences, labels = load_data_and_labels()
    except:
        sentences, labels = load_data_and_labels()
    trainSet = 100000
    sentences_padded = pad_sentences(sentences)
    c = list(zip(sentences_padded, labels))
    random.shuffle(c)
    sentences_padded, labels = zip(*c)
    sentences_padded = sentences_padded[:trainSet]
    labels = labels[:trainSet]
    print('Loading Gensim Model')
    model = Doc2Vec.load('model/another/d2v.model')
    print('Generating Gensim Vector')
    totalDoc = []
    i = 0
    exit()
    with progressbar.ProgressBar(max_value=len(sentences_padded)) as bar:
        for eachDocument in sentences_padded:
            temp = np.array(model.infer_vector(eachDocument))
            temp.resize(1, 100)
            totalDoc.append(np.transpose(temp))
            i += 1
            bar.update(i)
    x = np.array(totalDoc)
    y = np.array(labels)
    return [x, y]
