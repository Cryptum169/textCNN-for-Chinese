from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re
import jieba
import progressbar

print('Loading Corpus')
with open('Content.txt','r') as fp:
    corpus = [re.split('。|！|？|；',lines[:-1]) for lines in fp.readlines()]

corpus = [x for x in corpus if x] # Eliminate empty string left due to splits

expand_corpus = [item for sublist in corpus for item in sublist]
print('Corpus Extraction Completed, Constructing Tagged Documents')

tagged_data = [TaggedDocument(words=jieba.lcut(_d), tags=[str(i)])
               for i, _d in enumerate(expand_corpus)]
del expand_corpus, corpus
print('Tagged Data construction completed')

max_epochs = 25
vec_size = 20
alpha = 0.025

print('Constructing Model')
model = Doc2Vec(vec_size = 400,
                alpha=alpha, 
                min_alpha=0.025,
                min_count=10,
                workers = 4,
                size = 400,
                dbow_words=1,
                compute_loss = True,
                dm =0)
print('Model Construction completed')
print('Building Vocabulary')
model.build_vocab(tagged_data)
print('Vocabulary Built, start training')
count = 0
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    model.alpha -= 0.0002
    model.min_alpha = model.alpha
    if epoch % 5 == 0:
        model.save("d2v.model")

print("Model Saved")