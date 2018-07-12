# An implementation of textCNN for Chinese with and without gensim Doc2Vec. 
With pre-trained Doc2Vec model, textCNN's model size shrink from around 200mb to 30 mb. Training speed has increased significantly. On my dataset, textCNN with Doc2Vec implementation is able to get to around 80% accuracy on first epoch, with average step time cost 1ms. Vanilla textCNN will take several more epoch to converge and average step time of 70~ms. 

Be aware tho that Doc2Vec model make take up to several giga bytes

# Requirements
Run ```pip install -r requirements.txt```

# Creating a Gensim Doc2Vec model
Use ```doc2vec_model_gen.py``` under ```src``` to generate a gensim Doc2Vec model. Directly call doc2vec_model_gen.py in terminal with corpus named "Content.txt" in the same directory. Content.txt need to be line-separated articles. i.e. Article1 \n Article2 \n Article3.

Put all four model files under model/doc2vec directory.

# Un-uploaded files
Pre-generated doc2vec model, pre-trained textCNN model with and without doc2vec versions can be found ~~here~~ still uploading. Training data can be found at this [link](https://pan.baidu.com/s/1CqSusnOfBFXtjG7o2NRcWQ).

When downloaded, place ```training_data``` under ```data/Sogou/training_data```, directly replace local model directory with downloaded model folder is ok.

# Sample Code for training the model
## Sample textCNN without Doc2Vec:
```
from src.textCNN import textCNN
from src.data_helpers import load_data

# Load all .txt file under directory
x, y, vocabulary, vocabulary_inv = load_data('data/Sogou/training_data/')

# Construct textCNN Object
classifier = textCNN(
    sequence_length=x.shape[1],
    vocabulary_size=len(vocabulary_inv),
    num_classifier=y.shape[1]
)

# Instantiate Model
classifier.construct_model()

# start Training, the function is already wrapped with random shuffler
classifier.train(x, y, checkpoint_path='model/textCNN/classification.hdf5', epochs=20)
```

## Sample textCNN with Doc2Vec model:
```
from src.textCNN import textCNN
from src.data_helpers_doc2vec import load_data_doc2vec
# Load data in terms of Doc2Vec vector matrix
x, y = load_data_doc2vec(doc2vec_directory = 'model/doc2vec/d2v.model')

# Construct model
classifier_doc2vec = textCNN(
    sequence_length=x.shape[1],
    num_classifier=y.shape[1],
    embedding_dim=x.shape[2],
    doc2vec=True)

# Instantiate Model
classifier_doc2vec.construct_model()

# Train
classifier_doc2vec.train(
    x, y, checkpoint_path='model/textCNN_with_doc2vec/classification.hdf5', epochs=20)
```

# Prediction Using the Model
```sample.py``` contains a code snippet on how to predict/verify the model. An wrapped-around feature will be added in the future.