import numpy as np
import pickle as pkl
import mxnet as mx
import ipywidgets as widgets
import re

sentence_length = 128
vocab_size = 20000

input_numpy = np.zeros((sentence_length, 128), dtype=np.int32)

vocab, rev_vocab = pkl.load(open('data/imdb.vocab', 'rb'))
imdb_data = pkl.load(open('data/imdb_data.pkl', 'r'))

class IMDB(object):
    
    def __init__(self, batch_size=128):
        data = {'X_train': mx.nd.array(imdb_data['X_train'], dtype=np.int32),
                'X_valid': mx.nd.array(imdb_data['X_valid'], dtype=np.int32),
                'Y_train': mx.nd.array(imdb_data['Y_train']),
                'Y_valid': mx.nd.array(imdb_data['Y_valid'])
               }
        
        self.train_set = mx.io.NDArrayIter(data['X_train'], data['Y_train'], batch_size=batch_size)
        self.valid_set = mx.io.NDArrayIter(data['X_valid'], data['Y_valid'], batch_size=batch_size)
        

def text_window():

    text = widgets.Textarea(
    value="""This movie was a well-written story of Intel's history, from its highs to its lows. I especially liked the character development. They should have gotten Brad Pitt to play Robert Noyce though, the actor's acting was bad. For example, that scene where they left Fairchild to start Intel was way too exaggerated and melodramatic. The pace of the movie was exciting enough to overlook those minor issues. I was on the edge of my seat the whole time, and my brother was equally enthralled!""",
    placeholder='Type something',
    description='Review:',
    disabled=False,
    layout=widgets.Layout(height='200px', width='50%'))

    return text

sentence_length = 128
vocab_size = 20000

def clean_string(string):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def preprocess(sent, flag=False):

    if flag is False:
        tokens = clean_string(sent).strip().split()

        sent = [len(vocab) + 1 if t not in vocab else vocab[t] for t in tokens]
        sent = [1] + [w + 3 for w in sent]
        sent = [2 if w >= vocab_size else w for w in sent]

        trunc = sent[-sentence_length:]  # take the last sentence_length words
    else:
        trunc = sent
        
    input_numpy = np.zeros((128, 128), dtype=np.int32)  # fill with zeros
    input_numpy[0, -len(trunc):] = trunc   # place the input into the numpy array

    return mx.io.NDArrayIter(mx.nd.array(input_numpy), batch_size=128)