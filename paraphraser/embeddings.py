import numpy as np
import pickle
from six import iteritems
#from keras.layers.embeddings import Embedding

from ferutility.file import copen
from ferutility.gensim import KeyedVectors

# https://drive.google.com/file/d/1l2liCZqWX3EfYpzv9OmVatJAEISPFihW/view?usp=sharing
PARA_NMT_PATH = "./data/para-nmt-50m-demo.zip"
# http://nlp.stanford.edu/data/glove.6B.zip
GLOVE_PATH = "./data/glove.6B.zip"

def load_sentence_embeddings():
    '''Load John Wieting sentence embeddings'''
    with copen(PARA_NMT_PATH, "rb", archive_file="para-nmt-50m-demo/ngram-word-concat-40.pickle") as f:
        # [ numpy.ndarray(95283, 300), numpy.ndarray(74664, 300), (trigram_dict, word_dict)]
        x = pickle.load(f, encoding='latin1')
        word_vocab_size, embedding_size = x[1].shape

        trigram_embeddings, word_embeddings, _ = x
        trigram_to_id, word_to_id = x[2]

        word_to_id['<START>'] = word_vocab_size
        word_to_id['<END>'] = word_vocab_size + 1

        idx_to_word = { idx: word for word, idx in iteritems(word_to_id) }

        word_embeddings = np.vstack((word_embeddings, np.random.randn(2, embedding_size)))

        return (word_to_id, idx_to_word, word_embeddings, word_to_id['<START>'],
               word_to_id['<END>'], word_to_id['UUUNKKK'], word_to_id['★'])

def load_glove_embeddings():

    with copen(GLOVE_PATH, "rt", archive_file="glove.6B.300d.txt") as fr:
        vec = KeyedVectors.load_glove_format(fr)
        vec.add_word('<START>')
        vec.add_word('<END>')
        vec.add_word('UUUNKKK')
        vec.add_word('★')
        word_to_id, id_to_word, word_embeddings = vec.vocab, vec.index2word, vec.syn0
        return (word_to_id, id_to_word, word_embeddings, word_to_id['<START>'],
                word_to_id['<END>'], word_to_id['UUUNKKK'], word_to_id['★'])

if __name__ == '__main__':
    from pprint import pprint as pp

    word_to_id, idx_to_word, embedding, start_id, end_id, unk_id, mask_id = load_sentence_embeddings()
    pp(idx_to_word[mask_id])
    #pp(idx_to_word)
    #pp(word_to_id)
    #print(embedding.shape)
