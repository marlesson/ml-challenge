from unidecode import unidecode
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import spacy
from keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow import keras
import numpy as np

# https://leportella.com/pt-br/2017/11/30/brincando-de-nlp-com-spacy.html
LANG = {
    'pt': (spacy.load('pt'), nltk.corpus.stopwords.words('portuguese')),
    'es': (spacy.load('es'), nltk.corpus.stopwords.words('spanish'))
}


def read_stopwords():

    file_pt      = open('src/files/stopwords_pt.txt')
    stopwords_pt = file_pt.readlines()
    stopwords_pt = [unidecode(w.replace('\n', '').strip()) for w in stopwords_pt]
    file_pt.close()

    file_es      = open('src/files/stopwords_es.txt')
    stopwords_es = file_es.readlines()
    stopwords_es = [unidecode(w.replace('\n', '').strip()) for w in stopwords_es]
    file_es.close()

    return {'pt': stopwords_pt, 'es': stopwords_es}

def smooth_labels(y, mask, smooth_factor):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.

    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)

    # Returns
        A matrix of smoothed labels.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        
        y     = y.astype('float16')
        y_ori = y.astype('int8')#.copy()
        
        # discont
        y *= ((1 - smooth_factor))
        y += smooth_factor / y.shape[1]
        
        # add mask
        y -= (smooth_factor/y.shape[1])*np.array([mask]).T
        y += y_ori*smooth_factor*np.array([mask]).T
        
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y    
    
def pre_process(text):
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("<!--?.*?-->","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    # punk
    text = re.sub(r'[?|!|\'|#]', r'', text)
    text = re.sub(r'[.|,|)|(|\|/]', r' ', text)
    
    return text

def pre_process_word(word, stopwords = [], lang = 'pt'):
    
    # remove acentuação
    word = unidecode(word)
    
    word = re.sub('\s+',' ',word)
    
    word = word.strip()
    
    stopwords  = LANG[lang][1]

    if word in stopwords:
        return ""
    
    if len(word) < 2:
        return ""
    
    return word

# Tokenização
def tokenize(text, lang='pt'):
    # clean
    text = pre_process(text)
    nlp, stopwords = LANG[lang]

    stopwords = stopwords.extend(read_stopwords()[lang])

    # Transform 
    text = nlp(text)
    
    # Tokenização
    tokens = [token.lemma_ for token in text if not token.is_stop]

    # Clean tokens
    tokens = [pre_process_word(item, stopwords, lang) for item in tokens]
    
    # Stemização
#     stems  = []
#     for item in tokens:
#         stems.append(SnowballStemmer("portuguese").stem(item))
        
    return tokens

def text_clean_pt(text, lang='pt'):
    return " ".join(tokenize(text, lang=lang))

def text_clean_es(text, lang='es'):
    return " ".join(tokenize(text, lang=lang))
   

from tensorflow.keras.datasets import mnist
from tensorflow import keras
import numpy as np
def smooth_labels(y, mask, smooth_factor):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.

    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)

    # Returns
        A matrix of smoothed labels.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        
        y     = y.astype('float64')
        y_ori = y.copy()
        
        # discont
        y *= ((1 - smooth_factor))
        y += smooth_factor / y.shape[1]
        
        # add mask
        y -= (smooth_factor/y.shape[1])*np.array([mask]).T
        y += y_ori*smooth_factor*np.array([mask]).T
        
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y    
