from typing import List, Tuple

import luigi
import os
import math
from ast import literal_eval

import pandas as pd
import numpy as np
import requests

#from imblearn.over_sampling.random_over_sampler import RandomOverSampler
#from imblearn.under_sampling.prototype_selection.random_under_sampler import RandomUnderSampler
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm._tqdm import tqdm
import spacy
from unidecode import unidecode
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from src.util import util
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from src.task.data_preparation import CleanDataFrames, TokenizerDataFrames, LoadEmbeddings
from src.files import get_params_path, get_torch_weights_path, get_params, get_history_path, \
    get_tensorboard_logdir, get_classes_path, get_task_dir, get_keras_weights_path, get_history_plot_path
import json
import shutil
from sklearn.utils import class_weight
from src.task.train import ModelTraining

from keras.layers import Embedding, Conv1D, SpatialDropout1D, GlobalMaxPool1D,BatchNormalization, Dropout, LSTM
from keras.layers import Lambda, Layer, Input, Flatten, MaxPooling1D, concatenate, Dense
from keras import Model
from keras_radam import RAdam
from src.util.metrics import *

class InceptionTraining(ModelTraining):
    learning_rate: float = luigi.FloatParameter(1e-3)
    batch_size: int = luigi.IntParameter(default=50)
    epochs: int = luigi.IntParameter(default=30)
    input_size: int = luigi.IntParameter(default=20)
    output_size: int = luigi.IntParameter(default=677)

    def create_model(self):
        embedding_matrix = self.get_embeddings()

        # Input
        input_x = Input(shape=(self.input_size,), dtype='int32')

        # Embedding Layer
        x = Embedding(len(embedding_matrix),
                    self.dim,
                    weights=[embedding_matrix],
                    input_length=self.input_size,
                    trainable=True)(input_x)

        x = SpatialDropout1D(0.3)(x)

        def inseption_mod(input_layer, layers = 10):
            ### 1st layer
            layer_1 = Conv1D(layers, 1, padding='same', activation='relu')(input_layer)
            layer_1 = Conv1D(layers, 3, padding='same', activation='relu')(layer_1)

            layer_2 = Conv1D(layers, 1, padding='same', activation='relu')(input_layer)
            layer_2 = Conv1D(layers, 5, padding='same', activation='relu')(layer_2)

            layer_3 = MaxPooling1D(3, strides=1, padding='same')(input_layer)
            layer_3 = Conv1D(layers, 1, padding='same', activation='relu')(layer_3)

            x_rnn = LSTM(30, return_sequences=True)(input_layer)
            
            x   = concatenate([layer_1, layer_2, layer_3, x_rnn], axis = -1)
            return x
        
        x = inseption_mod(x, 30)
        x = Dropout(0.3)(x)

        # RNN Layer
        #model.add(LSTM(seq_size))
        x = GlobalMaxPool1D()(x)
        x = BatchNormalization()(x)
        
        # Dense Layer
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)    
        output = Dense(self.output_size, activation='softmax')(x)
        
        model  = Model([input_x], output)

        model.compile(loss = 'categorical_crossentropy', 
                      optimizer=RAdam(),
                      metrics = ['accuracy', f1_m])
        
        model.summary()
        
        return model
