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

# PYTHONPATH="." luigi --module src.task.model.simple_cnn SimpleCNNTraining --local-scheduler --batch-size 500 --epochs 40 --sample 100000 --output-size 1583 --seq-size 50 --input-size 50
class SimpleCNNTraining(ModelTraining):
    learning_rate: float = luigi.FloatParameter(1e-3)
    batch_size: int = luigi.IntParameter(default=50)
    epochs: int = luigi.IntParameter(default=30)
    input_size: int = luigi.IntParameter(default=20)
    output_size: int = luigi.IntParameter(default=677)
    dropout: float = luigi.FloatParameter(default=0.3)
    kernel_size: int = luigi.IntParameter(default=3)
    filters: int = luigi.IntParameter(default=256)

    def create_model(self):
        embedding_matrix = self.get_embeddings()

        # Input
        input_x = Input(shape=(self.input_size,), dtype='int32')

        # Embedding Layer
        x   = Embedding(len(embedding_matrix),
                        self.dim,
                        weights=[embedding_matrix],
                        input_length=self.input_size,
                        trainable=True)(input_x)

        x   = SpatialDropout1D(selfdropout)(x)
        x   = Conv1D(self.filters, self.kernel_size, padding='same', activation='relu')(x)
        
        x   = GlobalMaxPool1D()(x)
        x   = BatchNormalization()(x)
        
        x   = Dense(750, activation='relu')(x)
        out = Dense(self.output_size, activation='softmax')(x)
        
        model  = Model([input_x], out)

        model.compile(loss = 'categorical_crossentropy', 
                      optimizer=RAdam(),
                      metrics = ['accuracy', f1_m])
        
        model.summary()
        
        return model
