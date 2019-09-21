from typing import List, Tuple

import luigi
import os
import math
from ast import literal_eval

import pandas as pd
import numpy as np
import requests
from PIL import Image, ImageOps
import zipfile

#from imblearn.over_sampling.random_over_sampler import RandomOverSampler
#from imblearn.under_sampling.prototype_selection.random_under_sampler import RandomUnderSampler
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm._tqdm import tqdm
from skimage import exposure
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
    get_tensorboard_logdir, get_classes_path, get_task_dir, get_keras_weights_path, get_history_plot_path, get_submission_path
import json
import shutil
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report 
from contextlib import redirect_stdout


SEED = 42
DATASET_DIR = "output/dataset"
DEFAULT_DEVICE = "cuda" 

class ModelTraining(luigi.Task):
    device: str = luigi.ChoiceParameter(choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    val_size: float = luigi.FloatParameter(default=0.2)
    seed: int = luigi.IntParameter(default=42)
    num_words: int = luigi.IntParameter(default=5000)
    seq_size: int = luigi.IntParameter(default=20)

    dim: int = luigi.IntParameter(default=300)
    embedding: str = luigi.ChoiceParameter(choices=["glove", "fasttext"], default="fasttext")

    learning_rate: float = luigi.FloatParameter(1e-3)
    batch_size: int = luigi.IntParameter(default=50)
    val_batch_size: int = luigi.IntParameter(default=100)
    epochs: int = luigi.IntParameter(default=30)
    sample: int = luigi.IntParameter(default=20000000)


    def requires(self):
        return (CleanDataFrames(val_size=self.val_size, seed=self.seed, sample=self.sample),
                TokenizerDataFrames(val_size=self.val_size, seed=self.seed, 
                                        num_words=self.num_words, seq_size=self.seq_size, sample=self.sample),
                LoadEmbeddings(val_size=self.val_size, seed=self.seed, dim=self.dim, embedding=self.embedding,
                                        num_words=self.num_words, seq_size=self.seq_size, sample=self.sample))

    def output(self):
        return luigi.LocalTarget(get_task_dir(self.__class__, self.task_id))

    @property
    def resources(self):
        return {"cuda": 1} if self.device == "cuda" else {}

    @property
    def device_id(self):
        if not hasattr(self, "_device_id"):
            if self.device == "cuda":
                self._device_id = CudaRepository.get_avaliable_device()
            else:
                self._device_id = None
        return self._device_id

    def _save_params(self):
        params = self._get_params()
        with open(get_params_path(self.output_path), "w") as params_file:
            json.dump(params, params_file, default=lambda o: dict(o), indent=4)

    def _get_params(self):
        return self.param_kwargs


    @property
    def output_path(self):
        if hasattr(self, "_output_path"):
            return self._output_path
        return self.output().path


    def after_train(self):
        self.create_submission_file()

    def create_submission_file(self):
        X_test = self.get_test_ds()
        model  = self.get_trained_model()
        Y_classes = pd.read_csv(os.path.join(self.input()[1][1].path), index_col='index').columns

        pred_test    = model.predict(X_test, batch_size=self.batch_size).argmax(1)
        pred_classes = Y_classes[pred_test]

        f = open(get_submission_path(self.output_path), 'w')
        f.write('id,category\n')
        for i in range(len(pred_classes)):
            f.write('{},{}\n'.format(i, pred_classes[i]))
        f.close()      

    def create_model(self):
        pass        

    def get_trained_model(self):
        model = self.create_model()
        model.load_weights(get_keras_weights_path(self.output_path))
        return model

    def get_embeddings(self):
        with open(os.path.join(self.input()[2].path),'rb') as f:
            embedding_matrix = pickle.load(f)

        return embedding_matrix

    def get_train_ds(self):
        X = np.load(os.path.join(self.input()[1][0].path))
        Y = pd.read_csv(os.path.join(self.input()[1][1].path), index_col='index')

        return X, Y

    def get_val_ds(self):
        X = np.load(os.path.join(self.input()[1][2].path))
        Y = pd.read_csv(os.path.join(self.input()[1][3].path), index_col='index')
        
        return X, Y

    def get_test_ds(self):
        X = np.load(os.path.join(self.input()[1][4].path))
        
        return X

    def train(self):
        # Load Dataset
        print("Load Dataset")
        X_train, Y_train = self.get_train_ds()
        X_valid, Y_valid = self.get_val_ds()
        X_test = self.get_test_ds()
      
        # Classes
        Y_classes = Y_train.columns
        Y_train = Y_train.values
        Y_valid = Y_valid.values

        print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape)

        # Model Build
        print("==> class_weight")
        class_weights = class_weight.compute_class_weight('balanced', 
                            np.unique(Y_train.argmax(1)), Y_train.argmax(1))


        model = self.create_model()

        with open("%s/summary.txt" % self.output_path, "w") as summary_file:
            with redirect_stdout(summary_file):
                model.summary()

        # Train Model
        hist = model.fit(X_train, Y_train, 
                          validation_data =(X_valid, Y_valid),
                          batch_size = self.batch_size, 
                          epochs     = self.epochs,  
                          verbose    = 1,
                          class_weight = class_weights,
                          callbacks  = self._get_callbacks())

        # Evaluate
        model.load_weights(get_keras_weights_path(self.output_path))

        # classification_report.txt
        y_valid     = list(Y_classes[Y_valid.argmax(1)])
        pred        = model.predict(X_valid, batch_size=self.batch_size)
        predictions = Y_classes[pred.argmax(1)]

        with open("%s/classification_report.txt" % self.output_path, "w") as summary_file:
            with redirect_stdout(summary_file):
                print(classification_report(y_valid, predictions))


        # metrics.txt
        with open("%s/metrics.txt" % self.output_path, "w") as summary_file:
            with redirect_stdout(summary_file):
              val_loss, val_acc, fi_score = model.evaluate(X_valid, Y_valid, batch_size=self.batch_size)
              print("{} {} {}".format(val_loss, val_acc, fi_score))

        # history.jpg
        self.plot_hist(hist).savefig(get_history_plot_path(self.output_path))

    def _get_callbacks(self):
      checkpoint = ModelCheckpoint(get_keras_weights_path(self.output_path), 
                                   monitor='val_f1_m', 
                                   verbose=1, save_best_only=True, 
                                   mode='max')

      earlystop  = EarlyStopping(monitor='val_f1_m',
                                  min_delta=0,
                                  patience=5,
                                  verbose=0, 
                                  mode='max')      
      return [checkpoint, earlystop]


    def run(self):
        os.makedirs(self.output_path, exist_ok=True)
        self._save_params()
        try:
            self.train()
            self.after_train()
        except Exception:
            shutil.rmtree(self.output_path)
            raise
                
    def plot_hist(self, hist):
      # summarize history for loss
      fig, ax = plt.subplots()  # create figure & 1 axis

      plt.plot(hist.history['loss'][1:])
      plt.plot(hist.history['val_loss'][1:])
      
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      
      return fig   
