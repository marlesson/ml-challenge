from typing import List, Tuple

import luigi
import os
import math
from ast import literal_eval

import pandas as pd
import numpy as np
import requests
import zipfile

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

SEED = 42

DATASET_DIR = "output/dataset"


class DownloadDataset(luigi.Task):
    def output(self):
        return [luigi.LocalTarget("output/train.csv.gz"), luigi.LocalTarget("output/test.csv"), luigi.LocalTarget("output/sample_submission.csv")]

    def run(self):
        self.download("https://meli-data-challenge.s3.amazonaws.com/train.csv.gz", self.output()[0].path)
        self.download("https://meli-data-challenge.s3.amazonaws.com/test.csv", self.output()[1].path)
        self.download("https://meli-data-challenge.s3.amazonaws.com/sample_submission.csv", self.output()[2].path)

    def download(self, url, output_path):
        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)


        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        wrote = 0
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB',
                             unit_scale=True):
                wrote = wrote + len(data)
                f.write(data)
        if total_size != 0 and wrote != total_size:
            raise ConnectionError("ERROR, something went wrong")        


class UnzipDataset(luigi.Task):
    def requires(self):
        return DownloadDataset()

    def output(self):
        return luigi.LocalTarget(DATASET_DIR)

    def run(self):
        with zipfile.ZipFile(self.input().path, "r") as zip_ref:
            zip_ref.extractall(self.output().path)


class CleanDataFrames(luigi.Task):
    val_size: float = luigi.FloatParameter(default=0.3)
    seed: int = luigi.IntParameter(default=42)
    sample: int = luigi.IntParameter(default=20000000)
    with_smooth_labels: bool = luigi.BoolParameter(default=False)
    smooth_labels_intensity: float = luigi.FloatParameter(default=0.7)

    def requires(self):
        return UnzipDataset()

    def output(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget, luigi.LocalTarget]:
        task_hash = self.task_id.split("_")[-1]
        return (luigi.LocalTarget(os.path.join(self.input().path,
                                               "train_%.2f_%d_%d_%s.csv" % (
                                                   self.val_size, self.seed, self.sample, task_hash))),
                luigi.LocalTarget(
                    os.path.join(self.input().path, "val_%.2f_%d_%d_%s.csv" % (self.val_size, self.seed, self.sample, task_hash))),
                luigi.LocalTarget(os.path.join(self.input().path, "test_%d.csv" %(self.sample))),
                luigi.LocalTarget(os.path.join(self.input().path, "df_dummies_category_%s_%d.csv" % (self.with_smooth_labels, self.smooth_labels_intensity))))


    def run(self):
        train_df = pd.read_csv(os.path.join(self.input().path, "train.csv")).sample(self.sample, random_state=self.seed)
        test_df  = pd.read_csv(os.path.join(self.input().path, "test.csv")).sample(self.sample, random_state=self.seed)
        
        print("Shape: ")
        print("train_df: ", train_df.shape)
        print("test_df: ", test_df.shape)

        train_es_df = train_df[train_df.language == 'spanish']
        train_pt_df = train_df[train_df.language == 'portuguese']

        test_es_df = test_df[test_df.language == 'spanish']
        test_pt_df = test_df[test_df.language == 'portuguese']

        # Clean Text
        print("==> Transform...")

        train_es_df['title_clean'] = train_es_df['title'].parallel_apply(util.text_clean_es).fillna("") 
        train_pt_df['title_clean'] = train_pt_df['title'].parallel_apply(util.text_clean_pt).fillna("") 
        test_es_df['title_clean']  = test_es_df['title'].parallel_apply(util.text_clean_es).fillna("") 
        test_pt_df['title_clean']  = test_pt_df['title'].parallel_apply(util.text_clean_pt).fillna("") 

        # Join
        train_df = pd.concat([train_es_df, train_pt_df]).sort_index()
        test_df  = pd.concat([test_es_df, test_pt_df]).sort_index()

        print(train_df.head())

        # Get Classes
        df_dummies_category = pd.get_dummies(train_df['category'])
        print(df_dummies_category.head())

        if self.with_smooth_labels:
            mask_label = train_df.label_quality.apply(lambda x: 1 if x == 'reliable' else 0).values
            Y          = util.smooth_labels(df_dummies_category.values, mask_label, self.smooth_labels_intensity)
            df_dummies_category = pd.DataFrame(Y, columns = df_dummies_category.columns)

        train_df, val_df = train_test_split(train_df, test_size=self.val_size, random_state=self.seed)

        print(self.output()[0].path)

        train_df.to_csv(self.output()[0].path, index_label='index')
        val_df.to_csv(self.output()[1].path, index_label='index')
        test_df.to_csv(self.output()[2].path, index_label='index')
        df_dummies_category.to_csv(self.output()[3].path, index_label='index')


class TokenizerDataFrames(luigi.Task):
    val_size: float = luigi.FloatParameter(default=0.3)
    seed: int = luigi.IntParameter(default=42)
    num_words: int = luigi.IntParameter(default=5000)
    seq_size: int = luigi.IntParameter(default=20)
    sample: int = luigi.IntParameter(default=20000000)
    with_smooth_labels: bool = luigi.BoolParameter(default=False)
    smooth_labels_intensity: float = luigi.FloatParameter(default=0.7)


    def requires(self):
        return CleanDataFrames(val_size=self.val_size, seed=self.seed, sample=self.sample,
            with_smooth_labels = self.with_smooth_labels, smooth_labels_intensity = self.smooth_labels_intensity)

    def output(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget, luigi.LocalTarget]:
        task_hash = self.task_id.split("_")[-1]
        return (luigi.LocalTarget(os.path.join(DATASET_DIR,
                                               "X_seq_train_%.2f_%d_%s.pkl" % (
                                                   self.num_words, self.seq_size, task_hash))),
                luigi.LocalTarget(os.path.join(DATASET_DIR,
                                               "Y_train_%s.pkl" % (task_hash))),        
                luigi.LocalTarget(os.path.join(DATASET_DIR, 
                                                "X_seq_val_%.2f_%d_%s.pkl" % (self.num_words, self.seq_size, task_hash))),
                luigi.LocalTarget(os.path.join(DATASET_DIR,
                                                "Y_val_%s.pkl" % (task_hash))),        
                luigi.LocalTarget(os.path.join(DATASET_DIR, 
                                                "X_seq_test_%.2f_%d_%s.pkl"% (self.num_words, self.seq_size, task_hash))),
                luigi.LocalTarget(os.path.join(DATASET_DIR, 
                                                "wordindex_%.2f_%d_%s.pkl"% (self.num_words, self.seq_size, task_hash))))

    def run(self):
        train_df = pd.read_csv(os.path.join(self.input()[0].path), index_col='index')
        val_df   = pd.read_csv(os.path.join(self.input()[1].path), index_col='index')
        test_df  = pd.read_csv(os.path.join(self.input()[2].path), index_col='index')
        df_dummies_category = pd.read_csv(os.path.join(self.input()[3].path), index_col='index')

        print(train_df.info())
        
        # Tokenize
        tokenizer  = Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(train_df['title_clean'].astype(str))

        word_index = tokenizer.word_index
        print('\nFound %s unique tokens.\n' % len(word_index))

        # Build dataset
        # X = [..,..,..]
        train_sequences  = tokenizer.texts_to_sequences(train_df['title_clean'].astype(str))
        train_X = pad_sequences(train_sequences, maxlen=self.seq_size)

        val_sequences  = tokenizer.texts_to_sequences(val_df['title_clean'].astype(str))
        val_X   = pad_sequences(val_sequences, maxlen=self.seq_size)

        test_sequences  = tokenizer.texts_to_sequences(test_df['title_clean'].astype(str))
        test_X = pad_sequences(test_sequences, maxlen=self.seq_size)

        # Load
        print("==> Save...")
        with open(self.output()[0].path,'wb') as f:
            np.save(f, train_X) 
        df_dummies_category.loc[train_df.index].to_csv(self.output()[1].path, index_label='index')

        with open(self.output()[2].path,'wb') as f:
            np.save(f, val_X) 
        df_dummies_category.loc[val_df.index].to_csv(self.output()[3].path, index_label='index')

        with open(self.output()[4].path,'wb') as f:
            np.save(f, test_X)     

        with open(self.output()[5].path,'wb') as f:
            pickle.dump(word_index, f)                                

class LoadEmbeddings(luigi.Task):
    val_size: float = luigi.FloatParameter(default=0.3)
    seed: int = luigi.IntParameter(default=42)
    num_words: int = luigi.IntParameter(default=5000)
    seq_size: int = luigi.IntParameter(default=20)
    dim: int = luigi.IntParameter(default=300)
    embedding: str = luigi.ChoiceParameter(choices=["glove", "fasttext"], default="fasttext")
    sample: int = luigi.IntParameter(default=20000000)
    with_smooth_labels: bool = luigi.BoolParameter(default=False)
    smooth_labels_intensity: float = luigi.FloatParameter(default=0.7)

    def requires(self):
        return TokenizerDataFrames(val_size=self.val_size, seed=self.seed, sample=self.sample,
                                        num_words=self.num_words, seq_size=self.seq_size,
                                        with_smooth_labels = self.with_smooth_labels, smooth_labels_intensity = self.smooth_labels_intensity)

    def output(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget, luigi.LocalTarget]:
        task_hash = self.task_id.split("_")[-1]
        return luigi.LocalTarget(os.path.join(DATASET_DIR,
                                               "embedding_matrix_%.2f_%d_%s_%s.pkl" % (
                                                   self.num_words, self.seq_size, self.embedding, task_hash)))

    def run(self):
        with open(self.input()[5].path,'rb') as f:
            word_index = pickle.load(f)
        
        # Build embs
        # https://fasttext.cc/docs/en/crawl-vectors.html
        embeddings_index = {}
        for f in ["embeddins/{}/pt.{}.vec".format(self.embedding, self.dim), 
                    "embeddins/{}/es.{}.vec".format(self.embedding, self.dim)]:
            f = open(os.path.join(DATASET_DIR, f))
            for line in f:
                values = line.split()
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except:
                    pass
            f.close()
            print('Found %s word vectors.' % len(embeddings_index))

        embedding_matrix = np.zeros((len(word_index) + 1, self.dim))
        coverage = 0
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                try:
                   embedding_matrix[i] = embedding_vector
                   coverage = coverage +1
                except:
                   pass            
                   
        print(embedding_matrix.shape)
        print("Total de cobertura dos embs: ", coverage/len(word_index))

        with open(self.output().path,'wb') as f:
            pickle.dump(embedding_matrix, f)        