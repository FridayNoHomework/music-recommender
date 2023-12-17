'''
These classes are designed to vectorize specified text columns in a dataframe. 
Initialization requires a list of designated feature names (e.g., ['track_genre', 'artists']) 
and an instance of a text vectorization model.
the W2V class also needs the user_data for training.

Given a dataframe containing the specified text features, 
these classes can vectorize the text columns of the samples and obtain the text feature matrix 
for all samples using their respective member functions.
'''



import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument,Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from helpers import textFeatures2list_series

FEATURE = 'text_feature'
USER = 'user_id'

class GetTextMat_W2V:
    def __init__(self, model, user_data, cols_for_soup):
        """
        Initialize an instance of the GetTextMat_W2V class.

        Parameters:
        - model: An instance of the Word2Vec model.
        - user_data: A Pandas DataFrame containing user data, used for training the model.
        - cols_for_soup: A list containing the column names of text features in the user data.
        """
        self.model = model
        self.user_data = user_data
        self.cols_for_soup = cols_for_soup
    
    def get_train_text(self):
        """
        Get the text data used for training the Word2Vec model.

        Returns:
        - list: A list containing lists of text features for each user's playlist.
        """
        # Merge text feature columns in user data
        self.user_data.loc[:, FEATURE] = textFeatures2list_series(self.user_data, self.cols_for_soup)
        
        # Explode text features for each user
        user_data_ex = self.user_data.explode(FEATURE)
        self.user_textFeature = user_data_ex.groupby(USER)[FEATURE].unique().apply(list)
        
        # Return a list containing lists of text features for each user's playlist
        return list(self.user_textFeature.to_dict().values())
    
    def train(self):
        """
        Train the Word2Vec model.
        """
        self.train_text = self.get_train_text()
        self.model.build_vocab(self.train_text)
        self.model.train(self.train_text, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def get_vector(self, word):
        """
        Get the vector representation of a specified word in the Word2Vec model.
        This is a utility function for obtaining the vector representation of a word.

        Parameters:
        - word: The word for which to obtain the vector.

        Returns:
        - np.ndarray: The vector representation of the word.
        """
        try:
            return self.model.wv[word]
        except KeyError:
            return np.zeros(self.model.vector_size)

    def transform(self, song_data):
        """
        Transform the text features of song data into a text feature matrix.

        Parameters:
        - song_data: A Pandas DataFrame containing song data.

        Returns:
        - np.ndarray: A matrix containing the mean text feature for each song.
        """
        # Process the text feature column of song data and train the Word2Vec model
        song_data[FEATURE] = textFeatures2list_series(song_data, self.cols_for_soup)
        self.train()
        
        # Generate a text feature matrix, taking the mean of each song's text features
        feature_mat = song_data[FEATURE].apply(lambda x: np.mean([self.get_vector(word) for word in x], axis=0))
        return np.vstack(feature_mat)

# the following is used as benchmark
class GetTextMat_SK:
    def __init__(self,cols_for_soup,model):
        self.cols_for_soup = cols_for_soup
        self.model = model
        #self.text_matrix = None
        self.text_feature_names = None
        
    def get_text_soup(self,dataset):
        soup_col = dataset.apply(lambda row: ' '.join(str(row[col]) for col in self.cols_for_soup), axis=1)
        return soup_col
    
    def fit(self,dataset):
        soup_col = self.get_text_soup(dataset)
        self.model.fit(soup_col)
        self.text_feature_names = self.model.get_feature_names_out()
        
    def transform(self,dataset):
        # get text matrix
        soup_col = self.get_text_soup(dataset)
        return self.model.transform(soup_col).toarray()
    
    def fit_transform(self,dataset):
        self.fit(dataset)
        return self.transform(dataset)
    
class GetTextMat_D2V(GetTextMat_SK):
        
    def fit(self, dataset):
        soup_col = self.get_text_soup(dataset)
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(soup_col)]
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def transform(self, dataset):
        soup_col = self.get_text_soup(dataset)
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(soup_col)]

        vector_list = []
        for ii in range(len(tagged_data)):
            vector = self.model.infer_vector(tagged_data[ii].words)
            vector_list.append(vector)
        text_matrix = np.vstack(vector_list)
        return text_matrix