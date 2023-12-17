import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentRec:
    def __init__(self,feature_mat,dataset):
        """
        Initializes the ContentRec class with a feature matrix and a dataset.

        Parameters:
        - feature_mat: Matrix containing features for each song. Each row corresponds to a song.
        - dataset: DataFrame containing information about each song. Each row corresponds to a song, and rows in feature_mat and dataset align.
        """
        self.feature_mat = feature_mat
        self.dataset = dataset
        
    def recommend_songs(self,k,song_artists_lst):#.song_artists_lst, k, combined_matrix, df=dataset):
        '''
        k: the number of songs we want to recommend
        song_artists_lst: the song_label we know a specific user has clicked, we generate the user's taste vector based on it
        '''
        dataset = self.dataset.copy()

        # Find the row index in the feature matrix corresponding to the clicked song.
        song2idx = dataset['song_label'].reset_index().set_index('song_label')['index'].to_dict()
        song2idx_df = pd.Series(song2idx)
        given_id_list = song2idx_df[song_artists_lst].to_list()
        # calculate vector feature
        taste_vector = np.mean(self.feature_mat[given_id_list,:],axis=0)
        
        # Use cosine_similarity to calculate the distance between each song and the taste vector.
        cosine_similarities = cosine_similarity(taste_vector.reshape(1, -1), self.feature_mat)
        # Select the closest k songs based on distance for recommendation.
        most_similar_indices = np.argsort(cosine_similarities[0])[-(k+len(song_artists_lst)):][::-1]
        k_indices = [idx for idx in most_similar_indices if idx not in given_id_list][:k]

        # return the information of the k songs
        return self.dataset.iloc[k_indices,:]