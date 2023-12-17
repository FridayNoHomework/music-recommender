from metaflow import FlowSpec, step, Parameter, IncludeFile, current
import numpy as np
import pandas as pd
import random
random_seed = 42
random.seed(random_seed)

# from comet_ml import Experiment
# assert 'COMET_API_KEY' in os.environ and os.environ['COMET_API_KEY']


class recommend_flow(FlowSpec):
    COMMON_DATA = IncludeFile(
        name = 'common_data',
        default= '../data/processed_data/common.csv')
    
    processd_data_file = '../data/processed_data/'
    serialized_data_file = '../data/serialized_data/'
    
    used_song_cols = ['track_id', 'artists', 'album_name', 'track_name',
    'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
    'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
    'track_genre','song_label']
    
    # we use following text features and the Word2Vec model is used to vectorize them
    cols_for_soup = ['artists','track_genre']

    # We are removing the "energy" attribute due to two main reasons:
    # 1) It exhibits high correlation with other numerical features.
    # 2) Its impact can be assimilated by other existing features.
    num_cols = ['popularity', 'duration_ms', 'danceability',# 'energy',
                'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo']
    
    # the proportions of users used in training, validation, and testing datasets for the Word2Vec model
    train_user_ratio = 12/14
    val_user_ratio = 1/14
    test_user_ratio = 1/14

    # In the validation and testing phases, we exclusively utilize data from users whose liked song count surpasses a specific threshold.
    liked_song_threshhold = 10

    # the hyperparameter we want to tune in Word2Vector model
    window_size = [2,5,10]

    # parameters for hit@k metrics
    # The metrics data will be organized as a dataframe, where each value corresponds to a combination of 'k' and 'n1'
    # The representative values for each metric are calculated by taking the mean across the dataframe
    k_list = [100,150,300]
    n1_list = list(range(2, 10, 2))

    @step
    def start(self):
        from io import StringIO
        self.common_df = pd.read_csv(StringIO(self.COMMON_DATA))
        #self.common_df = pd.read_csv('common.csv')
        self.common_df = self.common_df.dropna()
        self.next(self.prepare_song_data)

    @step
    def prepare_song_data(self):

        self.song_data = self.common_df[self.used_song_cols].drop_duplicates().reset_index(drop=True)
        # save the song_data for later demo
        self.song_data.to_csv(self.processd_data_file+'song_data.csv')

        self.next(self.prepare_user_data)
    
    @step
    def prepare_user_data(self):

        self.all_user_data = self.common_df.loc[:,['user_id', 'song_label'] + self.cols_for_soup].copy()

        # calculate the number of songs each user likes
        self.all_user_data['num_per_user'] = self.all_user_data.groupby('user_id')['song_label'].transform('nunique')
        self.all_users = list(self.all_user_data.user_id.unique())

        # split all user data into training, validation and testing sets
        train_end_point = int(self.train_user_ratio*len(self.all_users))
        val_end_point = train_end_point + int(self.val_user_ratio*len(self.all_users))

        self.tarin_users = self.all_users[:train_end_point]
        self.val_users = self.all_users[train_end_point:val_end_point]
        self.test_users = self.all_users[val_end_point:]

        self.train_user_data = self.all_user_data[self.all_user_data['user_id'].isin(self.tarin_users)]
        self.val_user_data   = self.all_user_data[self.all_user_data['user_id'].isin(self.val_users)]
        self.test_user_data  = self.all_user_data[self.all_user_data['user_id'].isin(self.test_users)]

        # In the validation and testing phases, we exclusively utilize data from users whose liked song count surpasses a specific threshold.
        self.val_user_data = self.val_user_data[self.val_user_data['num_per_user']>self.liked_song_threshhold].reset_index(drop=True)
        self.test_user_data = self.test_user_data[self.test_user_data['num_per_user']>self.liked_song_threshhold].reset_index(drop=True)


        self.next(self.get_feature_mat,foreach='window_size')

    @step
    def get_feature_mat(self):
        from gensim.models import Word2Vec
        from GetTextMat import GetTextMat_W2V
        from sklearn.preprocessing import MinMaxScaler
        pretrained_w2v_model = Word2Vec(vector_size=100, 
                                        window=self.input, 
                                        min_count=1, 
                                        workers=4) 
        self.vectorizer = GetTextMat_W2V(pretrained_w2v_model,self.train_user_data,self.cols_for_soup)
        self.text_mat = self.vectorizer.transform(self.song_data)

        self.num_mat = self.song_data[self.num_cols].values
        self.num_mat = MinMaxScaler().fit_transform(self.num_mat) 

        self.feature_mat = np.hstack([self.text_mat,self.num_mat])

        self.next(self.join)
    
    @step
    def join(self,inputs):
        '''
        In the "join" step, 
        models based on different window sizes generate feature matrices used for recommending songs to users in the validation set. 

        The hit@K results for these recommendations are recorded, 
        with the format being a dictionary where the key represents the window size and the value is the corresponding hit@K result. 
        The recorded results are then saved as a pickle file.

        The w2v model with highest mean Hit@K will be picked as the one we further use.
        the vector space generated based on this best model is saved and will be used for further test and demo app.
        '''
        from ContentRec import ContentRec
        from HitKCal import HitKCal
        from helpers import mean_df
        import pickle

        self.song_data = inputs[0].song_data
        self.val_user_data = inputs[0].val_user_data
        self.test_user_data = inputs[0].test_user_data
        self.all_user_data = inputs[0].all_user_data
        self.num_mat = inputs[0].num_mat
        self.song_data = inputs[0].song_data

        self.window_hitk = {}
        self.window_mat = {}
        hitK_means = []

        for ii,input in enumerate(inputs):

            self.window_mat[self.window_size[ii]] = input.feature_mat

            cr = ContentRec(input.feature_mat,self.song_data)
            hitK = HitKCal(self.val_user_data,cr)
            result_dict = hitK.get_all_user_hit(self.k_list,self.n1_list)
            hitk_df = mean_df(result_dict.values())
            self.window_hitk[self.window_size[ii]] = hitk_df
            print('when window = {}, the mean hit@K  = {}'.format(self.window_size[ii],hitk_df.values.mean()))
            hitK_means.append(hitk_df.values.mean())

        # # record in comet 
        # for model_rlt in self.window_hitk.items():
        #     exp = Experiment(project_name = "Double11_FinalProject_val", auto_param_logging = False)
        #     exp.log_metrics({"window":model_rlt[0]})
        #     exp.log_metrics({"Hit@K_mean":model_rlt[1].values.mean()})
            
        # pick the best W2V model based on mean Hit@K (highest -> best)
        helper_array = np.array(hitK_means)
        best_id = np.argmax(helper_array)
        self.best_window = self.window_size[best_id]
        print("so the final W2V model we choose is with 'window' equal to {}".format(self.best_window))

        # save the vector space generated by the best model
        self.mat_w2v = self.window_mat[self.best_window]
        with open(self.serialized_data_file+'w2v_mat.pkl', 'wb') as file:
            pickle.dump(self.mat_w2v, file)
        

        self.next(self.compare_benchmark)

    @step
    def compare_benchmark(self):
        from ContentRec import ContentRec
        from GetTextMat import GetTextMat_SK
        from sklearn.feature_extraction.text import TfidfVectorizer
        from HitKCal import HitKCal
        from helpers import mean_df
        import pickle

        def evaluate_pipeline(feature_mat):
            cr = ContentRec(feature_mat,self.song_data)
            hitK = HitKCal(self.test_user_data,cr)# use test set to compare different models
            result_dict = hitK.get_all_user_hit(self.k_list,self.n1_list)
            hitk_df = mean_df(result_dict.values())
            return hitk_df
        
        # the dictionary used to record testing result
        self.record_dict = {}

        # benchmark1
        self.record_dict['benchmark1_num'] = evaluate_pipeline(self.num_mat)
        print('the mean hit@K of benchmark1(numerical features only): {}'.format(self.record_dict['benchmark1_num'].values.mean()))
        
        # benckmark2
        tfidf_model = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.003,max_df=0.5,max_features=500)
        vectorizer_ti = GetTextMat_SK(self.cols_for_soup, tfidf_model)
        text_mat_ti = vectorizer_ti.fit_transform(self.song_data)
        mat_ti = np.hstack([text_mat_ti,self.num_mat])

        self.record_dict['benchmark2_ti'] = evaluate_pipeline(mat_ti)
        print('the mean hit@K of benchmark2(tf-idf): {}'.format(self.record_dict['benchmark2_ti'].values.mean()))
        
        # w2v
        self.record_dict['w2v'] = evaluate_pipeline(self.mat_w2v)
        print('the mean hit@K of w2v model: {}'.format(self.record_dict['w2v'].values.mean()))

        
        # for model_rlt in self.record_dict.items():
        #     exp = Experiment(project_name = "Double11_FinalProject_Test", auto_param_logging = False)
        #     exp.log_parameters({"model":model_rlt[0]})
        #     exp.log_metrics({"Hit@K_mean":model_rlt[1].values.mean()})

        with open(self.serialized_data_file+'test_result.pkl', 'wb') as file:
            pickle.dump(self.record_dict, file)
        
        self.next(self.end)

    @step
    def end(self):
        print('end')


if __name__ == "__main__":
    recommend_flow()