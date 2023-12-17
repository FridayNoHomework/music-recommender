'''
the HitKCal class is designed for evaluating a recommender using hit@K. 
the class initialization requires a recommender and user data for evaluation, 
and the presented hit@K values are the average hit@K for all users.

hit@K definition:
where for each user who has liked N songs, 
n1 songs are taken to generate recommendations of K songs. 
Set h represents the intersection between the recommended K songs and the remaining (N-n1) liked songs,
hit@K = h / (N-n1).

hit@K can be seen as a function of n1(# of songs used for recommendation) and K(# of recommended songs))
the calculator function in the class receives lists for K (k_list) and n1 (n1_list) as parameters 
and returns a DataFrame with K as the index, n1 as columns, and hit@K values.
'''


import pandas as pd
import random
random_seed = 42
random.seed(random_seed)

class HitKCal:
    def __init__(self,user_data,ContentRec):
        self.user_data = user_data
        self.ContentRec = ContentRec
        self.users = list(user_data['user_id'].unique())
        self.all_results = None
    
    def get_single_user_hit(self,user,k_list,n1_list):
        user_data_used = self.user_data[self.user_data['user_id']==user].reset_index(drop=True)
        songs_lst = user_data_used['song_label'].to_list()
        random.shuffle(songs_lst)
        
        hit_df = pd.DataFrame(index=k_list,columns=n1_list)
        hit_df.index.name = 'k'
        
        for k in k_list:
            for n1 in n1_list:
                songs_train = songs_lst[:n1]
                songs_true = songs_lst[n1:]
                df_rec = self.ContentRec.recommend_songs(k,songs_train)
                songs_pred = df_rec['song_label'].to_list()
                hit_num = len(set(songs_true) & set(songs_pred))
                hit_df.loc[k,n1] = hit_num/len(songs_true)

        return hit_df
    
    def get_all_user_hit(self,k_list,n1_list):
        self.all_results = {}
        for count,user in enumerate(self.users):
            self.all_results[user] = self.get_single_user_hit(user,k_list,n1_list)
            #print("\r", end="")    
            #print("# of users we calculated hit@k: {}/{}: ".format(count,len(self.users)),end="") 
        return self.all_results