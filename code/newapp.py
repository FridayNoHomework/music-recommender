'''
This demo webpage shows two columns of songs, allowing users to select their favorite from each column. 
users have the flexibility to update the song list in a specific column if they don't find a song they like.

Based on these two selected songs, the webpage generates and displays a list of ten recommended songs for the user.

The vector space utilized in this demo is generated from the previously selected and optimized Word2Vec model. 
This vector space, corresponding to each row in the song_data, has been precomputed in earlier steps. 
In the demo, it is seamlessly retrieved for use, providing an efficient and ready-to-use foundation for song recommendations.
'''


import streamlit as st
import pandas as pd
from ContentRec import ContentRec
import pickle

with open('../data/serialized_data/w2v_mat.pkl', 'rb') as file:
    feature_mat = pickle.load(file)

dataset = pd.read_csv('../data/processed_data/song_data.csv')

def main():

    cr = ContentRec(feature_mat,dataset) # generate recommender based on the matrix and song_data

    st.title('Music Recommendation System')
    st.header('Pick your favorite songs!')


    def resample_songs(column_name):
        st.session_state[column_name] = dataset[dataset['popularity'] >= 70].sample(10)

# Load or resample songs
    if 'sample_songs_1' not in st.session_state or 'sample_songs_2' not in st.session_state:
        st.session_state.sample_songs_1 = dataset[dataset['popularity'] >= 70].sample(10)
        st.session_state.sample_songs_2 = dataset[dataset['popularity'] >= 70].sample(10)

    col1, col2 = st.columns(2)

    # Create a refresh button for the first column
    if col1.button('Refresh List1'):
        resample_songs('sample_songs_1')

    # Create a refresh button for the second column
    if col2.button('Refresh List2'):
        resample_songs('sample_songs_2')

    with col1:
        st.write("Set 1")
        # Use 'song_label' for selection
        song_labels_1 = st.session_state.sample_songs_1['song_label']
        selected_song_label_1 = st.radio('Select a song from set 1:', song_labels_1, key="set_1")

    with col2:
        st.write("Set 2")
        # Use 'song_label' for selection
        song_labels_2 = st.session_state.sample_songs_2['song_label']
        selected_song_label_2 = st.radio('Select a song from set 2:', song_labels_2, key="set_2")

    if st.button("Get Recommendations"):

        song_label_list = [selected_song_label_1,selected_song_label_2]
        recommendations = cr.recommend_songs(10,song_label_list) #used recommende to recommend 10 songs based on user's choices

        # show the recommendations
        st.subheader('*Your recommendations are here! Enjoy!* :sunglasses:')
        st.table(recommendations[['track_name','artists','album_name', 'track_genre', 'popularity']])

if __name__ == "__main__":
    main()