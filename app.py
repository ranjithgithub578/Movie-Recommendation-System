# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Data
# -------------------------------
movies = pd.read_csv("movies.csv")   # must contain: movieId, title, genres
ratings = pd.read_csv("ratings.csv") # must contain: userId, movieId, rating, timestamp

# -------------------------------
# Content-Based Filtering
# -------------------------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))

def recommend_content(movie_title):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    if movie_title not in indices:
        return pd.DataFrame(columns=['title','genres'])
    idx = indices[movie_title]
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title','genres']]

# -------------------------------
# Collaborative Filtering
# -------------------------------
def recommend_collaborative(user_id, k=5):
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=['title','genres'])
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_item_matrix)
    distances, indices = model_knn.kneighbors(user_item_matrix.loc[user_id].values.reshape(1, -1), n_neighbors=k+1)
    neighbors = indices.flatten()[1:]
    neighbor_ratings = user_item_matrix.iloc[neighbors].mean(axis=0)
    recommended_movies = neighbor_ratings.sort_values(ascending=False).head(5).index
    return movies[movies['movieId'].isin(recommended_movies)][['title','genres']]

# -------------------------------
# Hybrid (simple blend)
# -------------------------------
def recommend_hybrid(user_id, movie_title, alpha=0.5):
    content_df = recommend_content(movie_title)
    collab_df = recommend_collaborative(user_id)
    hybrid = pd.concat([content_df, collab_df]).drop_duplicates().head(5)
    return hybrid

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 Simple Hybrid Movie Recommendation System")

option = st.sidebar.selectbox("Choose Recommendation Type", ["Content-Based", "Collaborative", "Hybrid"])

if option == "Content-Based":
    movie_title = st.text_input("Enter a movie title:")
    if movie_title:
        st.write("Recommended Movies:")
        st.table(recommend_content(movie_title))

elif option == "Collaborative":
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    if user_id:
        st.write("Recommended Movies:")
        st.table(recommend_collaborative(user_id))

elif option == "Hybrid":
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    movie_title = st.text_input("Enter a movie title:")
    if user_id and movie_title:
        st.write("Hybrid Recommendations:")
        st.table(recommend_hybrid(user_id, movie_title))

# -------------------------------
# Visualization
# -------------------------------
st.subheader("Ratings Distribution")
fig, ax = plt.subplots()
sns.histplot(ratings['rating'], bins=5, kde=True, ax=ax)
st.pyplot(fig)
