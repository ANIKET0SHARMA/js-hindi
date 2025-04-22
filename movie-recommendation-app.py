import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import re

# Set page configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# CSS styling
st.markdown("""
<style>
    .movie-card {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #f0f2f6;
    }
    .movie-title {
        font-weight: bold;
        font-size: 16px;
    }
    .movie-details {
        font-size: 14px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorites based on content features!")

# Function to load data
@st.cache_data
def load_data():
    # Download movie data from MovieLens dataset
    movies = pd.read_csv('https://raw.githubusercontent.com/microsoft/recommenders/main/examples/01_prepare_data/movielens/movies.csv')
    # Extract year from title if available
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('float')
    # Clean title by removing year
    movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
    
    # Extract genres and convert to list
    movies['genres'] = movies['genres'].str.split('|')
    
    # Create a simplified feature for recommendation
    movies['features'] = movies['genres'].apply(lambda x: ' '.join(x))
    
    return movies

# Function to get movie recommendations
def get_recommendations(movie_id, cosine_sim, movies_df):
    # Get the index of the movie
    idx = movies_df[movies_df['movieId'] == movie_id].index[0]
    
    # Get similarity scores with other movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 10 similar movies (excluding itself)
    sim_scores = sim_scores[1:11]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return top 10 similar movies with similarity scores
    recommendations = movies_df.iloc[movie_indices].copy()
    recommendations['similarity'] = [i[1] for i in sim_scores]
    return recommendations

# Load data
try:
    movies_df = load_data()
    
    # Create feature vectors
    vectorizer = CountVectorizer()
    feature_vectors = vectorizer.fit_transform(movies_df['features'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(feature_vectors)
    
    # Success message
    st.success(f"Loaded data for {len(movies_df)} movies!")
    
    # Sidebar for searching and filtering
    st.sidebar.header("Search for a Movie")
    
    # Search by title
    search_query = st.sidebar.text_input("Enter movie title:")
    if search_query:
        filtered_movies = movies_df[movies_df['clean_title'].str.contains(search_query, case=False, na=False)]
        if filtered_movies.empty:
            st.sidebar.warning("No movies found with that title.")
        else:
            st.sidebar.write(f"Found {len(filtered_movies)} movies")
            selected_movie = st.sidebar.selectbox(
                "Select a movie:",
                filtered_movies['clean_title'].tolist(),
                index=0
            )
            
            # Get selected movie details
            movie_details = filtered_movies[filtered_movies['clean_title'] == selected_movie].iloc[0]
            movie_id = movie_details['movieId']
            
            # Display selected movie info
            st.header("Selected Movie")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Placeholder image (in a real app, you would get poster image from an API)
                st.image("https://via.placeholder.com/200x300?text=Movie+Poster", width=200)
                
            with col2:
                st.subheader(movie_details['title'])
                st.write(f"**Genres:** {', '.join(movie_details['genres'])}")
                if not np.isnan(movie_details['year']):
                    st.write(f"**Year:** {int(movie_details['year'])}")
                
            # Get recommendations
            recommendations = get_recommendations(movie_id, cosine_sim, movies_df)
            
            # Display recommendations
            st.header("Recommended Movies")
            
            # Use columns for better layout
            cols = st.columns(2)
            
            for i, (_, movie) in enumerate(recommendations.iterrows()):
                col_idx = i % 2
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="movie-details">
                            <p>Genres: {', '.join(movie['genres'])}</p>
                            <p>Similarity Score: {movie['similarity']:.2f}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Filter by genre
    st.sidebar.header("Filter by Genre")
    all_genres = set()
    for genres in movies_df['genres']:
        all_genres.update(genres)
    
    selected_genre = st.sidebar.selectbox("Select a genre:", sorted(all_genres))
    
    if selected_genre:
        # Get movies with the selected genre
        genre_movies = movies_df[movies_df['genres'].apply(lambda x: selected_genre in x)]
        
        if not genre_movies.empty:
            st.header(f"Top Movies in {selected_genre} Genre")
            
            # Display a sample of movies in that genre
            sample_movies = genre_movies.sample(min(6, len(genre_movies)))
            
            # Use columns for better layout
            cols = st.columns(3)
            
            for i, (_, movie) in enumerate(sample_movies.iterrows()):
                col_idx = i % 3
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="movie-details">
                            <p>Genres: {', '.join(movie['genres'])}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info("""
    This content-based movie recommendation system uses features like genre to find similar movies.
    
    The recommendations are generated using cosine similarity between movie feature vectors.
    
    Data source: MovieLens dataset.
    """)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.write("Please check the data source or try again later.")
