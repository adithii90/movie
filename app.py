import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# SAMPLE MOVIE DATA
# -----------------------------
movies = pd.DataFrame({
    'title': [
        'Inception',
        'Interstellar',
        'The Dark Knight',
        'Tenet',
        'Avatar',
        'Titanic',
        'The Prestige',
        'Avengers Endgame',
        'Doctor Strange',
        'John Wick'
    ],
    'genre': [
        'Sci-Fi Thriller',
        'Sci-Fi Drama',
        'Action Crime',
        'Sci-Fi Action',
        'Sci-Fi Adventure',
        'Romance Drama',
        'Mystery Thriller',
        'Action Superhero',
        'Fantasy Action',
        'Action Thriller'
    ],
    'rating': [
        8.8,
        8.6,
        9.0,
        7.5,
        7.8,
        7.9,
        8.5,
        8.4,
        7.5,
        7.4
    ],
    'poster': [
        'https://image.tmdb.org/t/p/w500/qmDpIHrmpJINaRKAfWQfftjCdyi.jpg',
        'https://image.tmdb.org/t/p/w500/rAiYTfKGqDCRIIqo664sY9XZIvQ.jpg',
        'https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
        'https://image.tmdb.org/t/p/w500/k68nPLbIST6NP96JmTxmZijEvCA.jpg',
        'https://image.tmdb.org/t/p/w500/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg',
        'https://image.tmdb.org/t/p/w500/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg',
        'https://image.tmdb.org/t/p/w500/5MXyQfz8xUP3dIFPTubhTsbFY6N.jpg',
        'https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg',
        'https://image.tmdb.org/t/p/w500/uGBVj3bEbCoZbDjjl9wTxcygko1.jpg',
        'https://image.tmdb.org/t/p/w500/fZPSd91yGE9fCcCe6OoQr6E3Bev.jpg'
    ],
    'trailer': [
        'https://www.youtube.com/watch?v=YoHD9XEInc0',
        'https://www.youtube.com/watch?v=zSWdZVtXT7E',
        'https://www.youtube.com/watch?v=EXeTwQWrcwY',
        'https://www.youtube.com/watch?v=L3pk_TBkihU',
        'https://www.youtube.com/watch?v=5PSNL1qE6VY',
        'https://www.youtube.com/watch?v=kVrqfYjkTdQ',
        'https://www.youtube.com/watch?v=RLtaA9fFNXU',
        'https://www.youtube.com/watch?v=TcMBFSGVi1c',
        'https://www.youtube.com/watch?v=HSzx-zryEgM',
        'https://www.youtube.com/watch?v=2AUmvWm5ZDQ'
    ]
})

# -----------------------------
# CREATE SIMILARITY MATRIX
# -----------------------------
cv = CountVectorizer()

vectors = cv.fit_transform(movies['genre']).toarray()

similarity = cosine_similarity(vectors)

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append({
            "title": movies.iloc[i[0]].title,
            "rating": movies.iloc[i[0]].rating,
            "poster": movies.iloc[i[0]].poster,
            "trailer": movies.iloc[i[0]].trailer
        })

    return recommended_movies

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🎥 Movie Recommender")

page = st.sidebar.radio(
    "Go To",
    ["Home", "Recommendations", "Watchlist"]
)

# -----------------------------
# WATCHLIST SESSION
# -----------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":

    st.title("🎬 Movie Recommendation System")

    st.markdown("### Find movies you will love!")

    st.image(
        "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba",
        use_container_width=True
    )

    st.subheader("Top Movies")

    cols = st.columns(5)

    for idx, col in enumerate(cols):
        with col:
            st.image(movies.iloc[idx].poster)
            st.write(f"⭐ {movies.iloc[idx].rating}")
            st.caption(movies.iloc[idx].title)

# -----------------------------
# RECOMMENDATION PAGE
# -----------------------------
elif page == "Recommendations":

    st.title("🍿 Get Movie Recommendations")

    selected_movie = st.selectbox(
        "Select a Movie",
        movies['title'].values
    )

    if st.button("Recommend Movies"):

        recommendations = recommend(selected_movie)

        cols = st.columns(5)

        for idx, movie in enumerate(recommendations):

            with cols[idx]:

                st.image(movie['poster'])

                st.subheader(movie['title'])

                st.write(f"⭐ Rating: {movie['rating']}")

                st.markdown(
                    f"[▶ Watch Trailer]({movie['trailer']})"
                )

                if st.button(
                    f"Add to Watchlist",
                    key=movie['title']
                ):
                    st.session_state.watchlist.append(movie)

                    st.success(
                        f"{movie['title']} added!"
                    )

# -----------------------------
# WATCHLIST PAGE
# -----------------------------
elif page == "Watchlist":

    st.title("❤️ Your Watchlist")

    if len(st.session_state.watchlist) == 0:

        st.warning("No movies added yet!")

    else:

        for movie in st.session_state.watchlist:

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(movie['poster'])

            with col2:
                st.subheader(movie['title'])
                st.write(f"⭐ Rating: {movie['rating']}")
                st.markdown(
                    f"[▶ Watch Trailer]({movie['trailer']})"
                )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
