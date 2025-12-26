# ============================================
# Movie Recommendation System
# Content-Based Filtering (TMDB Dataset)
# ============================================

import pandas as pd
import os
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================
# 1. LOAD DATASETS (PATH SAFE)
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

movies_path = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
credits_path = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

print("Datasets loaded successfully")


# ============================================
# 2. MERGE MOVIES & CREDITS
# ============================================

movies = movies.merge(credits, left_on="id", right_on="movie_id")
print("Datasets merged")


# ============================================
# 3. FIX TITLE COLUMN (VERY IMPORTANT)
# ============================================

print("Available columns:\n", movies.columns)

if "title" in movies.columns:
    title_col = "title"
elif "original_title" in movies.columns:
    title_col = "original_title"
elif "movie_title" in movies.columns:
    title_col = "movie_title"
else:
    raise Exception("No movie title column found!")

movies.rename(columns={title_col: "title"}, inplace=True)


# ============================================
# 4. SELECT REQUIRED COLUMNS
# ============================================

movies = movies[["title", "overview", "cast", "crew"]]

movies["overview"] = movies["overview"].fillna("")
movies["cast"] = movies["cast"].fillna("")
movies["crew"] = movies["crew"].fillna("")


# ============================================
# 5. HELPER FUNCTIONS TO CLEAN DATA
# ============================================

def get_top_actors(cast):
    names = []
    try:
        cast = ast.literal_eval(cast)
        for person in cast[:3]:  # top 3 actors
            names.append(person["name"])
    except:
        pass
    return " ".join(names)


def get_director(crew):
    try:
        crew = ast.literal_eval(crew)
        for person in crew:
            if person["job"] == "Director":
                return person["name"]
    except:
        pass
    return ""


# ============================================
# 6. CLEAN CAST & CREW
# ============================================

movies["cast"] = movies["cast"].apply(get_top_actors)
movies["crew"] = movies["crew"].apply(get_director)


# ============================================
# 7. CREATE COMBINED TEXT FEATURE
# ============================================

movies["combined_text"] = (
    movies["overview"] + " " +
    movies["cast"] + " " +
    movies["crew"]
)


# ============================================
# 8. TF-IDF VECTORIZATION
# ============================================

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["combined_text"])

print("Text converted into numeric form")


# ============================================
# 9. COSINE SIMILARITY
# ============================================

similarity_matrix = cosine_similarity(tfidf_matrix)


# ============================================
# 10. RECOMMENDATION FUNCTION
# ============================================

def recommend_movie(movie_name, top_n=5):
    if movie_name not in movies["title"].values:
        return ["Movie not found in database"]

    index = movies[movies["title"] == movie_name].index[0]

    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True
    )

    recommendations = []
    for i in similarity_scores[1:top_n + 1]:
        recommendations.append(movies.iloc[i[0]]["title"])

    return recommendations


# ============================================
# 11. TEST THE SYSTEM
# ============================================

test_movie = "Avatar"

print(f"\nMovies similar to '{test_movie}':")
print(recommend_movie(test_movie))
