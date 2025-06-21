import sys
import io
import cv2
from deepface import DeepFace
import requests
import mysql.connector
from mysql.connector import Error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time

# Change console encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# TMDB API Configuration
TMDB_API_KEY = "8a2a1e6a69c1cbe641993c4ff16c12f6"  
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# MySQL Database Configuration
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",  
    "password": "123456789",  
    "database": "movie_recommender",
}

# Initialize MySQL connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        print("Connected to MySQL database successfully!")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Save user preference to the database
def save_preference(movie_id, movie_name, emotion):
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            # Debug: Print the query and values being inserted
            query = "INSERT INTO preferences (movie_id, movie_name, emotion) VALUES (%s, %s, %s)"
            values = (movie_id, movie_name, emotion)
            print(f"Executing query: {query} with values: {values}")  # Debug statement
            cursor.execute(query, values)
            connection.commit()
            cursor.close()
            print(f"Saved preference: Movie ID={movie_id}, Movie Name={movie_name}, Emotion={emotion}")
        except Error as e:
            print(f"Error saving preference: {e}")
        finally:
            connection.close()
    else:
        print("Failed to connect to the database.")  # Debug statement

# Fetch user's past preferences
def get_user_preferences():
    connection = get_db_connection()
    preferences = []
    if connection:
        try:
            cursor = connection.cursor()
            query = "SELECT movie_id, movie_name, emotion FROM preferences"
            print(f"Executing query: {query}")  # Debug statement
            cursor.execute(query)
            preferences = cursor.fetchall()
            cursor.close()
            print("Fetched preferences:", preferences)  # Debug statement
        except Error as e:
            print(f"Error fetching preferences: {e}")
        finally:
            connection.close()
    else:
        print("Failed to connect to the database.")  # Debug statement
    return preferences

# Fetch movies from TMDB API based on genre or emotion
def fetch_movies_from_tmdb(emotion):
    # Map emotions to TMDB genres
    emotion_to_genre = {
        "happy": 35,  # Comedy
        "sad": 18,    # Drama
        "angry": 28,  # Action
        "surprise": 10749,  # Romance
        "fear": 27,  # Horror
        "neutral": 9648,  # Mystery
    }
    genre_id = emotion_to_genre.get(emotion, 18)  # Default to Drama
    url = f"{TMDB_BASE_URL}/discover/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "with_genres": genre_id,
        "sort_by": "popularity.desc",
        "page": 1,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        return []

# Recommend movies based on emotion and past preferences
def recommend_movies(emotion):
    # Fetch movies from TMDB
    movies = fetch_movies_from_tmdb(emotion)
    if not movies:
        return []

    # Fetch user preferences
    preferences = get_user_preferences()
    preferred_movie_ids = [pref[0] for pref in preferences]

    # Filter out movies the user has already seen
    recommended_movies = [movie for movie in movies if movie["id"] not in preferred_movie_ids]
    return recommended_movies[:5]  # Return top 5 recommendations

# Real-time emotion detection using the camera
def detect_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return "neutral"

    emotion = "neutral"  # Default emotion
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Detect emotion using DeepFace
        try:
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            emotion = result[0]["dominant_emotion"]
            cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error: {e}")
            emotion = "neutral"

        # Show the frame
        cv2.imshow("Emotion Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return emotion

# Content-based filtering: Find similar movies
def content_based_filtering(preferred_movie_id, all_movies):
    # Create a DataFrame for all movies
    movies_df = pd.DataFrame(all_movies)

    # Use TF-IDF to vectorize movie overviews
    tfidf = TfidfVectorizer(stop_words="english")
    movies_df["overview"] = movies_df["overview"].fillna("")  # Fill missing overviews
    tfidf_matrix = tfidf.fit_transform(movies_df["overview"])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the preferred movie
    preferred_movie_index = movies_df[movies_df["id"] == preferred_movie_id].index[0]

    # Get similarity scores for the preferred movie
    sim_scores = list(enumerate(cosine_sim[preferred_movie_index]))

    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 5 similar movies (excluding the preferred movie itself)
    sim_scores = sim_scores[1:6]
    similar_movie_indices = [i[0] for i in sim_scores]

    # Return similar movies
    return movies_df.iloc[similar_movie_indices]

# Main function
def main():
    # Detect user's emotion
    print("Detecting your emotion...")
    emotion = detect_emotion()
    print(f"Detected Emotion: {emotion}")

    # Fetch and display user's past preferences
    preferences = get_user_preferences()
    if preferences:
        print("\nYour past preferences:")
        for pref in preferences:
            movie_id, movie_name, emotion = pref
            print(f"Movie: {movie_name} (ID: {movie_id}), Emotion: {emotion}")

    # Recommend movies
    recommendations = recommend_movies(emotion)
    if recommendations:
        print("\nHere are your movie recommendations:")
        for i, movie in enumerate(recommendations, start=1):
            print(f"{i}. {movie['title']} ({movie['release_date'][:4]}) - {movie['overview']}")

        # Ask the user which movie they prefer
        try:
            choice = int(input("\nWhich movie do you prefer? Enter the number: "))
            if 1 <= choice <= len(recommendations):
                preferred_movie = recommendations[choice - 1]
                print(f"\nYou selected: {preferred_movie['title']}")

                # Save the preference to the database
                save_preference(preferred_movie["id"], preferred_movie["title"], emotion)

                # Fetch all movies for content-based filtering
                all_movies = fetch_movies_from_tmdb(emotion)

                # Get similar movies using content-based filtering
                similar_movies = content_based_filtering(preferred_movie["id"], all_movies)

                # Display similar movies
                print("\nHere are some similar movies you might like:")
                for i, movie in similar_movies.iterrows():
                    print(f"{movie['title']} ({movie['release_date'][:4]}) - {movie['overview']}")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    else:
        print("No recommendations found.")

# Run the program
if __name__ == "__main__":
    main()
