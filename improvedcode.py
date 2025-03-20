import cv2
from deepface import DeepFace
import requests
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time

# Configuration
TMDB_API_KEY = "8a2a1e6a69c1cbe641993c4ff16c12f6"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456789",
    "database": "movie_recommender",
}

# Database connection
def get_db_connection():
    try:
        return mysql.connector.connect(**MYSQL_CONFIG)
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Save user preference
def save_preference(movie_id, movie_name, emotion):
    emotion = emotion.lower() if emotion in ["happy", "sad", "angry", "surprise", "fear", "neutral", "disgust"] else "neutral"
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("INSERT INTO preferences (movie_id, movie_name, emotion) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE emotion = %s",
                           (movie_id, movie_name, emotion, emotion))
            connection.commit()
            cursor.close()
        except mysql.connector.Error as e:
            print(f"Error saving preference: {e}")
        finally:
            connection.close()

# Fetch movies from TMDB
def fetch_movies(emotion):
    genre_map = {"happy": 35, "sad": 18, "angry": 28, "surprise": 10749, "fear": 27, "neutral": 9648, "disgust": 53}
    genre_id = genre_map.get(emotion, 18)
    response = requests.get(f"{TMDB_BASE_URL}/discover/movie", params={"api_key": TMDB_API_KEY, "with_genres": genre_id, "sort_by": "popularity.desc"})
    return response.json()["results"] if response.status_code == 200 else []

# Detect emotion from webcam
def detect_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return "neutral"
    
    emotions = []
    detection_time = 5  # Time in seconds
    start_time = time.time()
    
    print("Starting emotion detection. Look at the camera...")
    
    while time.time() - start_time < detection_time:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Analyze emotion
        try:
            result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            emotion = result[0]["dominant_emotion"]
            emotions.append(emotion)
            
            # Display the detected emotion on the frame
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Emotion Detection", frame)
        except Exception as e:
            print(f"Error analyzing frame: {e}")
        
        # Break the loop if 'q' is pressed (optional)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Return the most frequent emotion
    return max(set(emotions), key=emotions.count) if emotions else "neutral"

# Confirm or choose emotion
def confirm_emotion(detected_emotion):
    print(f"\nDetected emotion: {detected_emotion}")
    confirm = input("Is this correct? (y/n): ").strip().lower()
    
    if confirm == "y":
        return detected_emotion
    else:
        print("\nPlease select your current emotion:")
        emotions = ["happy", "sad", "angry", "surprise", "fear", "neutral", "disgust"]
        for i, emotion in enumerate(emotions, 1):
            print(f"{i}. {emotion}")
        
        try:
            choice = int(input("\nEnter the number of your current emotion: "))
            if 1 <= choice <= len(emotions):
                return emotions[choice - 1]
            else:
                print("Invalid choice. Using detected emotion.")
                return detected_emotion
        except ValueError:
            print("Invalid input. Using detected emotion.")
            return detected_emotion

# Recommend movies
def recommend_movies(emotion):
    movies = fetch_movies(emotion)
    if not movies:
        return []
    connection = get_db_connection()
    if connection:
        cursor = connection.cursor()
        cursor.execute("SELECT movie_id FROM preferences")
        preferred_ids = [row[0] for row in cursor.fetchall()]
        cursor.close()
        connection.close()
    else:
        preferred_ids = []
    return [movie for movie in movies if movie["id"] not in preferred_ids][:5]

# Content-based filtering
def similar_movies(movie_id, all_movies):
    movies_df = pd.DataFrame(all_movies)
    movies_df["overview"] = movies_df["overview"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["overview"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = movies_df[movies_df["id"] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    return movies_df.iloc[[i[0] for i in sim_scores]]

# Main function
def main():
    print("Welcome to the Movie Recommender System!")
    
    # Detect emotion
    detected_emotion = detect_emotion()
    
    # Confirm or choose emotion
    final_emotion = confirm_emotion(detected_emotion)
    print(f"Using emotion: {final_emotion}")
    
    # Recommend movies
    recommendations = recommend_movies(final_emotion)
    if recommendations:
        print("\nRecommended movies:")
        for i, movie in enumerate(recommendations, 1):
            print(f"{i}. {movie['title']} ({movie.get('release_date', 'N/A')[:4]})")
        choice = int(input("\nWhich movie do you prefer? (1-5): ")) - 1
        if 0 <= choice < len(recommendations):
            selected_movie = recommendations[choice]
            save_preference(selected_movie["id"], selected_movie["title"], final_emotion)
            print(f"\nSimilar movies to {selected_movie['title']}:")
            similar = similar_movies(selected_movie["id"], fetch_movies(final_emotion))
            for _, movie in similar.iterrows():
                print(f"- {movie['title']} ({movie.get('release_date', 'N/A')[:4]})")
        else:
            print("Invalid choice.")
    else:
        print("No recommendations found.")

if __name__ == "__main__":
    main()