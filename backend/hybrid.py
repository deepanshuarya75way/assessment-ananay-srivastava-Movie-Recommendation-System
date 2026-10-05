import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MovieGraphEngine:
    def __init__(self, data_path="data/movielens_100k.csv"):
        # Resolve path relative to project root if needed
        self.data_path = data_path
        self.movies_df = None
        self.content_sim = None
        self.collab_sim = None
        self.movie_indices = {}
        self.reverse_indices = {}
        self.load_and_train()

    def load_and_train(self):
        # Try finding the CSV in common locations
        paths_to_try = [self.data_path, "../data/movielens_100k.csv", "backend/data/movielens_100k.csv"]
        loaded_df = None
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    required_cols = {'MovieID', 'Title', 'Genres'}
                    if required_cols.issubset(df.columns):
                        df = df.rename(columns={'MovieID': 'movieId', 'Title': 'title', 'Genres': 'genres'})
                        self.movies_df = df[['movieId', 'title', 'genres']].drop_duplicates(subset=['movieId']).reset_index(drop=True)
                        
                        if 'UserID' in df.columns and 'Rating' in df.columns:
                            user_item = df.pivot_table(index='UserID', columns='movieId', values='Rating').fillna(0)
                            user_item = user_item.reindex(columns=self.movies_df['movieId'], fill_value=0)
                            self.collab_sim = cosine_similarity(user_item.T, user_item.T)
                        else:
                            self._create_mock_collab()
                        loaded_df = True
                        break
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        
        if loaded_df is None:
            self._create_mock_data()

        # Build lookup indices
        for idx, row in self.movies_df.iterrows():
            m_id = int(row['movieId'])
            self.movie_indices[m_id] = idx
            self.reverse_indices[idx] = m_id

        # Content Similarity (Genre-based TF-IDF)
        self.movies_df['genres_text'] = self.movies_df['genres'].fillna('').str.replace('|', ' ')
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies_df['genres_text'])
        self.content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Fallback collaborative similarity if not built from pivot table
        if self.collab_sim is None:
            self._create_mock_collab()

    def _create_mock_collab(self):
        np.random.seed(42)
        n_movies = len(self.movies_df)
        user_item_matrix = np.random.rand(50, n_movies)
        self.collab_sim = cosine_similarity(user_item_matrix.T, user_item_matrix.T)

    def _create_mock_data(self):
        self.movies_df = pd.DataFrame({
            'movieId': [242, 302, 377, 51, 346, 474],
            'title': ['Kolya (1996)', 'L.A. Confidential (1997)', 'Heavyweights (1994)', 'Legends of the Fall (1994)', 'Jackie Brown (1997)', 'Dr. Strangelove'],
            'genres': ['Comedy', 'Crime|Film-Noir|Mystery|Thriller', "Children's|Comedy", 'Drama|Romance|War|Western', 'Crime|Drama', 'Sci-Fi|War']
        })

    def get_related_movies(self, movie_id: int, threshold: float = 0.15):
        if movie_id not in self.movie_indices:
            return None

        idx = self.movie_indices[movie_id]
        content_scores = self.content_sim[idx]
        collab_scores = self.collab_sim[idx]

        connections = []
        for target_id, i in self.movie_indices.items():
            if i == idx:
                continue
            
            c_score = float(content_scores[i])
            b_score = float(collab_scores[i])
            
            # Hide weak or noisy connections below threshold
            if c_score < threshold and b_score < threshold:
                continue

            target_row = self.movies_df.iloc[i]
            
            rel_types = []
            if c_score >= threshold:
                rel_types.append({
                    "type": "Genre Overlap",
                    "strength": round(c_score, 2),
                    "explanation": f"Shares common genre classifications ({target_row['genres']})."
                })
            if b_score >= threshold:
                rel_types.append({
                    "type": "Shared Audience Behavior",
                    "strength": round(b_score, 2),
                    "explanation": "Users who watched the selected movie frequently rated this similarly."
                })

            connections.append({
                "movieId": int(target_row['movieId']),
                "title": target_row['title'],
                "genres": target_row['genres'],
                "relationships": rel_types
            })

        connections.sort(key=lambda x: max([r['strength'] for r in x['relationships']], default=0), reverse=True)
        
        current_movie = self.movies_df.iloc[idx]
        return {
            "movie": {
                "movieId": int(current_movie['movieId']),
                "title": current_movie['title'],
                "genres": current_movie['genres']
            },
            "connections": connections
        }