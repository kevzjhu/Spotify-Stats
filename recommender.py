import joblib
import numpy as np
import pandas as pd


class PlaylistRecommender:

    def __init__(
        self,
        scaler_path="artifacts/scaler.joblib",
        model_path="artifacts/knn_model.joblib",
        data_path="artifacts/tracks_metadata.feather",
    ):
        self.scaler = joblib.load(scaler_path)
        self.knn = joblib.load(model_path)
        self.df_meta = pd.read_feather(data_path)
        self.feature_cols = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        ]

    def recommend_from_features(
        self, playlist_audio_features_list, existing_track_ids=None, top_k=10
    ):
        # 1. Filter out empty/None features from API responses
        valid_features = [
            f for f in playlist_audio_features_list if f is not None
        ]
        if not valid_features:
            return pd.DataFrame()

        # 2. Extract relevant numeric columns
        feat_df = pd.DataFrame(valid_features)[self.feature_cols]

        # 3. Compute playlist centroid
        centroid = feat_df.mean().values.reshape(1, -1)
        scaled_centroid = self.scaler.transform(centroid)

        # 4. Query nearest neighbors (fetch extra candidates to account for duplicates/existing songs)
        n_queries = (
            top_k + len(existing_track_ids) if existing_track_ids else top_k + 20
        )
        distances, indices = self.knn.kneighbors(
            scaled_centroid, n_neighbors=min(n_queries, len(self.df_meta))
        )

        # 5. Build candidate DataFrame
        candidates = self.df_meta.iloc[indices[0]].copy()
        candidates["distance"] = distances[0]
        candidates["similarity_score"] = 1 - distances[0]

        # 6. Exclude tracks already present in the user's playlist
        if existing_track_ids:
            candidates = candidates[
                ~candidates["track_id"].isin(existing_track_ids)
            ]

        return candidates.head(top_k)