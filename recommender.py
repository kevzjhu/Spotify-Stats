import joblib
import numpy as np
import pandas as pd


class PlaylistRecommender:

    def __init__(
        self,
        scaler_path="artifacts/scaler.joblib",
        model_path="artifacts/knn_model.joblib",
        data_path="artifacts/tracks_metadata.feather",
        preprocessing_path="artifacts/preprocessing_config.joblib",
    ):

        # ----------------------------------------------------
        # Load trained artifacts
        # ----------------------------------------------------

        self.scaler = joblib.load(scaler_path)

        self.knn = joblib.load(model_path)

        self.df_meta = pd.read_feather(data_path)

        self.preprocessing = joblib.load(
            preprocessing_path
        )

        self.model_features = self.preprocessing[
            "model_features"
        ]

        self.lower_loudness = self.preprocessing[
            "lower_loudness"
        ]

        self.log_features = self.preprocessing[
            "log_features"
        ]

        # ----------------------------------------------------
        # Build fast track ID lookup
        # ----------------------------------------------------

        self.track_lookup = (
            self.df_meta
            .drop_duplicates("track_id")
            .set_index("track_id")
        )

        print(
            f"Loaded {len(self.df_meta):,} tracks "
            f"from local dataset."
        )


    # ========================================================
    # PREPROCESS AUDIO FEATURES
    # ========================================================

    def preprocess_features(self, df):
        """
        Apply EXACTLY the same preprocessing used during
        model training.
        """

        df = df.copy()

        # Clip loudness using the training dataset's cutoff
        df["loudness_processed"] = df[
            "loudness"
        ].clip(
            lower=self.lower_loudness
        )

        # Apply log1p transformations
        for feature in self.log_features:

            df[f"{feature}_processed"] = np.log1p(
                df[feature]
            )

        # Return the exact model feature order
        return df[self.model_features]


    # ========================================================
    # LOOK UP SPOTIFY TRACKS IN LOCAL DATASET
    # ========================================================

    def get_playlist_features(self, track_ids):
        """
        Look up Spotify track IDs in the local 100k-song
        dataset.

        Returns:
            features_df
            matched_track_ids
            missing_track_ids
        """

        # Remove duplicates while preserving order
        track_ids = list(dict.fromkeys(track_ids))

        # Find IDs present in our dataset
        matched_ids = [
            track_id
            for track_id in track_ids
            if track_id in self.track_lookup.index
        ]

        # IDs that aren't in our dataset
        missing_ids = [
            track_id
            for track_id in track_ids
            if track_id not in self.track_lookup.index
        ]

        if not matched_ids:
            return (
                pd.DataFrame(),
                [],
                missing_ids
            )

        # Retrieve matching rows
        features_df = self.track_lookup.loc[
            matched_ids
        ].copy()

        # Ensure DataFrame even if only one track
        if isinstance(features_df, pd.Series):
            features_df = features_df.to_frame().T

        # Reorder to match playlist order
        features_df = features_df.loc[
            matched_ids
        ]

        return (
            features_df,
            matched_ids,
            missing_ids
        )


    # ========================================================
    # RECOMMEND SONGS
    # ========================================================

    def recommend_from_features(
        self,
        playlist_audio_features,
        existing_track_ids=None,
        top_k=10,
    ):

        if existing_track_ids is None:
            existing_track_ids = []


        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if playlist_audio_features is None:
            return pd.DataFrame()

        if len(playlist_audio_features) == 0:
            return pd.DataFrame()


        # ----------------------------------------------------
        # Convert to DataFrame
        # ----------------------------------------------------

        feat_df = pd.DataFrame(
            playlist_audio_features
        )


        # ----------------------------------------------------
        # Make sure all required raw features exist
        # ----------------------------------------------------

        raw_features = [
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

        missing_columns = [
            col
            for col in raw_features
            if col not in feat_df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required audio features: "
                + ", ".join(missing_columns)
            )


        # ----------------------------------------------------
        # Apply training preprocessing
        # ----------------------------------------------------

        processed_features = self.preprocess_features(
            feat_df
        )


        # ----------------------------------------------------
        # Calculate playlist centroid
        # ----------------------------------------------------

        # Average the RAW/processed feature values before
        # scaling, matching the training feature space.

        centroid = (
            processed_features
            .mean()
            .values
            .reshape(1, -1)
        )


        # ----------------------------------------------------
        # Scale playlist centroid
        # ----------------------------------------------------

        scaled_centroid = self.scaler.transform(
            centroid
        )


        # ----------------------------------------------------
        # Query KNN
        # ----------------------------------------------------

        n_queries = min(
            top_k + len(existing_track_ids) + 20,
            len(self.df_meta)
        )

        distances, indices = self.knn.kneighbors(
            scaled_centroid,
            n_neighbors=n_queries
        )


        # ----------------------------------------------------
        # Build candidates
        # ----------------------------------------------------

        candidates = self.df_meta.iloc[
            indices[0]
        ].copy()

        candidates["distance"] = distances[0]

        # Cosine distance ranges from 0 to 1 for this use case.
        candidates["similarity_score"] = (
            1 - distances[0]
        )


        # ----------------------------------------------------
        # Remove songs already in playlist
        # ----------------------------------------------------

        if existing_track_ids:

            candidates = candidates[
                ~candidates["track_id"].isin(
                    existing_track_ids
                )
            ]


        # ----------------------------------------------------
        # Return top K
        # ----------------------------------------------------

        return candidates.head(top_k)