# 🎵 Spotify AI Playlist Song Recommender

A Spotify playlist recommendation app that uses audio-feature similarity and a K-Nearest Neighbors (KNN) model to recommend songs based on the overall acoustic profile of a user's playlist.

The project combines the Spotify Web API with a machine-learning pipeline trained on a large Spotify tracks dataset. Given a user's playlist, the application finds matching tracks in the local dataset, calculates the playlist's average audio profile, and uses cosine-distance KNN to find similar songs.

## Features

- 🔐 Spotify OAuth authentication
- 📋 Load and select from the user's Spotify playlists
- 🎧 Retrieve playlist track IDs through the Spotify Web API
- 🔎 Match Spotify tracks against the local training dataset
- 📊 Analyze a playlist using audio characteristics
- 🤖 KNN-based song recommendations
- 🎯 Configurable number of recommendations
- 🚫 Exclude songs already present in the selected playlist
- 📈 Display recommendation similarity scores and genres
- ⚡ Cache the trained recommender in Streamlit

## How It Works

The recommendation pipeline is:

```text
User's Spotify Playlist
        │
        ▼
Spotify Web API
        │
        ▼
Spotify Track IDs
        │
        ▼
Match IDs against local Spotify dataset
        │
        ▼
Retrieve audio features
        │
        ▼
Apply training-time preprocessing
        │
        ▼
Calculate playlist audio-feature centroid
        │
        ▼
StandardScaler
        │
        ▼
KNN with cosine distance
        │
        ▼
Nearest songs in training dataset
        │
        ▼
Recommended Songs
```

### Playlist Representation

Each playlist is represented by the mean of its audio features. The model uses the following nine features:

- Danceability
- Energy
- Valence
- Tempo
- Loudness
- Acousticness
- Instrumentalness
- Speechiness
- Liveness

The original Spotify audio features are preprocessed before being passed to the model:

- `loudness` is clipped at the 1st percentile calculated from the training dataset
- `instrumentalness`
- `speechiness`
- `liveness`
- `acousticness`

are transformed using `log1p`.

The processed features are then standardized using `StandardScaler`.

## Machine Learning Model

The recommender uses `sklearn.neighbors.NearestNeighbors` with:

```python
metric="cosine"
algorithm="brute"
```

The model is trained on the processed audio-feature matrix from the Spotify tracks dataset.

The playlist's feature centroid is transformed using the same scaler used during training, then passed to the KNN model to find the nearest songs.

### Recommendation Similarity

The application converts cosine distance into a simple similarity score:

```text
similarity = 1 - cosine_distance
```

The application then removes songs that are already present in the selected playlist and returns the requested number of recommendations.

## Dataset

The training data is based on the **Spotify Tracks Dataset** available through Hugging Face:

`maharshipandya/spotify-tracks-dataset`

The dataset contains Spotify track IDs, track metadata, genres, popularity, and audio features.

During preprocessing, the project:

1. Removes rows with missing feature values
2. Removes duplicate `track_id` values
3. Removes tracks with invalid tempos
4. Clips extreme low loudness values
5. Applies logarithmic transformations to heavily right-skewed features
6. Standardizes the final model features

## Project Structure

```text
Spotify Stats/
│
├── app.py
│   └── Streamlit application and Spotify OAuth flow
│
├── recommender.py
│   └── Playlist preprocessing, dataset lookup, and KNN recommendations
│
├── tracks_dataset.py
│   └── Dataset cleaning, preprocessing, model training, and artifact generation
│
├── .env
│   └── Spotify API credentials
│
└── artifacts/
    ├── scaler.joblib
    ├── knn_model.joblib
    ├── tracks_metadata.feather
    └── preprocessing_config.joblib
```

## Requirements

Python 3.10+ is recommended.

Install the required packages:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib pyarrow spotipy streamlit python-dotenv
```

The project uses:

- Python
- Pandas
- NumPy
- scikit-learn
- SciPy
- Joblib
- PyArrow
- Spotipy
- Streamlit
- python-dotenv
- Matplotlib
- Seaborn

## Spotify API Setup

You need a Spotify Developer application to authenticate users.

Create an application through the Spotify Developer Dashboard and obtain:

- Client ID
- Client Secret

Set your application's redirect URI to:

```text
http://127.0.0.1:8501/
```

Create a `.env` file in the project root:

```env
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8501/
```

Do **not** commit `.env` or your Spotify credentials to GitHub.

A `.gitignore` should include:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd "Spotify Stats"
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install the packages listed in the Requirements section.

## Generate the ML Artifacts

The model artifacts must be generated before running the Streamlit application.

Run:

```bash
python tracks_dataset.py
```

This generates:

```text
artifacts/
├── scaler.joblib
├── knn_model.joblib
├── tracks_metadata.feather
└── preprocessing_config.joblib
```

The `preprocessing_config.joblib` file stores the training-time preprocessing parameters so that playlist data is processed consistently during inference.

## Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

Then:

1. Click **Log in with Spotify**
2. Authorize the application
3. Select a Spotify playlist
4. Choose the number of recommendations
5. Click **Generate Recommendations**
6. Review the recommended songs

## Why the App Uses a Local Dataset for Audio Features

The application does **not** call Spotify's Audio Features endpoint when generating recommendations.

Instead, it retrieves playlist track IDs through Spotify and uses those IDs to look up the corresponding tracks and audio features in the local training dataset.

This allows the existing ML model to continue using the same feature space it was trained on without depending on the restricted Audio Features API endpoint.

Tracks that cannot be found in the local dataset are reported by the application and do not contribute to the playlist's audio profile.

## Model Evaluation

The training process includes evaluation using:

- Precision@K
- Mean Reciprocal Rank (MRR)
- Genre consistency

The evaluation compares the genres of nearest-neighbor tracks against the genre of sampled query tracks.

The project also examines **hubness**, measuring how frequently individual tracks appear among nearest-neighbor results.

These analyses were used to understand the behavior of the nearest-neighbor recommendation system.

## Limitations

### Dataset Coverage

Recommendations depend on the selected playlist's tracks being present in the local training dataset.

If a Spotify track is not present in the dataset, its audio features cannot currently be used to calculate the playlist profile.

### Audio Similarity vs. User Preference

The model recommends songs based on similarity in audio characteristics. It does not directly model:

- User listening history
- Likes/dislikes
- Skip behavior
- Personalized artist preferences
- Lyrics
- Collaborative filtering
- Context such as workout, study, commute, or mood

Therefore, a recommendation that is acoustically similar is not necessarily a song the user will personally prefer.

### Dataset Age

The local dataset is a fixed training dataset. Newer Spotify releases may not be represented.

### Spotify API Restrictions

Spotify API availability and application requirements can change over time. The application therefore avoids depending on the Audio Features endpoint for inference and uses the local dataset for audio-feature lookup.

## Future Improvements

Potential improvements include:

- Improve track matching when Spotify tracks are absent from the dataset
- Incorporate user listening history
- Add collaborative filtering
- Combine audio similarity with popularity and genre
- Add artist diversity constraints
- Add recommendation explanations
- Experiment with weighted playlist centroids
- Compare KNN against other recommendation algorithms
- Add PCA or dimensionality-reduction visualizations
- Improve evaluation beyond genre consistency
- Deploy the Streamlit application

## Technologies

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical processing |
| scikit-learn | Scaling and KNN model |
| Joblib | Model artifact serialization |
| PyArrow | Feather dataset storage |
| Spotipy | Spotify Web API client |
| Streamlit | Web application |
| Matplotlib | Visualization |
| Seaborn | Exploratory data analysis |
| Hugging Face | Source for the Spotify tracks dataset |

## License

This project is intended for educational and portfolio purposes.

Check the applicable terms and licenses for the Spotify Web API, Spotify data, and the underlying dataset before publicly deploying or redistributing the application.
