import os

import streamlit as st
import spotipy

from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

from recommender import PlaylistRecommender

# CONFIGURATION

load_dotenv()

st.set_page_config(
    page_title="Spotify Recommender",
    page_icon="🎵",
    layout="wide",
)

# LOAD RECOMMENDER
@st.cache_resource
def get_recommender():

    return PlaylistRecommender()


recommender = get_recommender()

# SPOTIFY OAUTH
sp_oauth = SpotifyOAuth(
    client_id=os.getenv(
        "SPOTIPY_CLIENT_ID"
    ),

    client_secret=os.getenv(
        "SPOTIPY_CLIENT_SECRET"
    ),

    redirect_uri=os.getenv(
        "SPOTIPY_REDIRECT_URI",
        "http://127.0.0.1:8501/",
    ),

    scope=(
        "playlist-read-private "
        "playlist-read-collaborative "
        "user-library-read"
    ),

    cache_handler=spotipy.MemoryCacheHandler(),

    show_dialog=True,
)

# UI
st.title("🎵 AI Playlist Song Recommender")


# AUTHENTICATION
token_info = st.session_state.get(
    "token_info",
    None,
)


# Capture OAuth callback
query_params = st.query_params

if (
    "code" in query_params
    and not token_info
):

    code = query_params["code"]

    token_info = sp_oauth.get_access_token(
        code
    )

    st.session_state[
        "token_info"
    ] = token_info

    st.query_params.clear()

    st.rerun()


# Not logged in
if not token_info:
    auth_url = (
        sp_oauth.get_authorize_url()
    )

    st.info(
        "Log in with Spotify to load "
        "your personal playlists."
    )

    st.link_button(
        "🔑 Log in with Spotify",
        auth_url,
    )

    st.stop()

# REFRESH TOKEN

if sp_oauth.is_token_expired(
    token_info
):
    token_info = (
        sp_oauth.refresh_access_token(
            token_info["refresh_token"]
        )
    )
    st.session_state[
        "token_info"
    ] = token_info


# CREATE SPOTIFY CLIENT

sp = spotipy.Spotify(auth=token_info["access_token"])

# USER INFO

try:
    user = sp.current_user()

    st.sidebar.success(
        f"Logged in as "
        f"**{user['display_name']}**"
    )

except spotipy.exceptions.SpotifyException as e:
    st.error(
        f"Spotify authentication error: {e}"
    )

    st.session_state.clear()
    st.stop()
 
# LOG OUT

if st.sidebar.button("Log out"):
    st.session_state.clear()
    st.rerun()

# FETCH PLAYLISTS

try:
    playlists_data = []
    offset = 0
    while True:
        response = sp.current_user_playlists(
            limit=50,
            offset=offset,
        )

        items = response.get(
            "items",
            []
        )

        playlists_data.extend(
            items
        )

        if (
            len(items) < 50
            or not response.get("next")
        ):
            break

        offset += 50

except spotipy.exceptions.SpotifyException as e:
    st.error(f"Unable to retrieve playlists: {e}")
    st.stop()

# Playlist selection

playlist_map = {
    p["name"]: p["id"]
    for p in playlists_data
}


if not playlist_map:

    st.warning(
        "No Spotify playlists were found."
    )

    st.stop()


selected_playlist_name = st.selectbox(
    "Select a Playlist to Match (Must be your own):",
    list(playlist_map.keys()),
)


top_k = st.slider(
    "Number of songs to recommend:",
    min_value=5,
    max_value=25,
    value=10,
)


# GENERATE RECOMMENDATIONS

if st.button(
    "Generate Recommendations",
    type="primary",
):

    with st.spinner("Analyzing your playlist..."):

        playlist_id = playlist_map[
            selected_playlist_name
        ]

        # 1. FETCH ALL PLAYLIST TRACKS

        try:

            playlist_items = []

            offset = 0

            while True:

                response = sp.playlist_items(
                    playlist_id,
                    limit=100,
                    offset=offset,
                )

                items = response.get(
                    "items",
                    []
                )

                playlist_items.extend(
                    items
                )

                if (
                    len(items) < 100
                    or not response.get("next")
                ):
                    break

                offset += 100


        except spotipy.exceptions.SpotifyException as e:

            st.error(
                f"Spotify API Error ({e.http_status}): "
                f"Unable to access playlist tracks."
            )

            st.stop()

        # 2. EXTRACT TRACK IDS
        track_ids = []
        track_names = []

        for entry in playlist_items:

            # Spotify has used both "item" and "track"
            track_obj = (
                entry.get("item")
                or entry.get("track")
            )

            if not isinstance(
                track_obj,
                dict
            ):
                continue

            track_id = track_obj.get(
                "id"
            )

            if not track_id:
                continue

            # Ignore local files
            if track_obj.get(
                "is_local",
                False
            ):
                continue

            # Make sure it's actually a track
            if track_obj.get(
                "type"
            ) != "track":
                continue

            track_ids.append(
                track_id
            )

            track_names.append(
                track_obj.get(
                    "name",
                    "Unknown"
                )
            )


        # Remove duplicate track IDs
        track_ids = list(
            dict.fromkeys(track_ids)
        )


        if not track_ids:

            st.warning(
                "The selected playlist is empty "
                "or contains only local files."
            )

            st.stop()


        # 3. LOOK UP TRACKS IN LOCAL DATASET

        (
            playlist_features,
            matched_ids,
            missing_ids,
        ) = recommender.get_playlist_features(
            track_ids
        )


        # 4. REPORT MATCHING

        matched_count = len(
            matched_ids
        )

        missing_count = len(
            missing_ids
        )

        total_count = len(
            track_ids
        )


        st.write(
            f"Found **{matched_count} / "
            f"{total_count}** playlist songs "
            f"in the training dataset."
        )

        # Handle missing tracks

        if missing_count > 0:

            with st.expander(
                f"⚠️ {missing_count} songs "
                f"were not found in the dataset"
            ):

                st.write(
                    "These songs cannot contribute "
                    "to the playlist profile because "
                    "Spotify's Audio Features endpoint "
                    "is unavailable to this application."
                )

                st.write(
                    "Missing Spotify track IDs:"
                )

                for track_id in missing_ids:

                    st.code(track_id)


        # Need at least one matching song

        if matched_count == 0:

            st.error(
                "None of the playlist songs were "
                "found in the local training dataset."
            )

            st.stop()


        # 5. GENERATE RECOMMENDATIONS

        try:

            recs = (
                recommender
                .recommend_from_features(
                    playlist_audio_features=(
                        playlist_features
                    ),
                    existing_track_ids=track_ids,
                    top_k=top_k,
                )
            )

        except Exception as e:

            st.error(
                f"Error generating recommendations: "
                f"{e}"
            )

            st.exception(e)

            st.stop()


        # 6. DISPLAY RESULTS

        st.subheader(
            "🎯 Recommended Songs Based on "
            "Your Playlist's Acoustic Profile"
        )


        if recs.empty:

            st.warning(
                "No recommendations were found."
            )

            st.stop()


        for rank, (_, row) in enumerate(
            recs.iterrows(),
            start=1,
        ):

            col1, col2, col3 = (
                st.columns(
                    [3, 2, 1]
                )
            )


            with col1:

                st.markdown(
                    f"**{rank}. "
                    f"{row['track_name']}**"
                )

                st.caption(
                    f"Artist: "
                    f"{row['artists']}"
                )


            with col2:

                st.badge(
                    f"Genre: "
                    f"{row['track_genre']}"
                )


            with col3:

                st.metric(
                    "Similarity",
                    f"{row['similarity_score'] * 100:.1f}%"
                )


            st.divider()