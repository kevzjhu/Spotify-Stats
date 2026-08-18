import os
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from recommender import PlaylistRecommender

load_dotenv()

# Set Streamlit page config
st.set_page_config(page_title="Spotify Recommender", page_icon="🎵", layout="wide")

# Cache model loader so it's loaded only once into RAM
@st.cache_resource
def get_recommender():
    return PlaylistRecommender()

recommender = get_recommender()

# Spotify OAuth Setup
sp_oauth = SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8501/"),
    scope="playlist-read-private playlist-read-collaborative user-library-read",
    cache_handler=spotipy.MemoryCacheHandler(),
    show_dialog=True
)

st.title("🎵 AI Playlist Song Recommender")

# 1. Check Authentication
token_info = st.session_state.get("token_info", None)

# Capture code from redirect URL query parameters
query_params = st.query_params
if "code" in query_params and not token_info:
    code = query_params["code"]
    token_info = sp_oauth.get_access_token(code)
    st.session_state["token_info"] = token_info
    # Clear URL params and rerun
    st.query_params.clear()
    st.rerun()

if not token_info:
    auth_url = sp_oauth.get_authorize_url()
    st.info("Log in with Spotify to load your personal playlists.")
    st.link_button("🔑 Log in with Spotify", auth_url)
else:
    # Check if token needs refreshing
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
        st.session_state["token_info"] = token_info

    sp = spotipy.Spotify(auth=token_info["access_token"])
    user = sp.current_user()
    st.sidebar.success(f"Logged in as **{user['display_name']}**")
    
    if st.sidebar.button("Log out"):
        st.session_state.clear()
        st.rerun()

    # 2. Fetch User Playlists
    playlists_data = sp.current_user_playlists(limit=50)["items"]
    playlist_map = {p["name"]: p["id"] for p in playlists_data}

    selected_playlist_name = st.selectbox("Select a Playlist to Match:", list(playlist_map.keys()))
    top_k = st.slider("Number of songs to recommend:", min_value=5, max_value=25, value=10)

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Analyzing acoustic features..."):
            playlist_id = playlist_map[selected_playlist_name]
            
            # 1. Fetch playlist items
            try:
                playlist_response = sp.playlist_items(playlist_id, limit=100)
                items = playlist_response.get("items", []) if playlist_response else []
            except spotipy.exceptions.SpotifyException as e:
                st.error(f"Spotify API Error ({e.http_status}): Unable to access tracks for this playlist.")
                st.info("Ensure your account is added to 'User Management' in the Spotify Developer Dashboard.")
                items = []

            # 2. Extract track IDs supporting both 'item' (new) and 'track' (legacy) keys
            track_ids = []
            for entry in items:
                # Support both new payload format ('item') and legacy format ('track')
                track_obj = entry.get("item") or entry.get("track")
                
                if (
                    track_obj 
                    and isinstance(track_obj, dict)
                    and track_obj.get("id") 
                    and not track_obj.get("is_local", False)
                    and track_obj.get("type") == "track"
                ):
                    track_ids.append(track_obj["id"])

            print(f"Extracted {len(track_ids)} valid tracks.")

            if not track_ids:
                st.warning("Selected playlist is empty or contains local files only.")
            else:
                # Fetch audio features in batches of 100
                audio_features = []
                for i in range(0, len(track_ids), 100):
                    batch = sp.audio_features(track_ids[i:i+100])
                    audio_features.extend(batch)

                # Query Recommender
                recs = recommender.recommend_from_features(
                    playlist_audio_features_list=audio_features,
                    existing_track_ids=track_ids,
                    top_k=top_k
                )

                # Display Results
                st.subheader("🎯 Recommended Songs Based on Acoustic Profile")
                for rank, (_, row) in enumerate(recs.iterrows(), start=1):
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.markdown(f"**{rank}. {row['track_name']}**")
                        st.caption(f"Artist: {row['artists']}")
                    with col2:
                        st.badge(f"Genre: {row['track_genre']}")
                    with col3:
                        st.metric("Similarity", f"{row['similarity_score'] * 100:.1f}%")
                    st.divider()