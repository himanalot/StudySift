import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheHandler
import pandas as pd
import requests
import json
import random
import concurrent.futures
import time
import streamlit as st
import threading
import os
from dotenv import load_dotenv

# Debug: List available keys in st.secrets


# --------------------------- Configuration --------------------------- #

SPOTIPY_REDIRECT_URI = 'https://studysift-jbyhh4glfowhcs8xszu9xr.streamlit.app'  # Or your deployed app URL

GPT4_MINI_API_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
# Now you can access your variables
# Access variables from the 'spotify' section
client_id = st.secrets["spotify"]["SPOTIPY_CLIENT_ID"]
client_secret = st.secrets["spotify"]["SPOTIPY_CLIENT_SECRET"]

# Access variables from the 'openai' section
api_key = st.secrets["openai"]["GPT4_MINI_API_KEY"]
# Ensure the variables are loaded
if not all([client_id, client_secret, api_key]):
    raise ValueError("Missing environment variables. Please set them in Render.")

# --------------------------- Custom Cache Handler --------------------------- #

class StreamlitSessionCacheHandler(CacheHandler):
    def __init__(self, token_info_key):
        self.token_info_key = token_info_key

    def get_cached_token(self):
        return st.session_state.get(self.token_info_key)

    def save_token_to_cache(self, token_info):
        st.session_state[self.token_info_key] = token_info

# --------------------------- Authentication --------------------------- #

def authenticate_spotify():
    token_info_key = 'token_info'
    global client_id, client_secret

    # Initialize session state for 'token_info' if not already set
    if token_info_key not in st.session_state:
        st.session_state[token_info_key] = None

    # Create SpotifyOAuth object with correct client_id and client_secret
    sp_oauth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope='playlist-modify-public playlist-modify-private',
        cache_handler=StreamlitSessionCacheHandler(token_info_key),
        show_dialog=True
    )

    # Step 1: Check if token_info is already available in session_state
    token_info = st.session_state.get(token_info_key)

    if token_info is None:
        auth_url = sp_oauth.get_authorize_url()
        st.write(f'Please [authorize]({auth_url}) to access your Spotify account.')

        # Step 2: Retrieve the authorization code from query parameters
        auth_code = st.query_params.get('code')

        if auth_code:
            # Explicitly set as_dict=True to handle the response as a dictionary
            token_info = sp_oauth.get_access_token(auth_code, as_dict=True)

            if token_info:
                st.session_state[token_info_key] = token_info

                # Remove the 'code' parameter from the URL to prevent reuse
                # Create a new dictionary excluding 'code'
                new_query_params = st.query_params.to_dict()
                new_query_params.pop('code', None)
                st.query_params.from_dict(new_query_params)

                st.rerun()  # Trigger a rerun to use the new token_info
            else:
                st.error("Failed to obtain access token. Please try authorizing again.")
                st.stop()

    # Step 3: Refresh token if it's expired
    token_info = st.session_state.get(token_info_key)
    if token_info:
        expires_at = token_info.get('expires_at')
        refresh_token = token_info.get('refresh_token')

        if expires_at and refresh_token:
            current_time = int(time.time())
            if expires_at - current_time < 60:
                try:
                    token_info = sp_oauth.refresh_access_token(refresh_token, as_dict=True)
                    st.session_state[token_info_key] = token_info
                except Exception as e:
                    st.error(f"Failed to refresh access token: {e}")
                    st.session_state[token_info_key] = None
                    st.stop()
        else:
            st.error("Invalid token information. Please re-authenticate.")
            st.session_state[token_info_key] = None
            st.stop()
    else:
        st.error("Provide the email that you use for your Spotify account to Ishan Ramrakhiani (ishanramrakhiani@gmail.com) for access.")
        st.stop()

    # Step 4: Return authenticated Spotify client
    access_token = st.session_state[token_info_key].get('access_token') if st.session_state[token_info_key] else None
    if access_token:
        return spotipy.Spotify(auth=access_token)
    else:
        st.error("Access token is missing. Please try again.")
        st.stop()

# --------------------------- Helper Functions --------------------------- #

def call_gpt4_mini(prompt):
    """
    Calls the GPT-4o Mini API with the provided prompt and returns the response.
    """
    global api_key
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    try:
        response = requests.post(GPT4_MINI_API_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if 'choices' not in result or not result['choices']:
            st.write("No choices in GPT-4o Mini response.")
            st.write("Response:", result)
            return None
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.HTTPError as http_err:
        st.write(f"HTTP error occurred: {http_err}")
        if 'response' in locals():
            st.write(f"Response Text: {response.text}")
    except json.JSONDecodeError as json_err:
        st.write(f"JSON decode error: {json_err}")
        if 'response' in locals():
            st.write(f"Response Text: {response.text}")
    except Exception as err:
        st.write(f"Other error occurred: {err}")
        if 'response' in locals():
            st.write(f"Response Text: {response.text}")
    return None

def get_feature_definitions():
    """
    Returns a string containing the definitions of Spotify audio features.
    """
    feature_definitions = """
- **acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic.
- **danceability**: Describes how suitable a track is for dancing based on tempo, rhythm stability, beat strength, and overall regularity.
- **energy**: Measure from 0.0 to 1.0 representing intensity and activity.
- **instrumentalness**: Predicts whether a track contains no vocals.
- **liveness**: Detects the presence of an audience in the recording.
- **loudness**: Overall loudness of a track in decibels (dB).
- **speechiness**: Detects the presence of spoken words.
- **valence**: Measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.
- **tempo**: The overall estimated tempo of a track in beats per minute (BPM).
"""
    return feature_definitions

def construct_prompt(batch_df, diagnostic, feature_definitions):
    """
    Constructs a prompt for GPT-4o Mini with the song data, the diagnostic, and feature definitions.
    """
    prompt = f"Based on the following user preferences:\n\n'{diagnostic}'\n\n"
    prompt += "Here are the definitions of the audio features:\n\n"
    prompt += feature_definitions + "\n\n"
    prompt += (
        "Please analyze the following songs and decide whether each song should be included in the playlist. "
        "For each song, provide a decision 'Keep' or 'Discard'. Here are the songs:\n\n"
    )

    songs_list = []
    for index, row in batch_df.iterrows():
        song_info = {
            "id": row["id"],
            "name": row["name"],
            "artists": row["artists"],
            "features": {
                "danceability": row["danceability"],
                "energy": row["energy"],
                "valence": row["valence"],
                "tempo": row["tempo"],
                "acousticness": row["acousticness"],
                "instrumentalness": row["instrumentalness"],
                "liveness": row["liveness"],
                "loudness": row["loudness"],
                "speechiness": row["speechiness"],
            },
        }
        songs_list.append(song_info)

    songs_json = json.dumps(songs_list, indent=2)
    prompt += songs_json + "\n\n"

    prompt += (
        "Respond ONLY with a JSON array of decisions in the following format, "
        "ensuring all property names and string values are enclosed in double quotes, "
        "and without any code fences, explanations, or additional text:\n\n"
        '[\n  {"id": "song_id", "decision": "Keep" or "Discard"},\n  ...\n]\n'
    )

    return prompt

def parse_model_response(response):
    """
    Parses the GPT-4o Mini response to extract song IDs to keep.
    """
    try:
        response = response.strip()
        if response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
            if response.startswith('json'):
                response = response[4:].strip()
        decisions = json.loads(response)
        ids_to_keep = [item['id'] for item in decisions if item['decision'].lower() == 'keep']
        return ids_to_keep
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        st.write(f"Failed to parse GPT-4o Mini response: {e}")
        st.write("Raw GPT-4o Mini response:", response)
        return []

def get_track_info_concurrent(track_ids, sp):
    """
    Retrieves track information (name, artists) for a list of track IDs using threading for concurrency.
    """
    track_info_list = []
    lock = threading.Lock()

    def fetch_batch(start):
        batch = track_ids[start:start+50]
        try:
            tracks = sp.tracks(batch)['tracks']
            with lock:
                for track in tracks:
                    if track and track['id']:
                        track_artists = [artist['name'] for artist in track['artists']]
                        track_info = {
                            'id': track['id'],
                            'name': track['name'],
                            'artists': ', '.join(track_artists)
                        }
                        track_info_list.append(track_info)
        except Exception as e:
            st.write(f"Error fetching track info for batch starting at index {start}: {e}")

    threads = []
    for i in range(0, len(track_ids), 50):
        thread = threading.Thread(target=fetch_batch, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return track_info_list

def get_audio_features_concurrent(track_ids, sp):
    """
    Retrieves audio features for a list of track IDs using threading for concurrency.
    """
    audio_features = []
    lock = threading.Lock()

    def fetch_batch(start):
        batch = track_ids[start:start + 100]
        try:
            features = sp.audio_features(batch)
            with lock:
                audio_features.extend([f for f in features if f is not None])
        except Exception as e:
            st.write(f"Error fetching audio features for batch starting at index {start}: {e}")

    threads = []
    for i in range(0, len(track_ids), 100):
        thread = threading.Thread(target=fetch_batch, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return audio_features

def create_new_playlist(name, track_ids, sp):
    """
    Creates a new Spotify playlist with the specified track IDs for the authenticated user.
    """
    try:
        user_id = sp.me()['id']  # Use sp.me() instead of sp.current_user()
        playlist = sp.user_playlist_create(
            user=user_id,
            name=name,
            public=True,
            description='Generated by StudySift'
        )
        # Add tracks to the playlist
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            sp.playlist_add_items(playlist_id=playlist['id'], items=batch)
        return playlist['external_urls']['spotify']
    except Exception as e:
        st.write(f"Error creating playlist: {e}")
        return None

def filter_songs_with_model(df, diagnostic, feature_definitions):
    """
    Filters songs using GPT-4o Mini based on the song features and the user's diagnostic.
    """
    filtered_ids = []
    batch_size = 7
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        prompt = construct_prompt(batch_df, diagnostic, feature_definitions)
        response = call_gpt4_mini(prompt)
        if response is None:
            continue
        ids_to_keep = parse_model_response(response)
        filtered_ids.extend(ids_to_keep)
    return filtered_ids

def determine_search_parameters(genre, mood, energy_level, additional_info):
    """
    Uses GPT-4o Mini to determine which genres and artists to search for based on user inputs.
    """
    prompt = f"""
Based on the following user preferences:
- Genre: {genre}
- Mood: {mood}
- Energy Level: {energy_level}
- Additional Info: {additional_info if additional_info else 'None'}

Please provide a list of genres and artists that match these preferences.

**Respond ONLY with valid JSON in the following format, without any code fences or additional text:**

{{
    "genres": ["genre1", "genre2", ...],
    "artists": ["artist1", "artist2", ...]
}}
"""
    response = call_gpt4_mini(prompt)
    if response:
        try:
            response = response.strip()
            if response.startswith("```") and response.endswith("```"):
                response = response[3:-3].strip()
                if response.startswith('json'):
                    response = response[4:].strip()
            search_params = json.loads(response)
            genres = search_params.get('genres', [])
            artists = search_params.get('artists', [])
            return genres, artists
        except (json.JSONDecodeError, TypeError) as e:
            st.write(f"Error parsing GPT-4o Mini response for search parameters: {e}")
            st.write("Raw GPT-4o Mini response:", response)
            return [genre], []
    else:
        return [genre], []

def filter_playlists_with_model(playlists_info, diagnostic):
    """
    Filters playlists using GPT-4o Mini based on the playlist names and the user's diagnostic.
    Returns a list of playlists to include.
    """
    playlist_names = [info[1] for info in playlists_info]
    prompt = f"""
Based on the following user preferences:
'{diagnostic}'

Here is a list of playlist names:
{json.dumps(playlist_names, indent=2)}

Please analyze the playlist names and decide whether each playlist is relevant to the user's preferences.
Respond ONLY with a JSON array of decisions in the following format, ensuring all property names and string values are enclosed in double quotes, and without any code fences, explanations, or additional text:

[
  {{"name": "playlist_name", "decision": "Include" or "Exclude"}},
  ...
]
"""
    response = call_gpt4_mini(prompt)
    if response:
        try:
            response = response.strip()
            if response.startswith("```") and response.endswith("```"):
                response = response[3:-3].strip()
                if response.startswith('json'):
                    response = response[4:].strip()
            decisions = json.loads(response)
            included_playlists = []
            for decision in decisions:
                name = decision.get('name')
                decision_value = decision.get('decision')
                if decision_value.lower() == 'include':
                    for info in playlists_info:
                        if info[1] == name:
                            included_playlists.append(info)
                            break
            return included_playlists
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            st.write(f"Failed to parse GPT-4o Mini response: {e}")
            st.write("Raw GPT-4o Mini response:", response)
            return playlists_info
    else:
        return playlists_info

def search_and_combine_playlists(
    search_genres,
    search_artists,
    limit=200,
    sample_size_per_playlist=10,
    diagnostic="",
    sp=None
):
    """
    Searches for playlists based on genres and artists, combines them.
    Returns a list of unique track IDs limited to a specified count.
    """
    track_ids = set()
    processed_playlists = set()
    lock = threading.Lock()

    def process_playlist(playlist_id, playlist_name, playlist_size):
        if playlist_size < 100 or playlist_size > 1500:
            return
        fetched_track_ids = fetch_playlist_tracks(playlist_id, sample_size=sample_size_per_playlist, sp=sp)
        with lock:
            track_ids.update(fetched_track_ids)

    def search_playlists(query):
        playlists_info = []
        search_results = sp.search(q=query, type='playlist', limit=10)
        playlist_items = search_results['playlists']['items']
        for playlist in playlist_items:
            playlist_id = playlist['id']
            with lock:
                if playlist_id in processed_playlists:
                    continue
                processed_playlists.add(playlist_id)
            playlist_name = playlist['name']
            playlist_size = playlist['tracks']['total']
            playlists_info.append((playlist_id, playlist_name, playlist_size))
        return playlists_info

    playlists_info = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(search_playlists, f'genre:"{genre}"'): genre for genre in search_genres}
        for future in concurrent.futures.as_completed(futures):
            genre = futures[future]
            try:
                result = future.result()
                playlists_info.extend(result)
            except Exception as e:
                st.write(f"Error searching playlists for genre '{genre}': {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(search_playlists, f'artist:"{artist}"'): artist for artist in search_artists}
        for future in concurrent.futures.as_completed(futures):
            artist = futures[future]
            try:
                result = future.result()
                playlists_info.extend(result)
            except Exception as e:
                st.write(f"Error searching playlists for artist '{artist}': {e}")

    playlists_info = filter_playlists_with_model(playlists_info, diagnostic)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_playlist, pid, pname, psize) for pid, pname, psize in playlists_info]
        concurrent.futures.wait(futures)

    track_ids = list(track_ids)
    if len(track_ids) > limit:
        track_ids = random.sample(track_ids, limit)
    return track_ids

def fetch_playlist_tracks(playlist_id, sample_size=10, sp=None):
    """
    Efficiently fetches all track IDs from a given playlist and selects a random sample.
    """
    track_ids = []
    try:
        total_tracks = sp.playlist_items(playlist_id, fields='total')['total']
        batch_size = 100
        offsets = range(0, total_tracks, batch_size)

        def fetch_batch(offset):
            results = sp.playlist_items(
                playlist_id,
                fields='items.track.id',
                additional_types=['track'],
                limit=batch_size,
                offset=offset
            )
            items = results['items']
            batch_track_ids = []
            for item in items:
                track = item['track']
                if track and track['id']:
                    batch_track_ids.append(track['id'])
            return batch_track_ids

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_offset = {executor.submit(fetch_batch, offset): offset for offset in offsets}
            for future in concurrent.futures.as_completed(future_to_offset):
                try:
                    batch_track_ids = future.result()
                    track_ids.extend(batch_track_ids)
                except Exception as e:
                    offset = future_to_offset[future]
                    st.write(f"Error fetching tracks at offset {offset} from playlist {playlist_id}: {e}")

        if len(track_ids) >= sample_size:
            track_ids = random.sample(track_ids, sample_size)
        else:
            pass
    except spotipy.exceptions.SpotifyException as e:
        st.write(f"Error fetching playlist {playlist_id}: {e}")
    except Exception as e:
        st.write(f"An unexpected error occurred while fetching playlist {playlist_id}: {e}")
    return track_ids

# --------------------------- Main Functionality --------------------------- #

def main():
    st.title("Echo (AI-Powered Playlists)")

    # Authenticate the user
    sp = authenticate_spotify()

    # Collect user inputs
    genre = st.text_input("Preferred Genre (e.g., Rock, Pop):")
    mood = st.text_input("Desired Mood (e.g., Calm, Happy, Sad):")
    energy_level = st.selectbox("Energy Level:", ["Low", "Medium", "High"])
    additional_info = st.text_input("Additional Info or Preferences (optional):")

    if st.button("Submit"):
        if not genre or not mood or not energy_level:
            st.error("Please fill in the required fields: Genre, Mood, and Energy Level.")
            return

        # Determine search parameters using GPT-4o Mini
        ai_genres, ai_artists = determine_search_parameters(genre, mood, energy_level, additional_info)
        if not ai_genres and not ai_artists:
            st.error("No genres or artists suggested by GPT-4o Mini for searching.")
            return

        selected_genres = ai_genres
        selected_artists = ai_artists

        if not selected_genres and not selected_artists:
            st.error("No genres or artists available for searching.")
            return

        # Start processing
        process_playlist_generation(
            genre, mood, energy_level, additional_info,
            selected_genres, selected_artists, sp
        )

def process_playlist_generation(
    genre,
    mood,
    energy_level,
    additional_info,
    selected_genres,
    selected_artists,
    sp
):
    """
    Handles the entire process of generating the playlist.
    """
    start_time = time.time()
    status_placeholder = st.empty()
    try:
        diagnostic = f"A playlist with {energy_level.lower()} energy, {mood.lower()} mood, in the {genre} genre."
        if additional_info:
            diagnostic += f" Additional info: {additional_info}"

        with st.spinner('Searching and combining playlists...'):
            combined_track_ids = search_and_combine_playlists(
                selected_genres,
                selected_artists,
                limit=200,
                sample_size_per_playlist=10,
                diagnostic=diagnostic,
                sp=sp
            )

        if not combined_track_ids:
            st.info("No tracks found based on the specified genres and artists.")
            return

        with st.spinner('Retrieving audio features for collected tracks...'):
            audio_features = get_audio_features_concurrent(combined_track_ids, sp)
        status_placeholder.text(f"Retrieved audio features for {len(audio_features)} tracks.")

        with st.spinner('Retrieving track information...'):
            track_info_list = get_track_info_concurrent(combined_track_ids, sp)
        status_placeholder.text(f"Retrieved track information for {len(track_info_list)} tracks.")

        combined_track_ids = [track['id'] for track in track_info_list]

        if not combined_track_ids:
            st.info("No tracks available after processing.")
            return

        with st.spinner('Merging track information and audio features...'):
            df_features = pd.DataFrame(audio_features)
            df_track_info = pd.DataFrame(track_info_list)
            df = pd.merge(df_track_info, df_features, on='id')
        status_placeholder.text(f"Merged data contains {len(df)} tracks.")

        features_to_handle = [
            'acousticness',
            'danceability',
            'energy',
            'instrumentalness',
            'liveness',
            'speechiness',
            'valence',
            'tempo',
            'loudness'
        ]
        for feature in features_to_handle:
            if feature in df.columns:
                df[feature] = df[feature].astype(float)
                df[feature] = df[feature].fillna(float('nan'))

        with st.spinner('Applying filtering with GPT-4o Mini...'):
            feature_definitions = get_feature_definitions()
            filtered_ids = filter_songs_with_model(df, diagnostic, feature_definitions)
        status_placeholder.text(f"Tracks after filtering: {len(filtered_ids)}")

        if not filtered_ids:
            st.info("No tracks match the specified criteria after filtering.")
            return

        with st.spinner('Creating new playlist...'):
            playlist_name = f"{genre} - {mood} - {energy_level} Energy"
            new_playlist_url = create_new_playlist(playlist_name, filtered_ids, sp)

        if new_playlist_url:
            total_time = time.time() - start_time
            status_placeholder.success(f"New Playlist Created: [Open Playlist]({new_playlist_url})\nTotal time taken: {total_time:.2f} seconds")
        else:
            st.error("Failed to create the new playlist.")

    except Exception as e:
        total_time = time.time() - start_time
        st.write(f"An error occurred during playlist generation: {e}")
        st.error(f"An error occurred: {e}\nTotal time before error: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()