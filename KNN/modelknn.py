import pickle
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def read_and_join_data():
    artists = pd.read_json("../dataset/artists.jsonl", lines=True)
    sessions = pd.read_json("../dataset/sessions.jsonl", lines=True)
    tracks = pd.read_json("../dataset/tracks.jsonl", lines=True)
    users = pd.read_json("../dataset/users.jsonl", lines=True)

    sessions = sessions[sessions["event_type"] != "advertisment"]
    sessions_tracks = pd.merge(sessions, tracks, left_on="track_id", right_on="id", how="left")
    sessions_tracks.drop("id", axis=1, inplace=True)
    artists.columns = ["id", "artist_name", "artist_genres"]
    artists_sessions_tracks = pd.merge(sessions_tracks, artists, left_on="id_artist", right_on="id", how="left")
    artists_sessions_tracks.drop("id", axis=1, inplace=True)
    users.columns = ["id", "user_name", "city", "street", "user_fav_genres", "premium_user"]
    complete_table = pd.merge(artists_sessions_tracks, users, left_on="user_id", right_on="id", how="left")
    complete_table.drop(["id"], axis=1, inplace=True)
    return complete_table, users

def process_joined_data(complete_table, users):
    complete_table['release_date'] = pd.to_datetime(complete_table['release_date'], errors='coerce')  # Konwersja na datetime
    complete_table['release_date'] = complete_table['release_date'].astype('int64') // 10**9  # Na timestamp (sekundy)

    # Obliczanie czasu słuchania w ciągach dla tego samego track_id
    # Grupowanie danych według session_id
    grouped_by_session = complete_table.groupby("session_id")
    # Obliczanie czasu słuchania w ramach sesji
    session_listening_times = {}
    for session_id, group in grouped_by_session:
        last_end_time = None
        current_track = None
        session_times = {}
        for idx, row in group.iterrows():
            track_id = row["track_id"]
            timestamp = pd.to_datetime(row["timestamp"])
            # Jeśli to pierwsze wystąpienie track_id w tej sesji
            if last_end_time is not None and current_track != track_id:
                # Obliczamy różnicę w czasie między ostatnim a bieżącym
                listening_time = (timestamp - last_end_time).total_seconds()
                session_times[current_track] = listening_time
                last_end_time = timestamp
            elif last_end_time == None:
                last_end_time = timestamp  # Ustawiamy nowy "ostatni czas"
            current_track = track_id
        session_listening_times[session_id] = session_times

    # Przekształcenie wyników w DataFrame
    listening_times_df = []
    for session_id, tracks in session_listening_times.items():
        for track_id, listening_time in tracks.items():
            listening_times_df.append(
                {"session_id": session_id, "track_id": track_id, "listening_time": listening_time})

    listening_times_df = pd.DataFrame(listening_times_df)
    # Połączenie tabel na podstawie session_id i track_id
    complete_table = pd.merge(complete_table, listening_times_df, on=["session_id", "track_id"], how="left")

    #konwersja list na krotki
    complete_table['artist_genres'] = complete_table['artist_genres'].apply(
        lambda x: tuple(x) if isinstance(x, list) else x
    )

    sc = StandardScaler()
    complete_table["popularity"] = sc.fit_transform(np.log(complete_table["popularity"]).values.reshape(-1, 1))
    complete_table["duration_ms"] = sc.fit_transform(np.log(complete_table["duration_ms"]).values.reshape(-1, 1))
    # complete_table["release_date"] = sc.fit_transform(complete_table["release_date"].values.reshape(-1, 1))
    complete_table["danceability"] = sc.fit_transform(np.exp(complete_table["danceability"]).values.reshape(-1, 1))
    complete_table["energy"] = sc.fit_transform(np.exp(complete_table["energy"]).values.reshape(-1, 1))
    complete_table["loudness"] = sc.fit_transform(complete_table["loudness"].values.reshape(-1, 1))
    complete_table["speechiness"] = sc.fit_transform((complete_table["speechiness"] ** (1 / 5)).values.reshape(-1, 1))
    complete_table["acousticness"] = sc.fit_transform((complete_table["acousticness"] ** (1 / 3)).values.reshape(-1, 1))
    complete_table["instrumentalness"] = sc.fit_transform(complete_table["instrumentalness"].values.reshape(-1, 1))
    complete_table["liveness"] = sc.fit_transform(np.log(complete_table["liveness"]).values.reshape(-1, 1))
    complete_table["valence"] = sc.fit_transform(complete_table["valence"].values.reshape(-1, 1))
    complete_table["tempo"] = sc.fit_transform(complete_table["tempo"].values.reshape(-1, 1))
    complete_table["premium_user"] = complete_table['premium_user'].astype(int)
    complete_table["listening_time"] = sc.fit_transform(np.log(complete_table["listening_time"]).values.reshape(-1, 1))

    # Obliczenie łącznego czasu słuchania dla każdego użytkownika i gatunku
    genre_listening_time = complete_table.groupby(['user_id', 'artist_genres'])['listening_time'].sum().reset_index()
    # Przemianowanie kolumny na bardziej czytelną
    genre_listening_time.rename(columns={'listening_time': 'total_genre_listening_time'}, inplace=True)
    pivoted_table = genre_listening_time.pivot_table(
        index='user_id',
        columns='artist_genres',
        values='total_genre_listening_time',
        aggfunc='sum',  # Suma czasów słuchania w przypadku duplikatów
        fill_value=0  # Uzupełnianie brakujących wartości zerami
    )

    def calculate_kurtosis(x):
        return kurtosis(x, fisher=True, nan_policy='omit')  # Kurtoza z "omit" ignoruje NaN

    user_features = complete_table.groupby('user_id').agg({
        'danceability': ['median', 'std', calculate_kurtosis],
        'release_date': ['median', 'std', calculate_kurtosis],  # Średnia timestampów dat
        'energy': ['median', 'std', calculate_kurtosis],
        'tempo': ['median', 'std', calculate_kurtosis],
        'popularity': ['median', 'std', calculate_kurtosis],
        'duration_ms': ['median', 'std', calculate_kurtosis],
        'loudness': ['median', 'std', calculate_kurtosis],
        'speechiness': ['median', 'std', calculate_kurtosis],
        'acousticness': ['median', 'std', calculate_kurtosis],
        'instrumentalness': ['median', 'std', calculate_kurtosis],
        'liveness': ['median', 'std', calculate_kurtosis],
        'valence': ['median', 'std', calculate_kurtosis],
        'event_type': lambda x: (x == 'like').mean(),  # Odsetek polubień
        'explicit': lambda x: (x == 1).mean(),  # Odsetek utworów z przekleństwami
        'track_id': 'count',  # Liczba odsłuchanych utworów
        'artist_name': lambda x: x.value_counts().idxmax(),  # Najczęściej słuchany artysta
        'artist_genres': lambda x: ', '.join(pd.Series(x.dropna()).explode().value_counts().head(5).index)  # Top 3 gatunki
    }).rename(columns={
        'event_type': 'like_ratio',
        'track_id': 'tracks_played',
        'explicit': 'explicit_ratio'
    }).reset_index()

    # Spłaszczenie poziomów kolumn
    user_features.columns = ['_'.join(col).strip() for col in user_features.columns]

    user_features['skip_ratio'] = complete_table.groupby('user_id')['event_type'].apply(
        lambda x: (x == 'skip').mean())

    user_features = user_features.rename(columns={
        'like_ratio_<lambda>': 'like_ratio',
        'explicit_ratio_<lambda>': 'explicit_ratio',
        'artist_name_<lambda>': 'favorite_artist',
        'artist_genres_<lambda>': 'top_genres'
    }).reset_index()
    user_features.rename(columns={'user_id_': 'user_id'}, inplace=True)

    # Dodanie cech użytkownika
    user_metadata = users.rename(columns={
        'id': 'user_id'
    })[['user_id', 'user_fav_genres', 'premium_user']]
    # Połączenie danych z cechami użytkownika
    user_table = pd.merge(user_features, user_metadata, on='user_id', how='left')

    # Dodanie łącznego czasu słuchania do tabeli user_table
    user_table = pd.merge(user_table, pivoted_table, on='user_id', how='left')

    user_table = user_table.fillna(0)  # Wypełnianie brakujących danych zerami
    # Wybór zmiennych tekstowych do zakodowania
    categorical_columns = ['user_fav_genres', 'premium_user', 'favorite_artist',
                            'top_genres']
    # Konwersja kolumn zawierających listy na string
    for col in categorical_columns:
        if user_table[col].apply(lambda x: isinstance(x, list)).any():
            user_table[col] = user_table[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = ohe.fit_transform(user_table[categorical_columns])
    encoded_feature_names = ohe.get_feature_names_out(categorical_columns)
    # Przekształcenie zakodowanych danych na DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    # Połączenie zakodowanych danych z oryginalnym DataFrame
    user_table = pd.concat(
        [user_table.drop(categorical_columns, axis=1), encoded_df],
        axis=1
    )
    return user_table

def preprocess_columns(X):
    # Spłaszczanie krotek w nazwach kolumn
    X.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in X.columns]
    return X

def create_and_fit_model(user_table):
    user_table_clean = user_table.drop(columns=['user_id', 'index'], errors='ignore')
    pca = PCA(n_components=20)  # Zredukowane do 20 głównych składowych
    user_table_clean = pca.fit_transform(user_table_clean)
    knn = KMeans(
        n_clusters=5,
        init='k-means++',
        n_init=20,
        max_iter=500,
        tol=1e-15,
        random_state=1234,
        algorithm='elkan',
    )
    # Dopasowanie modelu
    knn.fit(user_table_clean)
    user_table['cluster'] = knn.labels_
    return knn, user_table, pca

def recommend_tracks(user_id, user_table, complete_table, knn, pca):
    # Przygotowanie danych użytkownika do predykcji
    user_data = user_table[user_table['user_id'] == user_id].drop(['user_id', 'index', 'cluster'], axis=1, errors='ignore')
    user_data = pca.transform(user_data)

    tracks_played = user_table.loc[user_table['user_id'] == user_id, 'tracks_played_count'].values[0]
    # Ustalanie liczby rekomendacji jako 100% przesłuchanych utworów (minimum 1 rekomendacja)
    n_recommendations = max(1, int(tracks_played * 1))
    # Zastosowanie modelu KMeans do przewidywania klastra dla użytkownika

    user_cluster = knn.predict(user_data)
    # Filtruj użytkowników w tym samym klastrze
    cluster_users = user_table[user_table['cluster'] == user_cluster[0]]
    complete_table_cluster = complete_table[complete_table['user_id'].isin(cluster_users["user_id"].unique())]
    # Zbierz najczęściej słuchane utwory w tym klastrze
    top_tracks = complete_table_cluster['track_id'].value_counts().head(n_recommendations).index.tolist()
    return top_tracks

def precision_recall(user_id, recommendations, complete_table):
    actual_tracks = complete_table.loc[complete_table['user_id'] == user_id, 'track_id'].tolist()
    relevant_recommendations = [track for track in recommendations if track in actual_tracks]
    precision = len(relevant_recommendations) / len(recommendations) if recommendations else 0
    recall = len(relevant_recommendations) / len(actual_tracks) if actual_tracks else 0
    return precision, recall


if __name__ == '__main__':
    # Sprawdzenie dla przykładowego użytkownika rekomendacja
    complete_table, users = read_and_join_data()
    user_table = process_joined_data(complete_table, users)
    user_table = preprocess_columns(user_table)

    knn, user_table, pca = create_and_fit_model(user_table)
    with open('workspace_model_data/knn_model.pkl', 'wb') as file:
        pickle.dump(knn, file)
    with open('workspace_model_data/pca.pkl', 'wb') as file:
        pickle.dump(pca, file)
    user_table.to_csv("workspace_model_data/user_table.csv", index=False)
    complete_table.to_csv("workspace_model_data/complete_table.csv", index=False)
    del knn
    del user_table
    del pca
    del complete_table
    with open('workspace_model_data/knn_model.pkl', 'rb') as file:
        knn = pickle.load(file)
    with open('workspace_model_data/pca.pkl', 'rb') as file:
        pca = pickle.load(file)
    user_table = pd.read_csv("workspace_model_data/user_table.csv")
    complete_table = pd.read_csv("workspace_model_data/complete_table.csv")

    user_ids = user_table['user_id'].unique()
    user_id = user_ids[1]  # ID użytkownika do analizy
    print("Wybrany użytkownik do testowania:", user_id)

    # Rekomendacje
    recommended_tracks = recommend_tracks(user_id, user_table, complete_table, knn, pca)

    precision, recall = precision_recall(user_id, recommended_tracks, complete_table)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
