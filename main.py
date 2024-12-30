import math
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler


if __name__ == '__main__':
    artists = pd.read_json("artists.jsonl", lines=True)
    sessions = pd.read_json("sessions.jsonl", lines=True)
    tracks = pd.read_json("tracks.jsonl", lines=True)
    users = pd.read_json("users.jsonl", lines=True)

    sessions = sessions[sessions["event_type"] !="advertisment"]
    sessions_tracks = pd.merge(sessions, tracks, left_on="track_id", right_on="id", how="left")
    sessions_tracks.drop("id", axis=1, inplace=True)
    artists.columns = ["id","artist_name", "artist_genres"]
    artists_sessions_tracks = pd.merge(sessions_tracks, artists, left_on="id_artist", right_on="id", how="left")
    artists_sessions_tracks.drop("id", axis=1, inplace=True)
    users.columns = ["id", "user_name", "city", "street", "user_fav_genres", "premium_user"]
    complete_table = pd.merge(artists_sessions_tracks,users,left_on="user_id", right_on="id", how="left")
    complete_table.drop(["id"], axis=1, inplace=True)

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
    complete_table.to_csv("result.csv",index=False)

    # analiza rozkładów
    # column_names = ['session_id', 'timestamp', 'user_id', 'track_id', 'event_type', 'name',
    #    'popularity', 'duration_ms', 'explicit', 'id_artist', 'release_date',
    #    'danceability', 'energy', 'key', 'loudness', 'speechiness',
    #    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    #    'artist_name', 'artist_genres', 'user_name', 'city', 'street',
    #    'user_fav_genres', 'premium_user', 'listening_time']
    column_names = ['session_id', 'user_id', 'track_id', 'event_type', 'name',
       'popularity', 'duration_ms', 'explicit', 'id_artist',
       'danceability', 'energy', 'key', 'loudness', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'artist_name', 'artist_genres', 'user_name', 'city', 'street',
       'user_fav_genres', 'premium_user', 'listening_time']
    # for column_name in column_names:
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(complete_table[column_name], bins=30000, edgecolor='black')
    #     plt.title(f'Histogram dla kolumny {column_name} w tabeli sessions połączonej z pozostałymi tabelami')
    #     plt.xlabel(column_name)
    #     plt.ylabel('Liczba wystąpień')
    #     plt.show()
    sc = StandardScaler()
    encoder = LabelEncoder()
    # complete_table["timestamp"] = sc.fit_transform([complete_table["timestamp"]])
    complete_table["event_type"] = complete_table["event_type"].apply(
        lambda x: 1 if x == "like" else (0 if x == "skip" else None)
    )
    complete_table["name"] = encoder.fit_transform(complete_table['name'])
    complete_table["popularity"] = sc.fit_transform(np.log(complete_table["popularity"]).values.reshape(-1, 1))
    complete_table["duration_ms"] = sc.fit_transform(np.log(complete_table["duration_ms"]).values.reshape(-1, 1))
    # complete_table["release_date"] = sc.fit_transform([complete_table["release_date"]])
    complete_table["danceability"] = sc.fit_transform(np.exp(complete_table["danceability"]).values.reshape(-1, 1))
    complete_table["energy"] = sc.fit_transform(np.exp(complete_table["energy"]).values.reshape(-1, 1))
    complete_table["key"] = sc.fit_transform(complete_table["key"].values.reshape(-1, 1))
    complete_table["loudness"] = sc.fit_transform(complete_table["loudness"].values.reshape(-1, 1))
    complete_table["speechiness"] = sc.fit_transform((complete_table["speechiness"]**(1/5)).values.reshape(-1, 1))
    complete_table["acousticness"] = sc.fit_transform((complete_table["acousticness"]**(1/3)).values.reshape(-1, 1))
    complete_table["instrumentalness"] = sc.fit_transform(complete_table["instrumentalness"].values.reshape(-1, 1))
    complete_table["liveness"] = sc.fit_transform(np.log(complete_table["liveness"]).values.reshape(-1, 1))
    complete_table["valence"] = sc.fit_transform(complete_table["valence"].values.reshape(-1, 1))
    complete_table["tempo"] = sc.fit_transform(complete_table["tempo"].values.reshape(-1, 1))
    complete_table['artist_name'] = encoder.fit_transform(complete_table['artist_name'])
    complete_table['artist_genres'] = encoder.fit_transform(complete_table['artist_genres'])
    complete_table['user_name'] = encoder.fit_transform(complete_table['user_name'])
    complete_table['city'] = encoder.fit_transform(complete_table['city'])
    complete_table['street'] = encoder.fit_transform(complete_table['street'])
    complete_table['user_fav_genres'] = encoder.fit_transform(complete_table['user_fav_genres'])
    complete_table["premium_user"] = complete_table['premium_user'].astype(int)
    complete_table["listening_time"] = sc.fit_transform(np.log(complete_table["listening_time"]).values.reshape(-1, 1))
    complete_table = complete_table.dropna()

    X = complete_table.drop(columns=['event_type'])
    y = complete_table['event_type']
    mi_results = {}
    for column in X.columns:
        if X[column].dtype in ['int64', 'float64']:  # Dla zmiennych numerycznych
            mi_score = mutual_info_regression(X[column].values.reshape(-1, 1), y)
        else:  # Dla zmiennych kategorycznych
            mi_score = mutual_info_classif(X[column].values.reshape(-1, 1), y)
        mi_results[column] = mi_score[0]

    # Posortowanie wyników w porządku malejącym
    sorted_mi_results = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)

    # Wyświetlenie wyników
    for feature, score in sorted_mi_results:
        print(f'Feature: {feature}, MI Score: {score}')

    # Podział na cechy i cel
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)
    undersampler = RandomUnderSampler(random_state=1234)
    X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)

    param_grid = {
        'penalty': ['l1'],  # Typ regularyzacji
        'C': [0.01, 0.1, 1, 10, 100],  # Siła regularyzacji
        'solver': ['liblinear', 'saga'],  # Solvery obsługujące l1 i l2
        'max_iter': [10000]  # Liczba iteracji
    }

    # Inicjalizacja modelu
    model = LogisticRegression(random_state=1234)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',  # Metryka oceny
        cv=10,  # Liczba podziałów w walidacji krzyżowej
        verbose=1,  # Poziom szczegółowości
        n_jobs=-1  # Użyj wszystkich rdzeni procesora
    )
    noise = np.random.normal(loc=0, scale=1, size=X_resampled.shape[0])
    X_resampled['noise'] = noise
    grid_search.fit(X_resampled, y_resampled)
    best_params = grid_search.best_params_
    print("Najlepsze parametry: ", best_params)
    # Wynik na danych testowych z najlepszym modelem
    best_model = grid_search.best_estimator_
    noise = np.random.normal(loc=0, scale=1, size=X_test.shape[0])
    X_test['noise'] = noise
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    # Pobranie współczynników modelu
    coefficients = best_model.coef_.flatten()

    # Połączenie cech z ich współczynnikami
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Coefficient': coefficients
    })

    # Posortowanie cech według wartości współczynników
    feature_importance = feature_importance.reindex(
        feature_importance['Coefficient'].abs().sort_values(ascending=False).index)

    # Wyświetlenie najważniejszych cech
    print("Najważniejsze cechy (według współczynników):")
    print(feature_importance)

