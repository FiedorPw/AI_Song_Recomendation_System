import numpy as np
import json

import matplotlib.pyplot as plt

track_keys = [
    "id",
    "name",
    "popularity",
    "duration_ms",
    "explicit",
    "id_artist",
    "release_date",
    "danceability",
    "energy",
    "key",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def read_jsonl_file(file_path):

    data = {}
    null_data = {}
    for key in track_keys:
        data[key] = []
        null_data[key] = 0
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():  # Ensure the line is not empty
                    json_obj = json.loads(line)
                    for key in track_keys:
                        if json_obj[key] is not None:
                            data[key].append(json_obj[key])
                        else:
                            null_data[key] += 1

    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    for key in track_keys:
        try:
            data[key] = sorted(data[key])
        except Exception as e:
            print(f"An error occurred: {e}")
    return data, null_data


# Example usage
file_path = "tracks.jsonl"  # Replace with your JSONL file path
contents, null_stats = read_jsonl_file(file_path)

plt.figure(figsize=(7, 4))
plt.xlabel("Popularność")
plt.ylabel("Liczba utworów")
plt.title("Popularność")
plt.hist(
    contents["popularity"],
    bins=max(contents["popularity"]) - min(contents["popularity"]),
)

plt.figure(figsize=(7, 4))
plt.xlabel("Czas trwania [ms]")
plt.ylabel("Liczba utworów")
plt.title("Czas trwania")
plt.hist(contents["duration_ms"], bins=50)

plt.figure(figsize=(4, 4))
plt.ylabel("Liczba utworów")
plt.title("Wulgarne?")
plt.bar(
    ["Nie", "Tak"],
    height=[contents["explicit"].count(0), contents["explicit"].count(1)],
)

plt.figure(figsize=(7, 4))
plt.xlabel("Taneczność")
plt.ylabel("Liczba utworów")
plt.title("Taneczność")
plt.hist(contents["danceability"], bins=50)

plt.figure(figsize=(7, 4))
plt.xlabel("Energia")
plt.ylabel("Liczba utworów")
plt.title("Energia")
plt.hist(contents["energy"], bins=50)

plt.figure(figsize=(7, 4))
plt.xlabel("Głośność")
plt.ylabel("Liczba utworów")
plt.title("Głośność")
plt.hist(contents["loudness"], bins=50)

plt.figure(figsize=(7, 4))
plt.xlabel("Wokalność")
plt.ylabel("Liczba utworów")
plt.title("Wokalność")
plt.hist(contents["speechiness"], bins=50)

plt.figure(figsize=(7, 4))
plt.xlabel("Akustyczność")
plt.ylabel("Liczba utworów")
plt.title("Akustyczność")
plt.hist(contents["acousticness"], bins=50)

log_insument = np.zeros_like(contents["instrumentalness"], dtype=np.float64)

for i in range(0, len(contents["instrumentalness"])):
    log_insument[i] = np.log10(
        contents["instrumentalness"][i],
        where=(contents["instrumentalness"][i] != 0.0),
    )
log_insument.sort()

plt.figure(figsize=(7, 4))
plt.xlabel("Numer próbki")
plt.ylabel("Instrumentalność")
plt.title("Instrumentalność")
plt.hist(contents["instrumentalness"], bins=50)


plt.figure(figsize=(7, 4))
plt.xlabel("Numer próbki")
plt.ylabel("Instrumentalność")
plt.title("(log10) Instrumentalność")
plt.scatter(np.arange(0, len(log_insument), 1), log_insument, marker=".")

plt.figure(figsize=(7, 4))
plt.xlabel("Udział publiczności")
plt.ylabel("Liczba utworów")
plt.title("Udział publiczności")
plt.hist(contents["liveness"], bins=50)

plt.figure(figsize=(7, 4))
plt.xlabel("Nastrój")
plt.ylabel("Liczba utworów")
plt.title("Nastrój")
plt.hist(contents["valence"], bins=50)

plt.figure(figsize=(7, 4))
plt.xlabel("Tempo")
plt.ylabel("Liczba utworów")
plt.title("Tempo")
plt.hist(contents["tempo"], bins=50)

keys = []
values = []
for k in null_stats:
    keys.append(k)
    values.append(null_stats[k])

plt.figure(figsize=(8, 6))
plt.title("Brakujące dane")
plt.bar(keys, height=values)
plt.xticks(rotation="vertical")

print(f"Null count:{null_stats}")

plt.show()
