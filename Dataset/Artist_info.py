import numpy as np
import json

import matplotlib.pyplot as plt

artist_keys = ["id", "name", "genres"]


def read_jsonl_file(file_path):

    data = {}
    null_data = {}
    for key in artist_keys:
        data[key] = []
        null_data[key] = 0
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():  # Ensure the line is not empty
                    json_obj = json.loads(line)
                    for key in artist_keys:
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
    return data, null_data


file_path = "artists.jsonl"  # Replace with your JSONL file path
contents, null_stats = read_jsonl_file(file_path)

keys = []
values = []
for k in artist_keys:
    keys.append(k)
    values.append(null_stats[k])

genres_sizes = {}

for i in range(0, len(contents["genres"])):
    if len(contents["genres"][i]) not in genres_sizes:
        genres_sizes[len(contents["genres"][i])] = 0
    genres_sizes[len(contents["genres"][i])] += 1

ordered_sizes = np.zeros(len(genres_sizes) + 1, dtype=int)
for j in range(min(genres_sizes), max(genres_sizes) + 1):
    ordered_sizes[j] = genres_sizes[j]


plt.figure(figsize=(7, 4))
plt.xlabel("No. of genres")
plt.ylabel("No. of artists")
plt.title("No. of genres associated with the artist")
plt.bar(
    np.arange(start=min(genres_sizes) - 1, stop=max(genres_sizes) + 1, step=1),
    ordered_sizes,
)

plt.figure(figsize=(8, 6))
plt.title("Null data")
plt.bar(keys, height=values)
plt.xticks(rotation="vertical")

print(null_stats)

plt.show()
