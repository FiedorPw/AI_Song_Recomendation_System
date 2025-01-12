import numpy as np
import json

import matplotlib.pyplot as plt

session_keys = ["session_id", "timestamp", "user_id", "track_id", "event_type"]


def read_jsonl_file(file_path):

    data = {}
    null_data = {}
    for key in session_keys:
        data[key] = []
        null_data[key] = 0
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():  # Ensure the line is not empty
                    json_obj = json.loads(line)
                    for key in session_keys:
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
    for key in session_keys:
        try:
            data[key] = sorted(data[key])
        except Exception as e:
            print(f"An error occurred: {e}")
    return data, null_data


file_path = "sessions.jsonl"  # Replace with your JSONL file path
contents, null_stats = read_jsonl_file(file_path)


event_types = {}
for k in contents["event_type"]:
    if k not in event_types:
        event_types[k] = 0
    event_types[k] += 1

plt.figure(figsize=(7, 4))
plt.title("Number of events")
plt.bar(
    event_types.keys(),
    height=event_types.values(),
)
print(event_types)
keys = []
values = []
for k in null_stats:
    keys.append(k)
    values.append(null_stats[k])
plt.figure(figsize=(8, 6))
plt.title("Null data")
plt.bar(keys, height=values)
plt.xticks(rotation="vertical")

print(f"Null count:{null_stats}")

plt.show()
