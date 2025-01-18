# file: abtest_proxy.py
from flask import Flask, request, jsonify, Response
import requests
import random
import json
import io

app = Flask(__name__)

# In-memory storage for feedback in the format: [(model, track_id, reaction), ...]
feedback_data = []

track_model_mapping = {}
# Example endpoints for each model (adjust to real addresses/ports)
NCF_ENDPOINT = "http://localhost:7000/predict"
KNN_ENDPOINT = "http://localhost:6000/predict"

@app.route('/recommendations', methods=['GET'])
def recommendations():
    try:
        data = request.get_json()
        user_id = data.get('user_id', None)
    except:
        return jsonify({"error": "Invalid JSON format."}), 400

    if user_id is None:
        return jsonify({"error": "Please provide a valid 'user_id'"}), 400

    # Randomly choose which model to call
    chosen_model = random.choice(['ncf', 'knn'])
    target_url = NCF_ENDPOINT if chosen_model == 'ncf' else KNN_ENDPOINT

    payload = {"user_id": user_id}

    try:
        resp = requests.get(target_url, json=payload, timeout=5)
        resp.raise_for_status()
        tracks = resp.json().get("recommended_tracks", [])
    except requests.exceptions.RequestException:
        tracks = [f"track{i}" for i in range(1,6)]

    # Store which model was used for these tracks
    for track in tracks:
        track_model_mapping[track] = chosen_model  # Added this line to store model information

    return jsonify(tracks), 200

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Now accepts JSON in the format:
    {
        "feedback": [
            {"track_id": "track1", "reaction": "like"},
            {"track_id": "track2", "reaction": "dislike"}
        ]
    }
    """
    data = request.get_json()
    if not data or 'feedback' not in data:
        return jsonify({"error": "Invalid feedback format."}), 400

    fb = data.get("feedback", [])
    if not isinstance(fb, list):
        return jsonify({"error": "'feedback' should be a list."}), 400

    for entry in fb:
        track = entry.get("track_id", "").strip()
        reaction = entry.get("reaction", "").strip()

        if not track or not reaction:
            return jsonify({"error": "Each feedback entry must include 'track_id' and 'reaction'."}), 400

        # Get the model used for this track and add it to feedback_data
        model = track_model_mapping.get(track, 'unknown')
        feedback_data.append((model, track, reaction))

        # Clean up the tracking dictionary
        if track in track_model_mapping:
            del track_model_mapping[track]

    return jsonify({"status": "ok", "received": len(fb)}), 200

@app.route('/abtest', methods=['GET'])
def abtest():
    # Return the stored feedback as CSV
    # Format: model,track_id,reaction
    csv_buffer = io.StringIO()
    csv_buffer.write("model,track_id,reaction\n")
    print(feedback_data)
    for model, track_id, reaction in feedback_data:
        csv_buffer.write(f"{model},{track_id},{reaction}\n")

    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=abtest_results.csv"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
