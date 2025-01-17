from flask import Flask, jsonify
import requests
import random

app = Flask(__name__)

# Configuration
RECOMMENDATION_SERVICE_URL = 'http://other-container:8080/recommendations'  # Replace with the actual URL/path

# Possible actions to assign to each song
ACTIONS = ['like', 'play', 'skip']

@app.route('/', methods=['GET'])
def get_songs_with_actions():
    try:
        # Fetch recommendations from the other container
        response = requests.get(RECOMMENDATION_SERVICE_URL, timeout=5)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        # Validate the structure of the received data
        if 'songs' not in data or not isinstance(data['songs'], list):
            return jsonify({'error': 'Invalid data format from recommendation service.'}), 500

        # Ensure there are exactly 5 songs
        if len(data['songs']) != 5:
            return jsonify({'error': 'Recommendation service did not return exactly 5 songs.'}), 500

        # Create a dictionary to hold actions for each song
        actions = {}
        for song in data['songs']:
            song_id = song.get('id')
            if not song_id:
                return jsonify({'error': 'Song entry missing "id".'}), 500
            actions[song_id] = random.choice(ACTIONS)

        # Combine the original songs with the actions
        combined_response = {
            'songs': data['songs'],
            'actions': actions
        }

        return jsonify(combined_response), 200

    except requests.exceptions.RequestException as e:
        # Handle any requests exceptions (e.g., connection errors, timeouts)
        return jsonify({'error': f'Failed to fetch recommendations: {str(e)}'}), 502
    except ValueError:
        # Handle JSON decoding errors
        return jsonify({'error': 'Invalid JSON received from recommendation service.'}), 502
    except Exception as e:
        # Catch-all for any other exceptions
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the Flask app on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000)
