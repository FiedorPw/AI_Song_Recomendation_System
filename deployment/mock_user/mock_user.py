
from flask import Flask, jsonify
import requests
import random
import threading
import time
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROXY_URL = 'http://localhost:8080'
REQUEST_INTERVAL = 5  # seconds between requests

# Possible actions to assign to each song
ACTIONS = ['like', 'dislike', 'skip']

def send_request_and_feedback():
    while True:
        try:
            # Step 1: Get recommendations from proxy
            json_user = {"user_id": random.randint(100,200)}
            response = requests.get(f"{PROXY_URL}/recommendations", json=json_user)
            # response.raise_for_status()  # sprawdza czy nie było błędu w zapytaniu HTTP (wyrzuca wyjątek jeśli status != 2xx)
            tracks = response.json()  # parsuje odpowiedź JSON z proxy do pythonowego obiektu (listy track_id)
            print(tracks)

            # Log received recommendations
            logger.info("Received recommendations from proxy")

            # Step 2: Generate feedback for each track
            feedback_list = []
            for track_id in tracks:
                feedback_list.append({
                    "track_id": track_id,
                    "reaction": random.choice(ACTIONS)
                })

            # Step 3: Send feedback to proxy
            feedback_payload = {"feedback": feedback_list}
            feedback_response = requests.post(
                f"{PROXY_URL}/feedback",
                json=feedback_payload
            )
            feedback_response.raise_for_status()

            # Log successful feedback submission
            logger.info(f"Sent feedback for {len(feedback_list)} tracks")
            logger.info("-------------------")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to communicate with proxy: {str(e)}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")

        # Wait for the specified interval before next request
        time.sleep(REQUEST_INTERVAL)

@app.route('/status', methods=['GET'])
def status():
    """
    Endpoint to check if the service is running
    """
    return jsonify({
        "status": "running",
        "message": "Mock user is actively sending requests every 5 seconds"
    })

def start_background_thread():
    """
    Start the background thread for sending requests
    """
    thread = threading.Thread(target=send_request_and_feedback, daemon=True)
    thread.start()
    logger.info("Started background thread for sending requests")

start_background_thread()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
