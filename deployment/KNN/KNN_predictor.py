from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

from modelknn import recommend_tracks, precision_recall

app = Flask(__name__)

# Load pre-trained model (KMeans) from pickle file
with open('workspace_model_data/knn_model.pkl', 'rb') as file:
    knn = pickle.load(file)

# Load PCA transformer from pickle file
with open('workspace_model_data/pca.pkl', 'rb') as file:
    pca = pickle.load(file)

user_table = pd.read_csv("workspace_model_data/user_table.csv")
complete_table = pd.read_csv("workspace_model_data/complete_table.csv")

@app.route('/predict', methods=['GET'])
def predict():
    """
    This endpoint expects a JSON payload with 'user_id'.
    Example request JSON:
        {
            "user_id": 12345
        }

    It then returns a JSON response with the top 5 recommended track IDs.
    """

    data = request.get_json()
    user_id = data['user_id']

    # Use the recommend_tracks() function from modelKNN to get track recommendations
    recommended = recommend_tracks(user_id, user_table, complete_table, knn, pca)
    top_5_tracks = recommended[:5]
    # Return the result as JSON
    return jsonify({"recommended_tracks": top_5_tracks})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)
