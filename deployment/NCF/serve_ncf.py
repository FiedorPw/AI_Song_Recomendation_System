import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np


# -------------------------------
# 1) Define your NCF model class
# -------------------------------
#
#
class NeuralCF(nn.Module):
    """
    Minimal neural collaborative filtering model:
    - User embedding
    - Item embedding
    - MLP on top of concatenated embeddings
    """
    def __init__(self, num_users, num_tracks, embedding_dim=32):
        super().__init__()
        self.user_embedding  = nn.Embedding(num_embeddings=num_users,  embedding_dim=embedding_dim)
        self.track_embedding = nn.Embedding(num_embeddings=num_tracks, embedding_dim=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # single score
        )

    def forward(self, user_idx, track_idx):
        """
        user_idx: tensor of shape [batch_size]
        track_idx: tensor of shape [batch_size]
        """
        user_emb  = self.user_embedding(user_idx)       # [batch_size, embedding_dim]
        track_emb = self.track_embedding(track_idx)     # [batch_size, embedding_dim]
        x = torch.cat([user_emb, track_emb], dim=-1)    # [batch_size, 2*embedding_dim]
        out = self.mlp(x).squeeze(dim=-1)               # [batch_size]
        return out


# ---------------------------------------------------------------------
# 2) Initialize Flask application
# ---------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------
# 3) Load model artifacts (weights, user_map, track_map) at startup
# ---------------------------------------------------------------------
#   - model_weights.pth   : Trained NeuralCF weights
#   - user_map.csv        : Maps user_id -> user_idx
#   - track_map.csv       : Maps track_id_idx -> track_id
#   - device              : 'cuda' if available, else 'cpu'


def load_model_and_data():
    """
    Loads the trained model weights, user map, and track map once before handling the first request.
    """
    global model, user_map, track_map, device, num_users, num_tracks
    #  a) Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  b) Load user->idx mapping
    user_map = pd.read_csv("user_map.csv")  # columns: [user_id, user_idx]
    user_map_dict = dict(zip(user_map["user_id"], user_map["user_idx"]))

    #  c) Load track_id_idx->track_id mapping
    track_map = pd.read_csv("track_map.csv")  # columns: [track_id_idx, track_id]
    idx2track_dict = dict(zip(track_map["track_id_idx"], track_map["track_id"]))

    #  d) Identify the max user_idx and max track_idx from loaded CSVs
    #     We'll assume user_map.csv consistently enumerates user_idx from 0..(N-1)
    #     and track_map.csv enumerates track_id_idx from 0..(M-1).
    num_users  = len(user_map_dict)       # total unique users
    num_tracks = len(idx2track_dict)      # total distinct tracks

    #  e) Initialize the same NCF model architecture used in train_ncf.py
    embedding_dim = 16  # must match what was used in training
    model_temp = NeuralCF(num_users=num_users, num_tracks=num_tracks, embedding_dim=embedding_dim)
    model_temp.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model_temp.to(device)
    model_temp.eval()

    #  f) Store references globally
    model = model_temp

    #  g) Also store user and track maps as dictionaries for quick access
    #     We'll store them as part of 'user_map' and 'track_map' variables for reference.
    #     If you prefer, you can store them as separate dictionaries for clarity.
    user_map["user_idx"] = user_map["user_id"].map(user_map_dict)
    track_map["track_id"] = track_map["track_id_idx"].map(idx2track_dict)
    print("Model and data loaded successfully!")


# ---------------------------------------------------------------------
# 4) Define inference logic
# ---------------------------------------------------------------------
def get_top_n_recommendations(user_id, n=5):
    """
    Given a user_id, compute the model's predicted rating/score for all tracks.
    Return the top-n track IDs with highest predicted score.
    """
    # Access globals:
    global model, user_map, track_map, device, num_tracks

    #  a) Convert user_id to user_idx
    #     If user does not exist, handle gracefully by returning empty list.
    user_map_dict = dict(zip(user_map["user_id"], user_map["user_idx"]))
    if user_id not in user_map_dict:
        return []  # user not found

    user_idx_val = user_map_dict[user_id]

    #  b) Create a user Tensor of the same size as the total number of tracks
    #     This allows us to do a forward pass over all track IDs in a single batch.
    user_tensor  = torch.full((num_tracks,), fill_value=user_idx_val, dtype=torch.long).to(device)
    track_tensor = torch.arange(num_tracks, dtype=torch.long).to(device)

    #  c) Run inference in eval mode
    with torch.no_grad():
        scores = model(user_tensor, track_tensor)  # shape: [num_tracks]

    #  d) Convert to CPU NumPy array for sorting
    scores_np = scores.cpu().numpy()

    #  e) Sort track indices by descending scores, pick top-n
    top_indices = np.argsort(-scores_np)[:n]

    #  f) Map track_id_idx -> track_id
    #     track_map.csv has columns [track_id_idx, track_id]
    #     We'll retrieve the track IDs for 'top_indices'
    track_map_dict = dict(zip(track_map["track_id_idx"], track_map["track_id"]))
    recommendations = [track_map_dict[idx] for idx in top_indices]

    return recommendations


# ---------------------------------------------------------------------
# 5) Define a Flask endpoint for prediction
# ---------------------------------------------------------------------
@app.route('/predict', methods=['GET'])
def predict():
    """
    Example request JSON:
    {
       "user_id": 10
    }

    Returns:
    {
       "recommended_tracks": ["trackA", "trackB", "..."]
    }
    """
    # a) Parse the incoming JSON body
    data = request.get_json()
    user_id = data.get('user_id', None)

    if user_id is None:
        return jsonify({"error": "Please provide a valid 'user_id'"}), 400

    # b) Get top-5 recommendations
    top5 = get_top_n_recommendations(user_id, n=5)

    # c) Return recommendations
    return jsonify({"recommended_tracks": top5})


load_model_and_data()
# ---------------------------------------------------------------------
# 6) Run the Flask app on port 7000
# ---------------------------------------------------------------------
if __name__ == '__main__':
    # debug=True is useful during development but should be off in production
    app.run(host='0.0.0.0', port=7000, debug=True)
