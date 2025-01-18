import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

class NeuralCF(nn.Module):
    """
    - User embedding
    - Item embedding
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


app = Flask(__name__)


def load_model_and_data():
    global model, user_map, track_map, device, num_users, num_tracks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Load user->idx mapping
    user_map = pd.read_csv("user_map.csv")  # columns: [user_id, user_idx]
    user_map_dict = dict(zip(user_map["user_id"], user_map["user_idx"]))

    # Load track_id_idx->track_id mapping
    track_map = pd.read_csv("track_map.csv")  # columns: [track_id_idx, track_id]
    idx2track_dict = dict(zip(track_map["track_id_idx"], track_map["track_id"]))

    num_users  = len(user_map_dict)       # total unique users
    num_tracks = len(idx2track_dict)      # total distinct tracks

    # Initialize the same NCF model architecture used in train_ncf.py
    embedding_dim = 16
    model_temp = NeuralCF(num_users=num_users, num_tracks=num_tracks, embedding_dim=embedding_dim)
    model_temp.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model_temp.to(device)
    model_temp.eval()

    # Store references globally
    model = model_temp

    user_map["user_idx"] = user_map["user_id"].map(user_map_dict)
    track_map["track_id"] = track_map["track_id_idx"].map(idx2track_dict)
    print("Model and data loaded successfully!")


def get_top_n_recommendations(user_id, n=5):
    """
    Given a user_id, compute the model's predicted rating/score for all tracks.
    Return the top-n track IDs with highest predicted score.
    """
    global model, user_map, track_map, device, num_tracks

    user_map_dict = dict(zip(user_map["user_id"], user_map["user_idx"]))
    if user_id not in user_map_dict:
        return []

    user_idx_val = user_map_dict[user_id]

    user_tensor  = torch.full((num_tracks,), fill_value=user_idx_val, dtype=torch.long).to(device)
    track_tensor = torch.arange(num_tracks, dtype=torch.long).to(device)

    with torch.no_grad():
        scores = model(user_tensor, track_tensor)

    scores_np = scores.cpu().numpy()

    top_indices = np.argsort(-scores_np)[:n]

    track_map_dict = dict(zip(track_map["track_id_idx"], track_map["track_id"]))
    recommendations = [track_map_dict[idx] for idx in top_indices]

    return recommendations


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
    # Parse the incoming JSON body
    data = request.get_json()
    user_id = data.get('user_id', None)

    if user_id is None:
        return jsonify({"error": "Please provide a valid 'user_id'"}), 400

    # Get top-5 recommendations
    top5 = get_top_n_recommendations(user_id, n=5)

    # Return recommendations
    return jsonify({"recommended_tracks": top5})


load_model_and_data()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000, debug=False)
