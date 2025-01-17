import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ---------------------------
# 1) DATASET DEFINITION
# ---------------------------
class SongDataset(Dataset):
    """
    Reads CSV with columns: [user_id, track_id, event_type, track_id_idx].
    event_type is treated as an implicit 'rating' or signal of interaction.
    """
    def __init__(self, csv_path):
        super().__init__()
        self.data = pd.read_csv(csv_path)

        # Create user mapping (user -> user_idx)
        unique_users = self.data['user_id'].unique()
        self.user2idx = {u: i for i, u in enumerate(unique_users)}

        # Optionally create track mapping if you do NOT trust the existing 'track_id_idx'
        # But since it's given, we assume track_id_idx is already the correct integer index.
        # If you need to re-map, you'd do similarly: unique_tracks = self.data['track_id'].unique(), etc.

        # Prepare final (user_idx, track_idx, label) tuples
        self.samples = []
        for _, row in self.data.iterrows():
            user_idx = self.user2idx[row['user_id']]
            track_idx = row['track_id_idx']      # track_id_idx is already numeric
            label     = float(row['event_type']) # In many CF tasks label = 1 if interacted, else 0, ...
            self.samples.append((user_idx, track_idx, label))

        self.num_users  = len(self.user2idx)
        self.num_tracks = self.data['track_id_idx'].nunique()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, i, label = self.samples[idx]
        return (
            torch.tensor(u,     dtype=torch.long),
            torch.tensor(i,     dtype=torch.long),
            torch.tensor(label, dtype=torch.float)
        )

# ---------------------------
# 2) NEURAL CF MODEL
# ---------------------------
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
            nn.Linear(32, 1)  # Predict a single score (higher => more likely user likes track)
        )

    def forward(self, user_idx, track_idx):
        # Get embeddings
        user_emb  = self.user_embedding(user_idx)       # shape: [batch_size, embedding_dim]
        track_emb = self.track_embedding(track_idx)     # shape: [batch_size, embedding_dim]

        # Concat
        x = torch.cat([user_emb, track_emb], dim=-1)    # shape: [batch_size, 2*embedding_dim]

        # MLP forward
        out = self.mlp(x).squeeze(dim=-1)  # final shape: [batch_size]
        return out

# ---------------------------
# 3) EXAMPLE USAGE / TRAINING
# ---------------------------
if __name__ == "__main__":
    # Hyperparams
    csv_path       = "workspace_model_data/processed_table.csv"
    batch_size     = 16
    embedding_dim  = 16
    lr             = 1e-3
    epochs         = 5

    # Create dataset & dataloader
    dataset = SongDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate model
    model = NeuralCF(num_users=dataset.num_users,
                     num_tracks=dataset.num_tracks,
                     embedding_dim=embedding_dim)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Often for implicit feedback (event_type = 1 if played/streamed, 0 if not),
    # you might use BCEWithLogitsLoss. Here we demonstrate MSELoss as a simple example.
    criterion = nn.MSELoss()

    # TRAINING LOOP (simple example)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for user_idx, track_idx, label in dataloader:
            user_idx  = user_idx.to(device)
            track_idx = track_idx.to(device)
            label     = label.to(device)

            optimizer.zero_grad()
            preds = model(user_idx, track_idx)
            loss  = criterion(preds, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # -------------------------
    # 4) INFERENCE / GET TOP K
    # -------------------------
    def get_top_k_recommendations(model, user_raw_id, dataset, k=10):
        """
        Given a raw user_id (the same as in CSV) and a trained model,
        compute a score for each possible track in the dataset and pick top-k.

        Returns:
            top_k_track_idx (list[int]): top k track IDs (their indices in dataset)
        """
        model.eval()
        with torch.no_grad():
            # Convert the raw user id (the actual user_id from CSV) to user_idx
            user_idx_tensor = torch.tensor([dataset.user2idx[user_raw_id]], dtype=torch.long, device=device)

            # Make a list of all track indices
            all_track_indices = torch.arange(dataset.num_tracks, device=device)  # [0..num_tracks-1]

            # Expand the single user to match the shape of all tracks
            user_idx_expanded = user_idx_tensor.expand_as(all_track_indices)

            # Compute scores
            scores = model(user_idx_expanded, all_track_indices)

            # Top-k
            topk_vals, topk_indices = torch.topk(scores, k)
            # Convert to Python list
            topk_indices = topk_indices.cpu().numpy().tolist()

            return topk_indices

    # Example: Suppose we want top 10 recommendations for user_id=1
    user_of_interest = 1
    top10_track_indices = get_top_k_recommendations(model, user_of_interest, dataset, k=10)
    print(f"Top-10 track indices for user {user_of_interest} are: {top10_track_indices}")

    # If you want to map them back to actual track_id strings, you'd need an
    # inverse mapping from track_idx -> track_id. For instance, if your CSV
    # has track_id_idx -> track_id in a separate dictionary, you can do:
    #
    #   idx_to_track_id = ...
    #   recommended_track_ids = [idx_to_track_id[i] for i in top10_track_indices]
    #
    # That way you can see the actual track IDs.
