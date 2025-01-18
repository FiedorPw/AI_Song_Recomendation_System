import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class SongDataset(Dataset):
    """
    Reads a CSV with columns: [user_id, track_id, event_type, track_id_idx].
    - user_id: string or numeric user identifier
    - track_id: string or numeric track identifier
    - event_type: your implicit/explicit feedback (e.g., 1=played, 0=not played)
    - track_id_idx: integer index for the track
    """
    def __init__(self, csv_path):
        super().__init__()
        self.data = pd.read_csv(csv_path)

        # Create user mapping (user -> user_idx)
        unique_users = self.data['user_id'].unique().tolist()
        self.user2idx = {u: i for i, u in enumerate(unique_users)}

        # Create an inverse of track_id_idx if needed
        #   track_id_idx is already numeric, but we'll also track the mapping
        #   to get a distinct set of track_id_idx -> track_id
        #   e.g.: idx2track[0] = '1T76pppCs1bFBcVpRCWzxS'
        grouped_tracks = self.data[['track_id_idx', 'track_id']].drop_duplicates()
        self.idx2track = dict(zip(grouped_tracks['track_id_idx'], grouped_tracks['track_id']))

        # Prepare final (user_idx, track_idx, label) tuples
        self.samples = []
        for _, row in self.data.iterrows():
            user_idx = self.user2idx[row['user_id']]
            track_idx = int(row['track_id_idx'])     # track_id_idx is already numeric
            label = float(row['event_type'])         # e.g. 1 if played, 0 otherwise
            self.samples.append((user_idx, track_idx, label))

        self.num_users  = len(self.user2idx)
        self.num_tracks = len(self.idx2track)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, i, label = self.samples[idx]
        return (
            torch.tensor(u,     dtype=torch.long),
            torch.tensor(i,     dtype=torch.long),
            torch.tensor(label, dtype=torch.float)
        )

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
        # Get embeddings
        user_emb  = self.user_embedding(user_idx)       # shape: [batch_size, embedding_dim]
        track_emb = self.track_embedding(track_idx)     # shape: [batch_size, embedding_dim]

        # Concat
        x = torch.cat([user_emb, track_emb], dim=-1)    # shape: [batch_size, 2*embedding_dim]

        # MLP forward
        out = self.mlp(x).squeeze(dim=-1)  # shape: [batch_size]
        return out

if __name__ == "__main__":
    csv_path      = "workspace_model_data/processed_table.csv"  # Update to your CSV
    batch_size    = 16
    embedding_dim = 16
    lr            = 1e-3
    epochs        = 5

    dataset = SongDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NeuralCF(num_users=dataset.num_users,
                     num_tracks=dataset.num_tracks,
                     embedding_dim=embedding_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for user_idx, track_idx, label in dataloader:
            user_idx  = user_idx.to(device)
            track_idx = track_idx.to(device)
            label     = label.to(device)

            optimizer.zero_grad()
            preds = model(user_idx, track_idx)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Save the trained weights
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model weights saved to model_weights.pth")

    # Save user->idx mapping
    user_map_df = pd.DataFrame(list(dataset.user2idx.items()), columns=['user_id', 'user_idx'])
    user_map_df.to_csv("user_map.csv", index=False)
    print("User map saved to user_map.csv")

    # Save track_idx->track_id mapping
    track_map_df = pd.DataFrame(list(dataset.idx2track.items()), columns=['track_id_idx', 'track_id'])
    track_map_df.to_csv("track_map.csv", index=False)
    print("Track map saved to track_map.csv")

    print("Training script completed successfully.")
