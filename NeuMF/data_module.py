import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchmetrics import Precision, Recall


# pytorch custom dataset
class NeuMFDataset(Dataset):
    def __init__(self, interactions, num_users, num_items, negative_sampling_ratio=1):
        """
        Args:
            interactions (pd.DataFrame): DataFrame zawierający kolumny 'user_id' i 'track_id'.
            num_users (int): Liczba unikalnych użytkowników.
            num_items (int): Liczba unikalnych utworów.
            negative_sampling_ratio (int): Ilość negatywnych próbek na każdą pozytywną próbkę.
        """
        self.user_ids = interactions['user_id'].values
        self.item_ids = interactions['track_id'].values
        self.num_users = num_users
        self.num_items = num_items
        self.negative_sampling_ratio = negative_sampling_ratio

        # Tworzenie słownika użytkowników do setów pozytywnych utworów
        self.user_positive_items = {}
        for u, i in zip(self.user_ids, self.item_ids):
            if u not in self.user_positive_items:
                self.user_positive_items[u] = set()
            self.user_positive_items[u].add(i)

        # Wszystkie możliwe utwory
        self.all_items = set(range(num_items))

    def __len__(self):
        return len(self.user_ids) * self.negative_sampling_ratio

    def __getitem__(self, idx):
        # Wybierz pozytywną próbkę
        pos_idx = idx // self.negative_sampling_ratio
        user = self.user_ids[pos_idx]
        pos_item = self.item_ids[pos_idx]

        # Próbkuj negatywną próbkę
        neg_item = np.random.randint(0, self.num_items)
        while neg_item in self.user_positive_items[user]:
            neg_item = np.random.randint(0, self.num_items)

        return torch.tensor(user, dtype=torch.long), torch.tensor(pos_item, dtype=torch.long), torch.tensor(neg_item, dtype=torch.long)


# 2. Definicja DataModule
class NeuMFDataModule(pl.LightningDataModule):
    def __init__(self, complete_table_path, batch_size=1024, negative_sampling_ratio=1, test_size=0.2, random_state=42):
        super().__init__()
        self.complete_table_path = complete_table_path
        self.batch_size = batch_size
        self.negative_sampling_ratio = negative_sampling_ratio
        self.test_size = test_size
        self.random_state = random_state

    def prepare_data(self):
        # Wczytanie danych
        self.complete_table = pd.read_csv(self.complete_table_path)

        # Mapowanie user_id i track_id na indeksy
        self.user_id_mapping = {id: idx for idx, id in enumerate(self.complete_table['user_id'].unique())}
        self.track_id_mapping = {id: idx for idx, id in enumerate(self.complete_table['track_id'].unique())}

        self.complete_table['user_id'] = self.complete_table['user_id'].map(self.user_id_mapping)
        self.complete_table['track_id'] = self.complete_table['track_id'].map(self.track_id_mapping)

        self.num_users = len(self.user_id_mapping)
        self.num_items = len(self.track_id_mapping)

    def setup(self, stage=None):
        # Podział na treningowy i walidacyjny
        train_df, val_df = train_test_split(
            self.complete_table,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )

        self.train_dataset = NeuMFDataset(
            interactions=train_df,
            num_users=self.num_users,
            num_items=self.num_items,
            negative_sampling_ratio=self.negative_sampling_ratio
        )

        self.val_dataset = NeuMFDataset(
            interactions=val_df,
            num_users=self.num_users,
            num_items=self.num_items,
            negative_sampling_ratio=self.negative_sampling_ratio
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

if __name__ == '__main__':
    complete_table_path = "workspace_model_data/complete_table.csv"

    # Sprawdzenie, czy plik istnieje
    if not os.path.exists(complete_table_path):
        raise FileNotFoundError(f"Plik {complete_table_path} nie został znaleziony.")

    # Inicjalizacja DataModule
    data_module = NeuMFDataModule(
        complete_table_path=complete_table_path,
        batch_size=2, #1024
        negative_sampling_ratio=1,  # Można zwiększyć dla większej ilości negatywnych próbek
        test_size=0.2,
        random_state=42
    )

    # Przygotowanie danych
    data_module.prepare_data()
    data_module.setup()
    # Przykładowe użycie
    for batch in data_module.train_dataloader():
        print(batch)
        #shape of batch
        print(batch[0].shape)
        break
