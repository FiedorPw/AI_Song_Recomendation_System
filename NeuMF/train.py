import os
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from data_module import NeuMFDataModule
from torchmetrics import Precision, Recall


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=30, mlp_dims=[64, 32, 16], dropout=0.2):
        super(NeuMF, self).__init__()
        # GMF (Generalized Matrix Factorization) - część liniowa
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP (Multi-Layer Perceptron) - część nieliniowa
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)

        mlp_layers = []
        input_dim = embedding_dim * 2
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(input_dim, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout))
            input_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Połączenie wyników GMF i MLP do wspólnej warstwy przewidującej (tzw. NeuMF)
        self.predict_layer = nn.Linear(embedding_dim + mlp_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

        # Zainicjalizuj wagi
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user, item):
        # GMF - iloczyn Hadamarda (punktowy)
        gmf_user = self.gmf_user_embedding(user)
        gmf_item = self.gmf_item_embedding(item)
        gmf_output = gmf_user * gmf_item

        # MLP - konkatenacja wektorów użytkownika i przedmiotu, a potem przejście przez sieć
        mlp_user = self.mlp_user_embedding(user)
        mlp_item = self.mlp_item_embedding(item)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Połączenie GMF i MLP
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        logit = self.predict_layer(combined)
        output = self.sigmoid(logit).squeeze()  # zwraca wartość w przedziale (0, 1)
        return output


class NeuMFModule(pl.LightningModule):
    """
    Klasa główna do trenowania NeuMF z użyciem PyTorch Lightning.
    Odpowiada za pętlę treningową, walidacyjną, optymalizację i logowanie metryk.
    """
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=30,
        mlp_dims=[64, 32, 16],
        dropout=0.2,
        lr=0.001
    ):
        super(NeuMFModule, self).__init__()
        self.model = NeuMF(num_users, num_items, embedding_dim, mlp_dims, dropout)
        self.lr = lr
        self.criterion = nn.BCELoss()

        # Metryki: precyzja i recall do binarnej klasyfikacji
        self.train_precision = Precision(task='binary')
        self.train_recall = Recall(task='binary')
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')

    def forward(self, user, item):
        return self.model(user, item)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        user, pos_item, neg_item = batch
        # Pozytywne próbki
        pos_preds = self(user, pos_item)
        pos_labels = torch.ones_like(pos_preds)

        # Negatywne próbki
        neg_preds = self(user, neg_item)
        neg_labels = torch.zeros_like(neg_preds)

        # Obliczenie łącznej straty (BCELoss) dla pozytywnych i negatywnych
        loss_pos = self.criterion(pos_preds, pos_labels)
        loss_neg = self.criterion(neg_preds, neg_labels)
        loss = loss_pos + loss_neg

        # Zbieramy predykcje i etykiety, aby wyliczyć metryki
        preds = torch.cat([pos_preds, neg_preds], dim=0)
        targets = torch.cat([pos_labels, neg_labels], dim=0)
        self.train_precision(preds, targets)
        self.train_recall(preds, targets)

        # Logowanie strat i metryk
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/precision', self.train_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/recall', self.train_recall, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        user, pos_item, neg_item = batch
        # Pozytywne próbki
        pos_preds = self(user, pos_item)
        pos_labels = torch.ones_like(pos_preds)

        # Negatywne próbki
        neg_preds = self(user, neg_item)
        neg_labels = torch.zeros_like(neg_preds)

        # Obliczenie łącznej straty dla walidacji
        loss_pos = self.criterion(pos_preds, pos_labels)
        loss_neg = self.criterion(neg_preds, neg_labels)
        loss = loss_pos + loss_neg

        # Metryki walidacyjne
        preds = torch.cat([pos_preds, neg_preds], dim=0)
        targets = torch.cat([pos_labels, neg_labels], dim=0)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self, outputs):
        # Po zakończeniu każdej epoki resetujemy metryki, by nie kumulowały się w nieskończoność
        self.train_precision.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self, outputs):
        self.val_precision.reset()
        self.val_recall.reset()


if __name__ == '__main__':
    complete_table_path = "workspace_model_data/complete_table.csv"

    # Sprawdzenie, czy plik istnieje
    if not os.path.exists(complete_table_path):
        raise FileNotFoundError(f"Plik {complete_table_path} nie został znaleziony.")

    # Inicjalizacja specjalnego modułu danych (DataModule),
    # który zarządza ładowaniem i przetwarzaniem danych
    data_module = NeuMFDataModule(
        complete_table_path=complete_table_path,
        batch_size=1024,
        negative_sampling_ratio=1,  # Można zwiększyć, aby dodać więcej negatywnych próbek
        test_size=0.2,
        random_state=42
    )

    # Przygotowanie danych (np. jeśli trzeba je pobrać) oraz ich podział
    data_module.prepare_data()
    data_module.setup()

    # Tworzymy instancję naszego modelu
    model = NeuMFModule(
        num_users=data_module.num_users,
        num_items=data_module.num_items,
        embedding_dim=30,
        mlp_dims=[64, 32, 16],
        dropout=0.2,
        lr=0.001
    )

    # Logger do TensorBoard
    logger = TensorBoardLogger("lightning_logs", name="NeuMF")

    # Inicjalizacja trenera
    trainer = Trainer(
        logger=logger,
        max_epochs=10,
        accelerator='cpu',
        # accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        # devices=1 if torch.cuda.is_available() else 0
        devices=1
    )

    # Rozpoczęcie treningu
    trainer.fit(model, datamodule=data_module)

    # Testowanie modelu po treningu
    trainer.test(model, datamodule=data_module)
