import os
import torch
from torch import nn


class ReviewClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=100,
        hidden_size=200,
        layers=1,
        dropout=.2
    ):
        super().__init__()

        # store model hyperparameters so they can be
        # saved/loaded with the model itself
        self._hyperparameters = {
            "vocab_size": vocab_size,
            "embedding_size": embedding_size,
            "hidden_size": hidden_size,
            "layers": layers,
            "dropout": dropout
        }

        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.encode = nn.Embedding(vocab_size, embedding_size)
        self.decode = nn.Linear(hidden_size, 1)
        self.recurrent = nn.LSTM(
            embedding_size, hidden_size, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        # TODO?: init_weights

    def forward(self, input):
        encoded = self.encode(input)
        encoded = self.dropout(encoded)
        _, (state, _) = self.recurrent(encoded)
        # take state from the last layer of LSTM
        state = state[-1]
        state = self.dropout(state)
        decoded = self.decode(state)
        decoded = self.sigmoid(decoded)
        return decoded

    @staticmethod
    def load(name="model"):
        assert os.path.exists(f"{name}.pt")

        model_data = torch.load(f"{name}.pt", map_location=torch.device("cpu"))
        model = ReviewClassifier(**model_data["hyperparameters"])
        model.load_state_dict(model_data["state"])
        model.eval()

        return model


class Classifier:
    def __init__(self):
        self.network = ReviewClassifier.load()

    def classify(self, input):
        # TODO: preprocess input, map output to label
        return self.network(input).item()
