# model

import sys
sys.path.append("./Loss")

from Loss import *

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Encoder


class TransformerEncoderModel(nn.Module):
    def __init__(self, n_features, nhead, nhid, nlayers, n_output, dropout=0.5):
        super(TransformerEncoderModel, self).__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(
            n_features, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # This is the new embedding layer
        self.embedding = nn.Linear(n_features, n_output)
        self.n_features = n_features
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.bias.data.zero_()
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.transformer_encoder(src)
        embedding = self.embedding(output)  # Compute embeddings

        return embedding.squeeze(0)  # Return both risk scores and embeddings


class MLPEncoderModel(nn.Module):
    def __init__(self, n_features, nhid, nlayers, n_output, dropout=0.5):
        super(MLPEncoderModel, self).__init__()
        self.model_type = 'MLP'

        # Define hidden layers
        self.hidden_layers = []
        for _ in range(nlayers - 2):
            self.hidden_layers.append(nn.Linear(nhid, nhid))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(dropout))

        # Define model layers
        self.layers = nn.Sequential(
            nn.Linear(n_features, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            *self.hidden_layers,
            nn.Linear(nhid, n_output),
            nn.ReLU()
        )

    def forward(self, x):
        embedding = self.layers(x)
        return embedding

# Risk score Predictor


class RiskscorePredictor(nn.Module):
    def __init__(self, n_features, nhid, dropout=0.5):
        super(RiskscorePredictor, self).__init__()
        self.RiskscoreMLP = nn.Sequential(
            nn.Linear(n_features, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, 1),
            nn.ReLU()
        )

    def forward(self, embedding):
        risk_score = torch.sigmoid(self.RiskscoreMLP(embedding))
        return risk_score.squeeze()


# Rejector network
class ConfidencePredictor(nn.Module):
    def __init__(self, n_features, nhid, dropout=0.5):
        super(ConfidencePredictor, self).__init__()
        self.ConfidenceMLP = nn.Sequential(
            nn.Linear(n_features, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, 1),
        )

    def forward(self, embedding):
        confidence_score = torch.tanh(self.ConfidenceMLP(embedding))
        return confidence_score.squeeze()

# Main network


class scRank(nn.Module):
    def __init__(self, n_features, nhead, nhid1, nhid2, nlayers, n_output, dropout=0.5, encoder_type="MLP"):
        super(scRank, self).__init__()
        self.encoder_type = encoder_type

        if self.encoder_type == "Transformer":
            self.encoder = TransformerEncoderModel(
                n_features, nhead, nhid1, nlayers, n_output, dropout)
        if self.encoder_type == "MLP":
            self.encoder = MLPEncoderModel(
                n_features, nhid1, nlayers, n_output, dropout)

        self.riskscorepredictor = RiskscorePredictor(n_output, nhid2, dropout)
        self.confidencepredictor = ConfidencePredictor(
            n_output, nhid2, dropout)

    def forward(self, x):
        embedding = self.encoder(x)
        risk_score = self.riskscorepredictor(embedding)
        confidence_score = self.confidencepredictor(embedding)

        return embedding, risk_score, confidence_score
