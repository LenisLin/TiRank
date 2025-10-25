# model
import os
import pickle
import math
import random
import numpy as np
from collections import Counter

import torch
from torch import nn
import torch.nn.functional as F

from .Loss import *

# Initial


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def initial_model_para(
        savePath,
        nhead = 2,
        nhid1=96, 
        nhid2=8, 
        n_output=32,
        nlayers=3,
        n_pred=1, 
        dropout=0.5,
        mode = "Cox",
        infer_mode="SC",
        encoder_type = "MLP"
):
    savePath_2 = os.path.join(savePath,"2_preprocessing")
    savePath_3 = os.path.join(savePath,"3_Analysis")
    savePath_data2train = os.path.join(savePath_3,"data2train")

    ## Load train bulk gene pair matrix 
    f = open(os.path.join(savePath_2, 'train_bulk_gene_pairs_mat.pkl'), 'rb')
    train_bulk_gene_pairs_mat = pickle.load(f)
    f.close()

    ## Load patholabels
    f = open(os.path.join(savePath_data2train, 'patholabels.pkl'), 'rb')
    patholabels = pickle.load(f)
    f.close()
    n_patho_cluster = len(Counter(patholabels).keys())

    # Pack all the parameters into a dictionary
    model_para = {
        'n_features': train_bulk_gene_pairs_mat.shape[1],
        'nhead': nhead,
        'nhid1': nhid1,
        'nhid2': nhid2,
        'n_output': n_output,
        'nlayers': nlayers,
        'n_pred': n_pred,
        "n_patho" : n_patho_cluster,
        'dropout': dropout,
        'mode': mode,
        'infer_mode': infer_mode,
        'encoder_type': encoder_type,
        'model_save_path' : os.path.join(savePath_3,"checkpoints")
    }

    with open(os.path.join(savePath_3, 'model_para.pkl'), 'wb') as f:
        print("The parameters setting of model is:", model_para)
        pickle.dump(model_para, f) ## bet parameters set
    f.close()

    return None


# Encoder


class TransformerEncoderModel(nn.Module):
    def __init__(self, n_features, nhead, nhid, nlayers, n_output, dropout=0.5):
        super(TransformerEncoderModel, self).__init__()
        self.model_type = 'Transformer'
        self.fc_in = nn.Linear(n_features, nhid)  # Added line

        self.pos_encoder = PositionalEncoding(nhid, dropout)
        encoder_layers = nn.TransformerEncoderLayer(nhid, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.fc_out = nn.Linear(nhid, n_output)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc_in.bias.data.zero_()  # Added line
        self.fc_in.weight.data.uniform_(-initrange, initrange)  # Added line
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.fc_in(x)  # Added line
        x = x.unsqueeze(0)  # Add sequence length dimension
        x = self.pos_encoder(x)
        embedding = self.transformer_encoder(x)
        # Remove sequence length dimension for the FC layer
        embedding = embedding.squeeze(0)
        embedding = self.fc_out(embedding)
        return embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        

class DenseNetEncoderModel(nn.Module):
    def __init__(self, n_features, nlayers, n_output, dropout=0.5, growth_rate=0.5):
        super(DenseNetEncoderModel, self).__init__()
        self.model_type = 'DenseNet'

        self.n_features = n_features
        self.growth_rate = growth_rate
        self.nlayers = nlayers
        self.n_output = n_output
        self.dropout = dropout

        # Calculate the number of output features for each dense layer based on growth rate
        dense_layer_sizes = [int(n_features + i * n_features * growth_rate) for i in range(nlayers)]

        # Create dense layers
        self.layers = nn.ModuleList()
        for i, layer_size in enumerate(dense_layer_sizes):
            if i == 0:
                self.layers.append(nn.Linear(n_features, layer_size))
            else:
                self.layers.append(nn.Linear(dense_layer_sizes[i-1], layer_size))
        
        # Final layer that takes the last dense layer size into account
        self.final_layer = nn.Linear(dense_layer_sizes[-1], n_output)

        self.activation = nn.ELU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        features = x
        for layer in self.layers:
            # Compute the current layer's output
            layer_output = layer(features)
            # Apply activation and dropout to the current layer's output
            layer_output = self.dropout_layer(self.activation(layer_output))
            # Use the current layer's output as input for the next layer
            features = layer_output

        # Compute the final embedding without activation or dropout
        embedding = self.final_layer(features)
        return embedding


class MLPEncoderModel(nn.Module):
    def __init__(self, n_features, nhid, nlayers, n_output, dropout=0.5):
        super(MLPEncoderModel, self).__init__()
        self.model_type = 'MLP'

        # Define hidden layers
        self.hidden_layers = []
        for _ in range(nlayers - 2):
            self.hidden_layers.append(nn.Linear(nhid, nhid))
            self.hidden_layers.append(nn.ELU())
            self.hidden_layers.append(nn.Dropout(dropout))

        # Define model layers
        self.layers = nn.Sequential(
            nn.Linear(n_features, nhid),
            nn.ELU(),
            nn.Dropout(dropout),
            *self.hidden_layers,
            nn.Linear(nhid, n_output)
        )

    def forward(self, x):
        embedding = self.layers(x)
        return embedding

# Risk score Predictor


class RiskscorePredictor(nn.Module):
    def __init__(self, n_features, nhid, nhout=1, dropout=0.5):
        super(RiskscorePredictor, self).__init__()
        self.RiskscoreMLP = nn.Sequential(
            nn.Linear(n_features, nhid),
            nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(nhid, nhid),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            nn.Linear(nhid, nhout),
            # nn.Linear(n_features, nhout),
        )

    def forward(self, embedding):
        risk_score = torch.sigmoid(self.RiskscoreMLP(embedding))
        return risk_score.squeeze()

# Regression score Predictor


class RegscorePredictor(nn.Module):
    def __init__(self, n_features, nhid, nhout=1, dropout=0.5):
        super(RegscorePredictor, self).__init__()
        self.RegscoreMLP = nn.Sequential(
            nn.Linear(n_features, nhid),
            nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(nhid, nhid),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            nn.Linear(nhid, nhout),
            # nn.Linear(n_features, nhout),
        )

    def forward(self, embedding):
        risk_score = self.RegscoreMLP(embedding)
        return risk_score.squeeze()

# Bionomial Predictor


class ClassscorePredictor(nn.Module):
    def __init__(self, n_features, nhid, nhout=2, dropout=0.5):
        super(ClassscorePredictor, self).__init__()
        self.ClassscoreMLP = nn.Sequential(
            nn.Linear(n_features, nhid),
            nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(nhid, nhid),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            nn.Linear(nhid, nhout),
            # nn.Linear(n_features, nhout),
        )

    def forward(self, embedding):
        proba_score = F.softmax(self.ClassscoreMLP(embedding))
        return proba_score


# Pathology Predictor


class PathologyPredictor(nn.Module):
    def __init__(self, n_features, nhid, nclass, dropout=0.5):
        super(PathologyPredictor, self).__init__()
        self.PathologyMLP = nn.Sequential(
            nn.Linear(n_features, nhid),
            nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(nhid, nhid),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            nn.Linear(nhid, nclass),
            # nn.Linear(n_features, nclass),
        )

    def forward(self, embedding):
        pathology_score = F.softmax(self.PathologyMLP(embedding))
        return pathology_score

# Main network


class TiRankModel(nn.Module):
    def __init__(self, n_features, nhead, nhid1, nhid2, nlayers, n_output, n_pred=1, n_patho=0, dropout=0.5, mode="Cox", encoder_type="MLP"):
        super(TiRankModel, self).__init__()

        # Initialize the learnable weight matrix
        self.feature_weights = nn.Parameter(torch.Tensor(n_features, 1),requires_grad=True)
        nn.init.xavier_uniform_(self.feature_weights)

        ## Encoder
        self.encoder_type = encoder_type

        if self.encoder_type == "Transformer":
            self.encoder = TransformerEncoderModel(
                n_features, nhead, nhid1, nlayers, n_output, dropout)
        elif self.encoder_type == "MLP":
            self.encoder = MLPEncoderModel(
                n_features, nhid1, nlayers, n_output, dropout)
        elif self.encoder_type == "DenseNet":
            self.encoder = DenseNetEncoderModel(
                n_features, nlayers, n_output, dropout)
        else:
            raise ValueError(f"Unsupported Encoder Type: {self.encoder_type}")

        ## Mode
        if mode == "Cox":
            self.predictor = RiskscorePredictor(
                n_output, nhid2, n_pred, dropout)

        elif mode == "Regression":
            self.predictor = RegscorePredictor(
                n_output, nhid2, n_pred, dropout)

        elif mode == "Classification":
            self.predictor = ClassscorePredictor(
                n_output, nhid2, n_pred, dropout)
                
        else:
            raise ValueError(f"Unsupported Mode: {mode}")

        self.pathologpredictor = PathologyPredictor(
            n_output, nhid2, n_patho, dropout)

    def forward(self, x):
        scaled_x = x * self.feature_weights.T
        embedding = self.encoder(scaled_x)
        risk_score = self.predictor(embedding)
        patho_pred = self.pathologpredictor(embedding)

        return embedding, risk_score, patho_pred

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # print("Perfoem xavier_uniform initiate")
            if m.bias is not None:
                m.bias.data.fill_(0.0)