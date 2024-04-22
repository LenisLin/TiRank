import dash_bootstrap_components as dbc
import dash
from dash import dcc, html
import os
import pickle
from shutil import copy

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])


def cox(col1):
    res = []
    for item in col1:
        if 'Unnamed' not in item:
            temp_dict = dict()
            temp_dict['label'] = str(item)
            temp_dict['value'] = item
            res.append(temp_dict)
    return html.Div(id='cox',
                    children=dbc.Row(
                        [
                            dbc.Col(dbc.Label('Survival Time:'), style={'margin-top': '1%', 'margin-right': '-50px'}),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='cox-time',
                                    options=res
                                )
                            ),
                            dbc.Col(dbc.Label('Survival Status:'), style={'margin-top': '1%', 'margin-right': '-50px'}),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='cox-status',
                                    options=res
                                )
                            ),
                            html.Div(id='regression-var'),
                            html.Div(id='classification-var')
                        ])
                    )


def regression(col):
    res = []
    for item in col:
        if 'Unnamed' not in item:
            temp_dict = dict()
            temp_dict['label'] = str(item)
            temp_dict['value'] = item
            res.append(temp_dict)

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Label('Continuous Variable:'), style={'margin-top': '1%', 'margin-right': '-50px'}),
            dbc.Col(
                dcc.Dropdown(
                    id='regression-var',
                    options=res
                )
            ),
            html.Div(id='cox-time'),
            html.Div(id='cox-status'),
            html.Div(id='classification-var')
        ])
    ])


def classification(col):
    res = []
    for item in col:
        if 'Unnamed' not in item:
            temp_dict = dict()
            temp_dict['label'] = str(item)
            temp_dict['value'] = item
            res.append(temp_dict)

    return html.Div([
        dbc.Row([
            dbc.Col(dbc.Label('Binary Variable:'), style={'margin-top': '1%', 'margin-right': '-50px'}),
            dbc.Col(
                dcc.Dropdown(
                    id='classification-var',
                    options=res
                )
            ),
            html.Div(id='cox-time'),
            html.Div(id='cox-status'),
            html.Div(id='regression-var'),
        ])
    ])


def none_state():
    return html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Alert(
                    [
                        html.I(className="bi bi-exclamation-triangle-fill me-2"),
                        "You need to select a model from the Mode drop-down box.",

                    ],
                    color="warning",
                    dismissable=True,
                    className="d-flex align-items-center",
                ),
            )
        ])
    ])


def filtering_card():
    return html.Div([
        dbc.Card(
            dbc.CardBody(
                [
                    dbc.InputGroup(
                        children=[
                            dbc.InputGroupText("Min count :"),
                            dbc.Input(id='filtering_card_min_count', placeholder="Amount", type="number"),
                        ],
                        className="mb-3",
                    ),
                    dbc.InputGroup(
                        children=[
                            dbc.InputGroupText("Max count :"),
                            dbc.Input(id='filtering_card_max_count', placeholder="Amount", type="number"),
                        ],
                        className="mb-3",
                    ),
                    dbc.InputGroup(
                        children=[
                            dbc.InputGroupText("Min gene :"),
                            dbc.Input(id='filtering_card_min_gene', placeholder="Amount", type="number"),
                        ],
                        className="mb-3",
                    ),
                    dbc.InputGroup(
                        children=[
                            dbc.InputGroupText("Max MT :"),
                            dbc.Input(id='filtering_card_max_mt', placeholder="Amount", type="number"),
                        ],
                        className="mb-3",
                    ),
                ]
            ),
            style={"width": "18rem", 'margin-top': '-10%'},
        )])


def train_advanced_card():
    return [
        dbc.Row([
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Nhead :"),
                        dbc.Input(id='nhead', placeholder="Amount", type="number"),
                    ],
                    className="mb-3",
                ),

            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("n_output :"),
                        dbc.Input(id='n_output', placeholder="Amount", type="number"),
                    ],
                    className="mb-3",
                ),
            )

        ], style={'margin-top': '1%'}),
        dbc.Row([
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("nhid2 :"),
                        dbc.Input(id='nhid2', placeholder="Amount", type="number"),
                    ],
                    className="mb-3",
                ),

            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("nhid1 :"),
                        dbc.Input(id='nhid1', placeholder="Amount", type="number"),
                    ],
                    className="mb-3",
                ),
            )

        ], style={'margin-top': '1%'}),
        dbc.Row([
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("nlayer :"),
                        dbc.Input(id='nlayer', placeholder="Amount", type="number"),
                    ],
                    className="mb-3",
                ),

            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("dropout :"),
                        dbc.Input(id='dropout', placeholder="Amount", type="number"),
                    ],
                    className="mb-3",
                ),
            )

        ], style={'margin-top': '1%'}),
        dbc.Row([
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("n_trails  :"),
                        dbc.Input(id='n_trails', placeholder="Amount", type="number"),
                    ],
                    className="mb-3",
                ),

            ),
        ], style={'margin-top': '1%'})

    ]


def predict_advanced_card():
    return [
        dbc.Row([
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Tolerance :"),
                        dbc.Input(id='tolerance', placeholder="Amount", type="number"),
                    ],
                    className="mb-3",
                ),

            ),
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Reject_mode :"),
                        dcc.Dropdown(id='reject_model', options=[
                            {"label": "GMM", "value": 'GMM'},
                            {"label": "Strict", "value": 'Strict'},
                        ], style={'width': '150px'}),
                    ],
                    className="mb-3",
                ),
            )

        ], style={'margin-top': '1%'}),
    ]


def qc_violins_layout():
    if os.path.exists('./assets/qc_violins.png'):
        return dbc.Col(html.Img(src='./assets/qc_violins.png',
                                style={'width': '700px', 'height': '400px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))
    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '700px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


def leiden_cluster_layout():
    if os.path.exists('./assets/leiden cluster.png'):
        return dbc.Col(html.Img(src='./assets/leiden cluster.png',
                                style={'width': '700px', 'height': '300px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))
    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '700px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


def patho_label_layout():
    if os.path.exists('./assets/patho_label.png'):
        return dbc.Col(html.Img(src='./assets/patho_label.png',
                                style={'width': '700px', 'height': '700px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))
    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '700px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


def tirank_pred_score_distribution():
    if os.path.exists('./assets/TiRank Pred Score Distribution.png'):
        return dbc.Col(html.Img(src='./assets/TiRank Pred Score Distribution.png',
                                style={'width': '700px', 'height': '600px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))

    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '300px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


def umap_of_tirank_label_score():
    if os.path.exists('./assets/UMAP of TiRank Label Score.png'):
        return dbc.Col(html.Img(src='./assets/UMAP of TiRank Label Score.png', ))

    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '300px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


def umap_of_tirank_pred_score():
    if os.path.exists('./assets/UMAP of TiRank Pred Score.png'):
        return dbc.Col(html.Img(src='./assets/UMAP of TiRank Pred Score.png',
                                style={'width': '700px', 'height': '600px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))

    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '300px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


def spatial_of_tirank_pred_score():
    if os.path.exists('./assets/Spatial of TiRank Pred Score.png'):
        return dbc.Col(html.Img(src='./assets/Spatial of TiRank Pred Score.png',
                                style={'width': '700px', 'height': '600px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))

    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '300px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


def spatial_of_tirank_label_score():
    if os.path.exists('./assets/Spatial of TiRank Label Score.png'):
        return dbc.Col(html.Img(src='./assets/Spatial of TiRank Label Score.png',
                                style={'width': '700px', 'height': '600px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))

    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '300px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


def distribution_of_tirank_label_in_leiden_clusters_group():
    if os.path.exists('./assets/Distribution of TiRank label in leiden_clusters.png'):
        return dbc.Col(html.Img(src='./assets/Distribution of TiRank label in leiden_clusters.png',
                                style={'width': '700px', 'height': '600px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))

    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '300px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


def deg_volcano_plot():
    if os.path.exists('./assets/DEG_volcano_plot.png'):
        return dbc.Col(html.Img(src='./assets/DEG_volcano_plot.png',
                                style={'width': '500px', 'height': '600px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))
    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '600px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


def train_epoch_loss_plot():
    f = open('./data/best_params.pkl', 'rb')
    param = pickle.load(f)
    f.close()
    png_path = './data/checkpoints/'
    png_path_list = os.listdir(png_path)
    param_seq = 'lr_' + str(param['lr']) + '_epochs_' + str(param['n_epochs']) + '_alpha0_' + str(
        param['alpha_0']) + '_alpha1_' + str(param['alpha_1']) + '_alpha2_' + str(param['alpha_2']) + '_alpha3_' + str(
        param['alpha_3'])
    for item in png_path_list:
        if param_seq in item and item[-3:] == 'png':
            from_path = png_path + item
            to_path = './assets/' + item
            copy(from_path, to_path)

    return dbc.Col(html.Img(src=to_path,
                            style={'width': '800px', 'height': '600px', 'margin': 'auto',
                                   'display': 'flex', 'align-items': 'center'}))
