# -*- coding: utf-8 -*-
# @Date : 1.19.2024
# @Author : LingLUO

from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls

controls = dbc.Card(
    [
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(dbc.Label("1. Training TiRank Model"), width=5),
                        # dbc.Col(
                        #     html.Div([html.Img(
                        #         id='help_of_training_tirank_model',
                        #         src='./assets/help.svg',
                        #         style={'height': '12px', 'margin-left': '-200%', 'margin-top': '-120%'}
                        #     ),
                        #         dbc.Tooltip(
                        #             "#TODO Information to be added ",
                        #             target="help_of_training_tirank_model",
                        #         )]),
                        #
                        #     width=1, style={'margin-right': '-20px', 'margin-top': '3%'}
                        # )
                    ],

                ),
                # dbc.Row(
                #     [
                #         dbc.Col(
                #             dbc.Label('Save Path'),
                #             style={'margin-right': '-180px', 'margin-top': '1%'}
                #         ),
                #         dbc.Col(
                #             dbc.Input(
                #                 id="upload-tirank-save-path", placeholder="", value='./data/',
                #                 disabled=True
                #
                #             ), width=5,
                #             style={'margin-right': '10px'}
                #
                #         ),
                #         dbc.Col(
                #             dbc.Button("Load Data", id='load_data_button', color="dark", className="me-1"),
                #             style={'margin-right': '10px'}
                #
                #         ),
                #         dcc.ConfirmDialog(
                #             id='confirm-load-data',
                #             message='load data successfully!',
                #         ),
                #     ]
                # ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Label('Device'),
                            style={'margin-right': '-180px', 'margin-top': '1%'}
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="device-dropdown",
                                options=[
                                    {"label": "cuda", "value": 'cuda'},
                                    {"label": "cpu", "value": 'cpu'},
                                ]
                            ),
                            width=5,
                            style={'margin-left': '-59%'}

                        ),
                    ], style={'margin-top': '1%'}
                ),
                dbc.Row([
                    dbc.Col(
                        dbc.Checklist(
                            options=[
                                {"label": "advanced", "value": 1},
                            ],
                            value=[],
                            id="advanced-train-checklist",
                        ),
                    )
                ],
                    style={'margin-top': '2%'}),
                dbc.Row(id='advanced-train-turn-row', children=[

                ]),
                dbc.Row(
                    html.Div([
                        dbc.Button(
                            "Train", id='train_id', outline=True, color="secondary", className="me-1"
                        ),
                    ],
                        className="d-grid gap-2 col-6 mx-auto"), style={'margin-top': '5%'}
                ),

            ]
        ),
        html.Hr(),
        html.Div(
            [
                dbc.Label("2. Prediction"),
                dbc.Row([
                    dbc.Col(
                        dbc.Checklist(
                            options=[
                                {"label": "Reject", "value": 1},
                            ],
                            value=[],
                            id="do-reject-checklist",
                        ),
                    )
                ],
                    style={'margin-top': '2%'}),
                dbc.Row([
                    dbc.Col(
                        dbc.Checklist(
                            options=[
                                {"label": "advanced", "value": 1},
                            ],
                            value=[],
                            id="advanced-predict-checklist",
                        ),
                    )
                ],
                    style={'margin-top': '2%'}),
                dbc.Row(id='advanced-predict-turn-row', children=[

                ]),
                html.Div(id='tolerance'),
                html.Div(id='reject_model'),
                html.Div(id='nhead'),
                html.Div(id='n_output'),
                html.Div(id='nhid2'),
                html.Div(id='nhid1'),
                html.Div(id='nlayer'),
                html.Div(id='dropout'),
                html.Div(id='n_trails'),
                dbc.Row(
                    html.Div([
                        dbc.Button(
                            "Predict", id='predict-button', outline=True, color="secondary", className="me-1"
                        ),
                    ],
                        className="d-grid gap-2 col-6 mx-auto"),
                ),

            ]
        ),
    ],
    body=True
)


def tirank_layout():
    return html.Div(dbc.Row(
        [
            dbc.Col(
                dbc.Container(
                    [
                        # html.Hr(),
                        dbc.Label("TiRank analysis"),
                        dbc.Row(
                            [
                                dbc.Col(controls, md=16),
                            ],
                            align="center",
                        ),
                    ],
                    fluid=True, style={'margin-top': '8%', 'margin-left': '1%'}
                ), md=5
            ),

            dbc.Col(
                dbc.Container(
                    [
                        # html.Hr(),
                        dbc.Label("View Results"),

                        dbc.Card(
                            [dbc.Row(
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="tirank-res-radioitems",
                                        options=[
                                            {"label": "Loss curve", "value": 'epoch-loss'},
                                            {"label": "Predicted score distribution",
                                             "value": 'tiRank_pred_score_distribution'},
                                            {"label": "UMAP of Predicted label", "value": 'UMAP_of_TiRank_Label_Score'},
                                            {"label": "UMAP of Predicted score", "value": 'UMAP of TiRank Pred Score'},
                                            {"label": "Spatial distribution of Predicted score (ST data only)",
                                             "value": 'Spatial of TiRank Pred Score'},
                                            {"label": "Spatial distribution of Predicted label (ST data only)",
                                             "value": 'Spatial of TiRank Label Score'},
                                        ],
                                    ), md=6, style={'margin-left': '1%', 'margin-top': '1%'}
                                ),

                            ),
                                dls.Hash(
                                    dbc.Row(
                                        [
                                            dbc.Col(

                                            )
                                        ],
                                        align="center", id='tirank-res-row'
                                    ),
                                    color="#333333",
                                    speed_multiplier=2,
                                    fullscreen=True
                                ), ],
                            style={'margin-top': '1%', 'width': '900px', 'height': '700px'}
                        ),
                    ],
                    fluid=True, style={'margin-top': '6%', 'margin-left': '3%'}
                )
            ),
        ],
        align="center"), style={'margin-bottom': '10%'})
