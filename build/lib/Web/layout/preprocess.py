# -*- coding: utf-8 -*-
# @Date : 1.15.2024
# @Author : LingLUO

from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls

controls = dbc.Card(
    [
        html.Div(
            [
                # dbc.Row(
                #     [
                #         dbc.Row([
                #             dbc.Col(dbc.Label("1. Selecting Data"), width=4),
                #             dbc.Col(
                #                 html.Div([
                #                     html.Img(id='help_for_sd', src='./assets/help.svg', height='15px'),
                #                     dbc.Tooltip(
                #                         "#TODO Information to be added ",
                #                         target="help_for_sd",
                #                     )
                #                 ]), style={'margin-left': '-70px'}, md=1
                #
                #             )]
                #         ),
                #         dbc.Row(
                #             [
                #                 dbc.Col(
                #                     dbc.Label("Bulk", html_for="bulk", width=6),
                #                 ),
                #                 dbc.Col(
                #                     dbc.Input(
                #                         type="email", id="expression-matrix", placeholder=""
                #                     ),
                #                     width=8,
                #                 ),
                #                 dbc.Col(
                #                     html.Div([
                #                         html.Img(
                #                             id='view_for_em',
                #                             src='./assets/view.svg',
                #                             style={'height': '15px'}
                #                         ),
                #                         dbc.Tooltip(
                #                             "View",
                #                             target="view_for_em",
                #                         )
                #                     ]),
                #                     width=1
                #                 ),
                #                 dbc.Col(
                #                     html.Div([
                #                         html.Img(
                #                             id='help_for_bulk',
                #                             src='./assets/help.svg',
                #                             style={'height': '15px'}
                #                         ),
                #                         dbc.Tooltip(
                #                             "#TODO Information to be added ",
                #                             target="help_for_bulk",
                #                         )
                #                     ]),
                #                     width=1
                #                 ),
                #             ],
                #             className="mb-3", style={'margin-top': '1%'}
                #         ),
                #         dbc.Row(
                #             [
                #                 dbc.Col(
                #                     dbc.Label("SC/ST", html_for="sc/st", width=8),
                #                 ),
                #                 dbc.Col(
                #                     dbc.Input(
                #                         type="email", id="clinical-info", placeholder=""
                #                     ),
                #                     width=8,
                #                 ),
                #                 dbc.Col(
                #                     html.Div([
                #                         html.Img(
                #                             id='view_for_scst',
                #                             src='./assets/view.svg',
                #                             style={'height': '15px'}
                #                         ),
                #                         dbc.Tooltip(
                #                             "View",
                #                             target="view_for_scst",
                #                         )
                #                     ]),
                #                     width=1
                #                 ),
                #                 dbc.Col(
                #                     html.Div([
                #                         html.Img(
                #                             id='help_for_scst',
                #                             src='./assets/help.svg',
                #                             style={'height': '15px'}
                #                         ),
                #                         dbc.Tooltip(
                #                             "#TODO Information to be added ",
                #                             target="help_for_scst",
                #                         )
                #                     ]),
                #                     width=1
                #                 ),
                #             ],
                #             className="mb-3",
                #         ),
                #     ],
                #
                # ),
            ]
        ),
        # html.Hr(),
        html.Div(
            [
                dbc.Row([
                    dbc.Col(dbc.Label("1. Pre-processing Data")),
                    # dbc.Col(
                    #     html.Div([
                    #         html.Img(id='help_for_ppd', src='./assets/help.svg', height='15px'),
                    #         dbc.Tooltip(
                    #             "#TODO Information to be added ",
                    #             target="help_for_ppd",
                    #         )
                    #     ]), style={'margin-left': '-280px'}
                    # )
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Row(
                            [
                                dbc.Col([
                                    dbc.Label("Normalization", html_for="example-radios-row"),
                                    dbc.Col(
                                        dbc.RadioItems(
                                            id="normalization-radios-row",
                                            options=[
                                                {"label": "Enable", "value": 1},
                                                {"label": "Bypass", "value": 2},
                                            ],
                                        ),

                                    ), ]
                                ),
                                dbc.Col([
                                    dbc.Label("Log-transformation", html_for="example-radios-row"),
                                    dbc.Col(
                                        dbc.RadioItems(
                                            id="log-radios-row",
                                            options=[
                                                {"label": "Enable", "value": 1},
                                                {"label": "Bypass", "value": 2},
                                            ],
                                        ),
                                    ), ]
                                )
                            ],
                            className="mb-3",
                        ),
                        # dbc.Row(
                        #     [
                        #         dbc.Col([
                        #             dbc.Label("Clustering", html_for="example-radios-row"),
                        #             dbc.Col(
                        #                 dbc.RadioItems(
                        #                     id="clustering-radios-row",
                        #                     options=[
                        #                         {"label": "Enable", "value": 1},
                        #                         {"label": "Bypass", "value": 2},
                        #                     ],
                        #                 ),
                        #
                        #             ), ]
                        #         ),
                        #         dbc.Col([
                        #             dbc.Label("Histo Clustering", html_for="example-radios-row"),
                        #             dbc.Col(
                        #                 dbc.RadioItems(
                        #                     id="histo-clustering-radios-row",
                        #                     options=[
                        #                         {"label": "Enable", "value": 1},
                        #                         {"label": "Bypass", "value": 2},
                        #                     ],
                        #                 ),
                        #             ), ]
                        #         )
                        #     ],
                        #     className="mb-3",
                        # ),
                        dbc.Row(
                            html.Div([

                                dbc.Button(
                                    "Perform", id='preprocess-perform', outline=True, color="secondary",
                                    className="me-1"
                                ),

                            ],
                                className="d-grid gap-2 col-6 mx-auto"), style={'margin-top': '3%'}
                        ),

                    ]),
                ]),
            ]
        ),
        html.Hr(),
        dbc.Row(
            html.Div(
                [
                    dbc.Label("2. Mode Select"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Label('Mode'), width=1,
                            style={'margin-top': '1%', 'margin-left': '1%'}
                        ),
                        dbc.Col(
                            # html.Div([
                            #     html.Img(
                            #         id='help_for_gp_trans',
                            #         src='./assets/help.svg',
                            #         style={'height': '15px'}
                            #     ),
                            #     dbc.Tooltip(
                            #         "#TODO Information to be added ",
                            #         target="help_for_gp_trans",
                            #     )
                            # ]),
                            width=1,
                            style={'margin-right': '-10px'}
                        ),
                        dbc.Col(dcc.Dropdown(
                            id="mode-dropdown",
                            options=[
                                {"label": "Cox", "value": 1},
                                {"label": "Regression", "value": 2},
                                {"label": "Classification", "value": 3},
                            ])
                        ),
                        dls.Hash(
                            html.Div(id='cox-time'),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                            fullscreen=True,
                        ),
                        dls.Hash(
                            html.Div(id='cox-status'),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                            fullscreen=True,
                        ),
                        dls.Hash(
                            html.Div(id='regression-var'),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                            fullscreen=True,
                        ),
                        dls.Hash(
                            html.Div(id='classification-var'),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                            fullscreen=True,
                        ),
                    ]),
                    dbc.Container(id='selected-mode', style={'margin-top': '2%'})
                ]
            ), className="mb-3", ),

        html.Hr(),
        html.Div(
            [
                dbc.Row([
                    dbc.Col(dbc.Label("3. GenePair Transformation")),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Label('Top var genes')),
                    dbc.Col(dcc.Dropdown(
                        id="tvg-dropdown",
                        options=[
                            {"label": "2000(Recommend)", "value": 2000},
                            {"label": "1000", "value": 1000},
                            {"label": "500", "value": 500},
                        ])
                    ),
                    dbc.Col(dbc.Label('P value threshold')),
                    dbc.Col(dcc.Dropdown(
                        id="pvt-dropdown",
                        options=[
                            {"label": "0.05(Recommend)", "value": 0.05},
                            {"label": "0.01", "value": 0.01},
                        ])
                    ),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Label('Top gene pairs')),
                    dbc.Col(dcc.Dropdown(
                        id="tgp-dropdown",
                        options=[
                            {"label": "2000(Recommend)", "value": 2000},
                            {"label": "5000", "value": 5000},
                            {"label": "10000", "value": 10000},
                        ]), style={'margin-left': '-75%'}, md=4
                    ),
                    # dbc.Col(dbc.Label('adj_p_value_threshold')),
                    # dbc.Col(dcc.Dropdown(
                    #     id="adj-dropdown",
                    #     options=[
                    #         {"label": "None", "value": 'None'},
                    #         {"label": "0.05", "value": 0.05},
                    #     ])
                    # ),
                ], style={'margin-top': '3%'}),
                dbc.Row(
                    html.Div([
                        dbc.Button(
                            "Perform", id='datasplit-perform', outline=True, color="secondary", className="me-1"
                        ),
                    ],
                        className="d-grid gap-2 col-6 mx-auto"), style={'margin-top': '3%'}
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Warning"), close_button=True),
                        dbc.ModalBody(
                            "A set of genes is empty. Try increasing the 'top_var_genes' value or loosening "
                            "the 'p.value' threshold."
                        ),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="confirm-gene-empty-danger-btd-close",
                                className="ms-auto",
                                n_clicks=0,
                            )
                        ),
                    ],
                    id="confirm-gene-empty-danger-btd",
                    centered=True,
                    is_open=False,
                ),
                # dbc.Row(
                #     html.Div([
                #         dbc.Button(
                #             "Perform", id='genepair-perform', outline=True, color="secondary", className="me-1"
                #         ),
                #     ],
                #         className="d-grid gap-2 col-6 mx-auto"), style={'margin-top': '3%'}
                # ),

            ]
        ),
    ],
    body=True,
)


def preprocess_layout():
    return html.Div(
        dbc.Row(
            [
                dbc.Col(
                    dbc.Container(
                        [
                            # html.Hr(),
                            dbc.Label("GP extractor"),
                            dbc.Row(
                                [
                                    dbc.Col(controls, md=16),
                                ],
                                align="center",
                            ),
                        ],
                        fluid=True, style={'margin-top': '13%', 'margin-bottom': '15%', 'margin-left': '2%'}
                    ), md=5
                ),
                dbc.Col(
                    dbc.Container(
                        [
                            # html.Hr(),
                            dbc.Row([
                                dbc.Col(dbc.Label("View Results"), md=3),
                            ]),
                            dbc.Row(
                                html.Div(
                                    [
                                        dbc.RadioItems(
                                            options=[

                                            ],
                                            value=1,
                                            id="preprocessing-res-radioitems",
                                            inline=True,
                                        ),
                                    ]
                                )
                            ),
                            dbc.Row(
                                html.Div(
                                    [
                                        dbc.RadioItems(
                                            options=[
                                                {"label": "Bulk gene pair heatmap",
                                                 "value": 'bulk_gene_pair_heatmap',
                                                 "disabled": False},
                                                {"label": "SC gene pair heatmap",
                                                 "value": 'sc_gene_pair_heatmap',
                                                 "disabled": False},
                                            ],
                                            value=1,
                                            id="preprocess-res-radioitems",
                                            inline=True,
                                        ),
                                    ]
                                )
                            ),
                            dbc.Card(
                                dls.Hash(
                                    dbc.Row(
                                        [
                                            dbc.Col(

                                            )
                                        ], align="center", id='preprocessing-res-col',
                                    ),
                                    color="#333333",
                                    speed_multiplier=2,
                                    fullscreen=True
                                ),
                                style={
                                    'margin-top': '1%',
                                    'width': '900px',
                                    'height': '700px',
                                    'display': 'flex',  # Make the card a flex container
                                    'align-items': 'center',  # Center children vertically
                                    'justify-content': 'center'  # Center children horizontally
                                }
                            ),

                        ],
                        fluid=True, style={'margin-top': '3%', 'margin-bottom': '3%'}
                    )
                ),
            ],
            align="center", style={'margin-bottom': '5%'})
    )
