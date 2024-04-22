# -*- coding: utf-8 -*-
# @Date : 1.15.2024
# @Author : LingLUO

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls

controls = dbc.Card(
    [
        # html.Div(
        #     [
        #         dbc.Row(
        #             [
        #                 dbc.Row([
        #                     dbc.Col(dbc.Label("Upload data"), width=4),
        #                     dbc.Col(
        #                         html.Div([
        #                             html.Img(id='help_for_cc', src='./assets/help.svg', height='15px'),
        #                             dbc.Tooltip(
        #                                 "#TODO Information to be added ",
        #                                 target="help_for_cc",
        #                             )
        #                         ]), style={'margin-left': '-17%'}, md=1
        #
        #                     )]
        #                 ),
        #                 dbc.Row(
        #                     [
        #                         dbc.Col(
        #                             dbc.Label("Category Column", html_for="cc", style={'margin-top': '1%'}),
        #                         ),
        #                         dbc.Col(
        #                             dbc.Input(
        #                                 type="email", id="category_column", placeholder=""
        #                             ), style={'margin-left': '-50%'}
        #                         ),
        #
        #                     ],
        #                     className="mb-3", style={'margin-top': '1%'}
        #                 ),
        #                 dbc.Row(
        #                     [
        #                         dbc.Col(
        #                             dbc.Label("Interesting Class", html_for="ic"),
        #                         ),
        #                         dbc.Col(
        #                             dbc.Input(
        #                                 type="email", id="interesting_class", placeholder=""
        #                             ), style={'margin-left': '-50%'}
        #                         ),
        #
        #                     ],
        #                     className="mb-3",
        #                 ),
        #                 dbc.Row(
        #                     html.Div([
        #                         dbc.Button(
        #                             "Merge", outline=True, color="secondary", className="me-1"
        #                         ),
        #                     ],
        #                         className="d-grid gap-2 col-6 mx-auto"), style={'margin-top': '3%'}
        #                 ),
        #             ],
        #
        #         ),
        #     ]
        # ),
        # html.Hr(),
        html.Div(
            [
                dbc.Row([
                    dbc.Col(dbc.Label("Differentially expressed genes")),
                    dbc.Col(
                        html.Div([
                            html.Img(id='help_for_deg', src='./assets/help.svg', height='15px'),
                            dbc.Tooltip(
                                "Select the threshold for defining differential expressing "
                                "genes between TiRank+ cells and TiRank- cells.",
                                target="help_for_deg",
                            )
                        ]), style={'margin-left': '-63%'}, md=1

                    )

                ]),

                dbc.Row([
                    dbc.Col(
                        dbc.Label('logFC threshold'), width=3,
                        style={'margin-top': '1%'}
                    ),
                    dbc.Col(dcc.Dropdown(
                        id="logfc-dropdown",
                        options=[
                            {"label": "1", "value": '1'},
                            {"label": "0.1", "value": '0.1'},
                            {"label": "0.05", "value": '0.05'},
                            {"label": "0.01", "value": '0.01'},
                        ]), style={'margin-left': '-3%'}
                    ),
                ], style={'margin-top': '1%'}),

                dbc.Row([
                    dbc.Col(
                        dbc.Label('P-value threshold'), width=3,
                        style={'margin-top': '1%'}
                    ),
                    dbc.Col(dcc.Dropdown(
                        id="pvalue-dropdown",
                        options=[
                            {"label": "1", "value": '1'},
                            {"label": "0.1", "value": '0.1'},
                            {"label": "0.05", "value": '0.05'},
                            {"label": "0.01", "value": '0.01'},
                        ]), style={'margin-left': '-3%'}
                    ),
                ], style={'margin-top': '1%'}),

                dbc.Row([
                    dbc.Col(
                        html.Div([
                            dbc.Button(
                                "Perform", id='deg-plot', outline=True, color="secondary", className="me-1"
                            ),
                        ], className="d-grid gap-2 col-6 mx-auto")
                    ),

                ], style={'margin-top': '3%'}
                ),

            ]
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Label("Pathway enrichment analysis"),
                ),
                dbc.Col(
                    html.Div([
                        html.Img(id='help_for_ea', src='./assets/help.svg', height='15px'),
                        dbc.Tooltip(
                            "Enrichment of differential expressing genes in selected database.",
                            target="help_for_ea",
                        )
                    ]), style={'margin-left': '-65%'}, md=1
                )
            ]
            , className="mb-3", ),
        dbc.Row(
            html.Div([
                dbc.Button(
                    "Perform", id='enrichment-run', outline=True, color="secondary", className="me-1"
                ),
            ],
                className="d-grid gap-2 col-6 mx-auto"),
        ),
    ],
    body=True,
)


def degpe_layout():
    return html.Div(
        dbc.Row(
            [
                dbc.Col(
                    dbc.Container(
                        [
                            # html.Hr(),
                            dbc.Label("Downstream analysis"),
                            dbc.Row(
                                [
                                    dbc.Col(controls, md=16),
                                ],
                                align="center",
                            ),
                        ],
                        fluid=True, style={'margin-top': '5%', 'margin-bottom': '15%', 'margin-left': '2%'}
                    ), md=5, style={'margin-top': '4%'}
                ),
                dbc.Col(
                    dbc.Container(
                        [
                            # html.Hr(),
                            dbc.Row([
                                dbc.Col(dbc.Label("View Results Here"), md=3),
                                dbc.Col(
                                    html.Div([
                                        html.Img(id='download_for_res_degpe', src='./assets/download.svg', height='15px'),
                                        dcc.Download(id='download-res-degpe'),
                                        dbc.Tooltip(
                                            "Download",
                                            target="download_for_res_degpe",
                                        )
                                    ]), style={'margin-bottom': '10px', 'margin-left': '-90px'}, md=1

                                )
                            ]),
                            dbc.Card(
                                dbc.Row(
                                    [
                                        dls.Hash(
                                            dbc.Col(
                                                dash_table.DataTable(id='degpe_dataframe_view', page_size=15,
                                                                     style_table={'overflowX': 'auto'})
                                            ),
                                            color="#333333",
                                            speed_multiplier=2,
                                            size=80,
                                        ),
                                    ],
                                    align="center", id='depge-row'
                                ),
                                style={'margin-top': '1%', 'width': '900px', 'height': '700px', 'display': 'flex',
                                       'justify-content': 'center',
                                       'align-items': 'center'}

                            ),
                        ],
                        fluid=True, style={'margin-top': '5%', 'margin-bottom': '10%'}
                    )
                ),
            ],
            align="center", )
    )
