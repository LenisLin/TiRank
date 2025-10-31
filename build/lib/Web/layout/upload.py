# -*- coding: utf-8 -*-
# @Date : 1.15.2024
# @Author : LingLUO

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import dash_loading_spinners as dls

controls = dbc.Card(
    [
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(dbc.Label("Load Example Data"), width=4),
                        dbc.Col(
                            html.Div([html.Img(
                                id='help_of_load_example_data',
                                src='./assets/help.svg',
                                style={'height': '12px', 'margin-left': '-200%'}
                            ),
                                dbc.Tooltip(
                                    """In order to facilitate the operation of TiRank-Web, 
                                    we provide sample data for your operation. Select "load spatial data" 
                                    or "load single-cell Data" below to load our sample data for you.""",
                                    target="help_of_load_example_data",
                                )]),

                            width=1
                        )
                    ],

                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div([
                                # dcc.Upload(
                                #     id='upload-data',
                                #     children=html.Div([
                                #         'Drag and Drop or ',
                                #         html.A('Select Files')
                                #     ]),
                                #     style={
                                #         'width': '100%',
                                #         'height': '60px',
                                #         'lineHeight': '60px',
                                #         'borderWidth': '1px',
                                #         'borderStyle': 'dashed',
                                #         'borderRadius': '5px',
                                #         'textAlign': 'center',
                                #         'margin-top': '10px',
                                #         'margin-bottom': '5px'
                                #     },
                                #     # Allow multiple files to be uploaded
                                #     multiple=True
                                # ),
                                # html.Div(id='output-data-upload'),
                                dbc.Checklist(
                                    options=[
                                        {"label": "Spatial transcriptome Data", "value": 1},
                                    ],
                                    value=[],
                                    id="load-example-switch-st",
                                    switch=True,
                                ),

                            ])
                        ),
                        dbc.Col(
                            html.Div([
                                # dcc.Upload(
                                #     id='upload-data',
                                #     children=html.Div([
                                #         'Drag and Drop or ',
                                #         html.A('Select Files')
                                #     ]),
                                #     style={
                                #         'width': '100%',
                                #         'height': '60px',
                                #         'lineHeight': '60px',
                                #         'borderWidth': '1px',
                                #         'borderStyle': 'dashed',
                                #         'borderRadius': '5px',
                                #         'textAlign': 'center',
                                #         'margin-top': '10px',
                                #         'margin-bottom': '5px'
                                #     },
                                #     # Allow multiple files to be uploaded
                                #     multiple=True
                                # ),
                                # html.Div(id='output-data-upload'),
                                dbc.Checklist(
                                    options=[
                                        {"label": "scRNA-seq Data", "value": 1},
                                    ],
                                    value=[],
                                    id="load-example-switch-sc",
                                    switch=True,
                                ),

                            ])
                        ),

                    ]
                )

            ]
        ),
        html.Hr(),
        html.Div(
            [
                dbc.Label("1. Bulk Transcription Data"),
                dbc.Row(
                    [
                        dbc.Label("Expression Matrix", html_for="expression-matrix", width=5),
                        dbc.Col(
                            dbc.Input(
                                type="email", id="expression-matrix", placeholder=""
                            ), width=8

                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    id='view_for_em',
                                    color="light",
                                    children=html.Img(
                                        src='./assets/view.svg',
                                        style={'height': '15px'}
                                    )),
                                dbc.Tooltip(
                                    "View",
                                    target="view_for_em",
                                )
                            ]),
                            width=1
                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    id='download_for_em',
                                    color="light",
                                    children=html.Img(
                                        src='./assets/download.svg',
                                        style={'height': '15px'}
                                    )),
                                dcc.Download(id="download-em"),
                                dbc.Tooltip(
                                    "Download",
                                    target="download_for_em",
                                )
                            ]),
                            width=1
                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    id='help_for_em',
                                    color="light",
                                    children=html.Img(
                                        src='./assets/help.svg',
                                        style={'height': '15px'}
                                    )),
                                dcc.Download(id='download-em'),
                                dbc.Tooltip(
                                    "Upload the Expression Matrix file.",
                                    target="help_for_em",
                                )
                            ]),
                            width=1
                        ),
                    ],
                    className="mb-3",
                ),
                dbc.Row(
                    [
                        dbc.Label("Clinical Information", html_for="clinical-info", width=5),
                        dbc.Col(
                            dbc.Input(
                                type="email", id="clinical-info", placeholder=""
                            ), width=8

                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    id='view_for_ci',
                                    color="light",
                                    children=html.Img(
                                        src='./assets/view.svg',
                                        style={'height': '15px'}
                                    )),
                                dbc.Tooltip(
                                    "View",
                                    target="view_for_ci",
                                )
                            ]),
                            width=1
                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    id="download_for_ci",
                                    color="light",
                                    children=html.Img(
                                        src='./assets/download.svg',
                                        style={'height': '15px'}
                                    )),
                                dcc.Download(id='download-ci'),
                                dbc.Tooltip(
                                    "Download",
                                    target="download_for_ci",
                                )
                            ]),
                            width=1
                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    id='help_for_ci',
                                    color="light",
                                    children=html.Img(
                                        src='./assets/help.svg',
                                        style={'height': '15px'}
                                    )),
                                dbc.Tooltip(
                                    "For Bulk clinical information files, in COX mode, "
                                    "the row name is sample name, the first column is time, and the second "
                                    "column is status. For Regression and Classifiation, the row name is sample "
                                    "name, and only one column is needed for the value corresponding to clinical "
                                    "characteristics. For Bulk clinical information files, in COX mode, the row name"
                                    " is sample name, with the first column time and the second column status."
                                    " For Regression and Classifiation, the row name is sample name and only one "
                                    "column of values corresponding to clinical features is required",
                                    target="help_for_ci",
                                )
                            ]),
                            width=1
                        ),
                    ],
                    className="mb-3",
                ),
                # dcc.ConfirmDialog(
                #     id='confirm-danger-btd',
                #     message='The rownames of clinical information was not match with expression profile !',
                # ),
                # dcc.ConfirmDialog(
                #     id='confirm-done-btd',
                #     message='You successfully passed the check!',
                # ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Warning"), close_button=True),
                        dbc.ModalBody("The rownames of clinical information was not match with expression profile !"),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="confirm-danger-btd-close",
                                className="ms-auto",
                                n_clicks=0,
                            )
                        ),
                    ],
                    id="confirm-danger-btd",
                    centered=True,
                    is_open=False,
                ),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Warning"), close_button=True),
                        dbc.ModalBody("You successfully passed the check!"),
                        dbc.ModalFooter(
                            dbc.Button(
                                "Close",
                                id="confirm-done-btd-close",
                                className="ms-auto",
                                n_clicks=0,
                            )
                        ),
                    ],
                    id="confirm-done-btd",
                    centered=True,
                    is_open=False,
                ),
                dbc.Row(
                    html.Div([
                        dbc.Button(
                            "Check", id='check_bulk', outline=True, color="secondary", className="me-1"
                        ),
                    ],
                        className="d-grid gap-2 col-6 mx-auto"),
                ),

            ]
        ),
        html.Hr(),
        dbc.Row(
            html.Div(
                [
                    dbc.Label("2-1. Spatial transcriptome Data"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(
                                type="email", id="st-info", placeholder=""
                            ),
                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    id="view_for_st",
                                    color="light",
                                    children=html.Img(
                                        src='./assets/view.svg',
                                        style={'height': '15px'}
                                    ),
                                ),

                                dbc.Tooltip(
                                    "View",
                                    target="view_for_st",
                                )
                            ]),
                            width=1
                        ),
                        # dbc.Col(
                        #     html.Div([
                        #         dbc.Button(
                        #             id='download_for_st',
                        #             color='light',
                        #             children=html.Img(
                        #                 src='./assets/download.svg',
                        #                 style={'height': '15px'}
                        #             )
                        #         ),
                        #         dcc.Download(id='download-st'),
                        #         dbc.Tooltip(
                        #             "Download",
                        #             target="download_for_st",
                        #         )
                        #     ]),
                        #     width=1
                        # ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    id='help_for_st',
                                    color='light',
                                    children=html.Img(
                                        src='./assets/help.svg',
                                        style={'height': '15px'}
                                    ),
                                ),
                                dbc.Tooltip(
                                    "Upload Spatial transcriptome Data",
                                    target="help_for_st",
                                )
                            ]),
                            width=1
                        ),
                    ])
                ]
            ), className="mb-3", ),
        dbc.Row(
            html.Div([
                dbc.Button(
                    "Confirm", id='confirm_st', outline=True, color="secondary", className="me-1"
                ),
            ],
                className="d-grid gap-2 col-6 mx-auto"),
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Warning"), close_button=True),
                dbc.ModalBody("You have selected ST data!"),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="confirm-st-btd-close",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="confirm-st-btd",
            centered=True,
            is_open=False,
        ),
        html.Hr(),
        dbc.Row(
            html.Div(
                [
                    dbc.Label("2-2. scRNA-seq Data"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(
                                type="email", id="sc-info", placeholder=""
                            ),
                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    id="view_for_sc",
                                    color="light",
                                    children=html.Img(
                                        src='./assets/view.svg',
                                        style={'height': '15px'}
                                    ),
                                ),

                                dbc.Tooltip(
                                    "View",
                                    target="view_for_sc",
                                )
                            ]),
                            width=1
                        ),
                        # dbc.Col(
                        #     html.Div([
                        #         dbc.Button(
                        #             id='download_for_sc',
                        #             color='light',
                        #             children=html.Img(
                        #                 src='./assets/download.svg',
                        #                 style={'height': '15px'}
                        #             )
                        #         ),
                        #         dcc.Download(id='download-sc'),
                        #         dbc.Tooltip(
                        #             "Download",
                        #             target="download_for_sc",
                        #         )
                        #     ]),
                        #     width=1
                        # ),
                        dbc.Col(
                            html.Div([
                                dbc.Button(
                                    id='help_for_sc',
                                    color='light',
                                    children=html.Img(
                                        src='./assets/help.svg',
                                        style={'height': '15px'}
                                    ),
                                ),
                                dbc.Tooltip(
                                    "Upload scRNA-seq Data",
                                    target="help_for_sc",
                                )
                            ]),
                            width=1
                        ),
                    ])
                ]
            ), className="mb-3", ),
        dbc.Row(
            html.Div([
                dbc.Button(
                    "Confirm", id='confirm_sc', outline=True, color="secondary", className="me-1"
                ),
            ],
                className="d-grid gap-2 col-6 mx-auto"),
        ),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Warning"), close_button=True),
                dbc.ModalBody("You have selected SC data!"),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close",
                        id="confirm-sc-btd-close",
                        className="ms-auto",
                        n_clicks=0,
                    )
                ),
            ],
            id="confirm-sc-btd",
            centered=True,
            is_open=False,
        ),
    ],
    body=True,
)


def upload_layout():
    return html.Div(dbc.Row(
        [
            dbc.Col(
                dbc.Container(
                    [
                        # html.Hr(),
                        dbc.Label("Input"),
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
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Warning"), close_button=True),
                    dbc.ModalBody("Check that you entered the correct absolute path to the file on the left!"),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close",
                            id="close-centered",
                            className="ms-auto",
                            n_clicks=0,
                        )
                    ),
                ],
                id="modal-centered",
                centered=True,
                is_open=False,
            ),
            dbc.Col(
                dbc.Container(
                    [
                        # html.Hr(),
                        dbc.Label("View Data"),
                        dbc.Card(
                            dbc.Row(
                                [
                                    dls.Hash(
                                        dbc.Col(dash_table.DataTable(id='upload_dataframe_view', page_size=15,
                                                                     style_table={'overflowX': 'auto'})),
                                        color="#333333",
                                        speed_multiplier=2,
                                        size=80,
                                    ),
                                    # dbc.Col(dash_table.DataTable(data=view_data.to_dict('records'), page_size=15,
                                    #                              style_table={'overflowX': 'auto'}))
                                ],
                                align="center"
                            ),
                            style={'width': '900px', 'height': '700px', 'display': 'flex', 'justify-content': 'center',
                                   'align-items': 'center'}  # 设置宽度和高度
                        ),
                    ],
                    fluid=True, style={'margin-top': '5%'}
                )
            ),
        ],
        align="center"), style={'margin-bottom': '8%'})
