# -*- coding: utf-8 -*-
# @Date : 1.19.2024
# @Author : LingLUO
import torch.cuda
from dash import html, dcc
import dash_bootstrap_components as dbc


def tutorial_layout():
    return html.Div(children=[
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4('Tutorial', className='text-center',
                            style={'font-family': 'Georgia, serif', 'margin-bottom': '20px'}),
                    dbc.Row([
                        dbc.Col(html.P([
                            "A introduction of TiRank Web"
                        ], style={'text-align': 'justify'}), width=12),
                    ], className='mb-4'),
                    dbc.Row([
                        html.P([
                            """
                            The development of the TiRank-web application leverages the Python Dash framework (
                            """,
                            html.A('https://dash.plotly.com', href='https://dash.plotly.com/', target='_blank'),
                            """
                            ) a product of Plotly designed for the creation and deployment 
                            of data-driven applications with custom interfaces. TiRank-web is structured into six key 
                            sections: Homepage, Upload Data, Pre-processing, Analysis, Tutorial, and FAQs, aimed at 
                            enhancing user engagement and efficiency. In the Upload Data section, users can upload their 
                            datasets, which TiRank then organizes into designated folders for temporary storage 
                            and figure generation. The Pre-processing phase involves comprehensive data cleaning,
                             normalization, and partitioning into training and validation subsets to enable 
                             hyperparameter optimization. This phase ends in the construction of a binary 
                             gene pair matrix, setting the stage for the subsequent Analysis phase. 
                             Here, a neural network model is employed to compute the 'TiRank Score', 
                             integrating a rejection mechanism for filtering out entities with low confidence. 
                             This phase also encompasses tools for differential gene expression analysis and 
                             pathway enrichment, thereby supporting biomarker discovery and further scientific
                              investigation. The Tutorial and FAQs segments provide extensive guidance and address 
                              common questions, thus ensuring effective exploration of TiRank-web's capabilities.
                            """
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col(html.Hr(), width=12)
                    ]),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '1. UploadData'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '1.1 Load Example Data'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([
                                html.P(
                                    """
                                    To facilitate a clearer understanding of this system's utility, we provide
                                     exemplary datasets for both Spatial Transcriptomics (ST) and Single-cell RNA
                                      sequencing (scRNA-seq). In the accompanying figure on the right, users are 
                                      given the option to select and load sample datasets pertaining to either 
                                      'Spatial Transcriptomics Data' or 'scRNA-seq Data'. 
                                    """
                                ),
                                html.P(
                                    """
                                    Upon selection of either the 'Spatial Transcriptomics Data' or 'scRNA-seq Data' 
                                    button, The Expression Matrix and Clinical Information in the Bulk Transcription
                                     Data below will automatically fill the path of our sample data. Moreover, 
                                     choosing 'Spatial Transcriptomics Data' will result in the auto-filling of 
                                     the form labeled 2-1 with our Spatial Transcriptomics (ST) sample data.  
                                     Conversely, opting for 'scRNA-seq data' will auto-complete the form labeled 2-2 
                                     with our Single-cell RNA sequencing (scRNA-seq) sample data.
                                    """
                                ),
                                html.P([
                                    html.B(
                                        """
                                        (Hint: Only fill out one of the 2-1 form and the 2-2 form)
                                        """
                                    )
                                ],

                                )
                            ]),
                            dbc.Col([
                                html.Img(src='./assets/load-data-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '-5%'
                                         }),
                            ])
                        ]
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '1.2 Bulk Transcription Data'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([
                                html.P("""
                                Next, we will use the loaded ST data as an example to tell how the whole system is used.
                                """),
                                html.Br(),
                                html.Ol([
                                    html.Li([
                                        """
                                        Upon loading the sample data, the paths to the sample datasets will be 
                                        automatically entered into the 'Expression Matrix' and 'Clinical Information' 
                                        forms.  Users also have the flexibility to input their own data by manually 
                                        entering the """,
                                        html.B("""
                                        absolute paths
                                        """),
                                        """
                                        to their datasets in the respective forms.
                                        """
                                    ]),
                                    html.Br(),
                                    html.Li(
                                        """
                                        After loading the Data, you can click the 'View' button (as shown 
                                        in the second step of the right image) to visualize the loaded table data.
                                         The data will be displayed in the 'View Data' card on the right.
                                        """

                                    ),
                                    html.Br(),
                                    html.Li(
                                        """
                                        Finally, you need to click the 'Check' button to check whether the 
                                        data format meets the requirements of the TiRank model.
                                        """
                                    )
                                ])
                            ]),
                            dbc.Col([
                                html.Img(src='./assets/bulk-data-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '-5%'
                                         }),
                            ])
                        ]
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '1.3 Spatial transcriptome Data / scRNA-seq Data'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([
                                html.Ol([
                                    html.Li([
                                        """
                                        Upon loading the sample data, the paths to the sample datasets will be 
                                        automatically entered into the 'Spatial transcriptome Data'
                                         """,
                                        html.B("""
                                        or 
                                        """),
                                        """
                                        'scRNA-seq Data' forms.  Users also have the flexibility to input their 
                                        own data by manually entering the
                                        """,
                                        # or

                                        html.B("""
                                            absolute paths
                                            """),
                                        """
                                        to their datasets in the respective forms.
                                        """
                                    ]),
                                    html.Br(),
                                    html.Li(
                                        """
                                        Same as visualizing bulk data, you can click the 'View' button (as shown 
                                        in the second step of the right image) to visualize the loaded table data.
                                         The data will be displayed in the 'View Data' card on the right.
                                        """

                                    ),
                                    html.Br(),
                                    html.Li(
                                        """
                                        Finally, you need to click the "Confirm" button to make sure that your data 
                                        is ST data (as shown in step 3 of the right image, or in the 2-2 card if you 
                                        are loading SC data). Moreover, this step is necessary because our subsequent 
                                        processing of ST data is different from that of SC data.
                                        """
                                    )
                                ])
                            ]),
                            dbc.Col([
                                html.Img(src='./assets/st-sc-data-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '-5%'
                                         }),
                            ])
                        ]
                    ),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '2. Preprocessing'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '2.1 Pre-processing Data'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([
                                html.Ol([
                                    html.Li([
                                        """
                                        In the data preprocessing section, you need to select "Enable" 
                                        or "Bypass" to decide whether to perform Normalization and
                                         Log-transformation (as shown in the first step of the right image).
                                        """
                                    ]),
                                    html.Br(),
                                    html.Li([
                                        """
                                        After that, you need to click the Perform button to preprocess the data. 
                                        """,
                                        html.Br(),
                                        html.Br(),
                                        html.B("Note"),
                                        """
                                        : The system will enter a Loading screen, this may take a few minutes, 
                                        please do not do anything in the interim, until the data Preprocessing is 
                                        completed, return to the preprocessing screen.
                                        """]

                                    ),
                                ])

                            ]),
                            dbc.Col([
                                html.Img(src='./assets/preprocess-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '-5%'
                                         }),
                            ])
                        ]
                    ),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '2.2 Mode select And GenePair Transformation'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([
                                html.Ol([
                                    html.Li([
                                        html.B("""Mode select: """),
                                        """
                                        Choose the mode you want.
                                        """
                                    ]),
                                    html.Br(),
                                    html.Br(),
                                    html.Br(),
                                    html.Li([
                                        html.B("""GenePair Transformation: """),
                                        """
                                        You need to select the values of 'Top var genes', 
                                        'P value threshold' and' Top gene pairs' from the drop-down box. 
                                        In addition, we recommend that you choose the parameter values listed below.
                                        """,
                                        html.Br(),
                                        dcc.Markdown(
                                            [
                                                """
                                                ```json
                                                'Top var genes': 2000
                                                'P value threshold': 0.05
                                                'Top gene pairs': 2000
                                                ```
                                                """
                                            ]
                                        ),
                                        """
                                        Finally, you need to click the 'Perform' button to perform the 
                                        GenePair Transformation.
                                        """,
                                        html.Br(),
                                        html.Br(),
                                        html.B("Note"),
                                        """
                                        : This is the same as the data Preprocessing, the system will 
                                        enter a Loadding interface, which may take more time, please do not 
                                        do anything in the meantime, until the data preprocessing is completed,
                                         return to the preprocessing interface.
                                        """]

                                    ),
                                ])

                            ]),
                            dbc.Col([
                                html.Img(src='./assets/preprocess-mode-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '1%'
                                         }),
                            ])
                        ]
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '2.3 View Results'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([

                                html.P([
                                    """
                                    After you have done the data preprocessing and GenePair Transformation, 
                                    you can use the radio above the right card to select the resulting
                                     plot that you are interested in for observation.
                                    """,

                                ]),

                            ]),
                            dbc.Col([
                                html.Img(src='./assets/preprocess-view-res-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '-5%'
                                         }),
                            ])
                        ]
                    ),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '3. Analysis/TiRank'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '3.1 Device select'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([

                                html.P([
                                    """
                                    You first need to choose whether you want to use CPU or GPU to train the model.
                                     In the Device drop down box, you can choose Cpu or Cuda(training with GPU). 
                                     Note that if you need to train the Tirank model with a GPU, 
                                     you will need to install torch with the appropriate version of your graphics 
                                     card driver. You can use the following command to check the version of 
                                     your graphics card driver.
                                    """,
                                    dcc.Markdown(
                                        [
                                            """
                                            ```bash
                                            nvidia-smi
                                            ```
                                            """
                                        ]
                                    ),
                                    """
                                    You can then use the following code to see if the current GPU 
                                    version of torch is available in your python environment.
                                    """,
                                    dcc.Markdown(
                                        [
                                            """
                                            ```python
                                            print(torch.cuda.is_available())
                                            ```
                                            """
                                        ]
                                    ),
                                    """
                                    If the output is True, then you can use the GPU to train the model. If the output
                                     is False, it means that the current version of torch does not correspond
                                      to your GPU version, and you need to download the corresponding version
                                      of torch from the """,
                                    html.A('Pytorch', href="https://pytorch.org/", target='_blank'),
                                    """ official website according to your GPU version.
                                    """

                                ]),

                            ]),
                            dbc.Col([
                                html.Img(src='./assets/device-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '-5%'
                                         }),
                            ])
                        ]
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '3.2 Training TiRank Model'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([
                                html.Ol([
                                    html.Li([
                                        """
                                        If necessary, you can click advanced to change the model parameters
                                         according to your needs. Or do not select advanced to use 
                                         the default parameters. The default parameters we set are as follows.
                                        """,
                                        dcc.Markdown(
                                            [
                                                """
                                                ```json
                                                'Nhead': 2
                                                'n_output': 32
                                                'nhid1': 96
                                                'nhid2': 8
                                                'nlayers': 2
                                                'n_trails': 20
                                                'dropout': 0.5
                                                ```
                                                """
                                            ]
                                        ),

                                    ]),

                                    html.Li(
                                        """
                                        Click the 'Train' button to start training the model. 
                                        This will take a lot of time, so please be patient.
                                        """

                                    ),
                                ])

                            ]),
                            dbc.Col([
                                html.Img(src='./assets/train-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '-5%'
                                         }),
                            ])
                        ]
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '3.3 Prediction'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([
                                html.Ol([
                                    html.Li([
                                        """
                                        You can select 'Reject' to perform the rejection and 'advanced'
                                         to change the prediction parameters. The default parameters are as follows.
                                        """,
                                        dcc.Markdown(
                                            [
                                                """
                                                ```json
                                                'Tolerance': 0.05
                                                'Reject_mode': 'GMM'
                                                ```
                                                """
                                            ]
                                        ),

                                    ]),

                                    html.Li(
                                        """
                                       Click the 'Predict' button to predict.
                                        """

                                    ),
                                ])

                            ]),
                            dbc.Col([
                                html.Img(src='./assets/predict-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '-5%'
                                         }),
                            ])
                        ]
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '3.4 View Results'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([
                                html.P([
                                    """
                                    When you have finished training and prediction, you can select 
                                    the resulting graph that interests you in the drop-down box in the 
                                    upper left corner of the right card for observation.
                                    """,

                                ]),
                            ]),
                            dbc.Col([
                                html.Img(src='./assets/train-predict-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '-5%'
                                         }),
                            ])
                        ]
                    ),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col(html.H5(
                            '4. Analysis/Differential expression genes & Pathway enrichment'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row(
                        [
                            dbc.Col([
                                html.Ol([
                                    html.Li([
                                        html.B("Differential expression genes"),
                                        """
                                        : You need to select the logFC threshold and P-value threshold 
                                        values from the drop-down boxes and click the "Perform" button to execute.
                                        """,
                                    ]),
                                    html.Br(),

                                    html.Li([
                                        html.B("Pathway enrichment"),
                                        """
                                        : You just need to click the 'Perform' button 
                                        to perform Pathway enrichment analysis very easily.
                                        """
                                    ]),
                                    html.Br(),

                                    html.Li([
                                        html.B("View Results"),
                                        """
                                        When you have finished the differential gene analysis, 
                                        you can see the analysis results on the right and click the download 
                                        button to download them.
                                        """
                                    ]),
                                ])

                            ]),
                            dbc.Col([
                                html.Img(src='./assets/degpe-tutorial.png',
                                         style={
                                             'height': '360px',
                                             'width': '640px',
                                             'margin-top': '-5%'
                                         }),
                            ])
                        ]
                    ),

                ], style={'font-family': 'Georgia, serif', 'padding': '20px'}
            ), )
    ], style={'margin-top': '5%', 'margin-bottom': '10%', 'margin-right': '5%', 'margin-left': '5%'}),
