# -*- coding: utf-8 -*-
# @Date : 1.19.2024
# @Author : LingLUO

from dash import html, dcc
import dash_bootstrap_components as dbc


def others_layout():
    return html.Div(children=[
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4('Other Methods', className='text-center',
                            style={'font-family': 'Georgia, serif', 'margin-bottom': '20px'}),
                    dbc.Row([
                        dbc.Col(html.P([
                            "This document provides an overview of several cutting-edge bioinformatics algorithms,"
                            " each designed to enhance our understanding of complex biological data. ",
                            "From single-cell analysis to drug vulnerability identification,"
                            " these tools represent the forefront of computational biology research."
                        ], style={'text-align': 'justify'}), width=12),
                    ], className='mb-4'),
                    dbc.Row([
                        dbc.Col(html.Hr(), width=12)
                    ]),
                    dbc.Row([
                        dbc.Col(html.H5(
                            'Scissor: Single-Cell Identification of Subpopulations with Bulk Sample '
                            'Phenotype Correlation'),
                            width=12)
                    ], className='mb-3'),
                    dbc.Row([
                        dbc.Col(html.P([
                            html.B('Scissor '),
                            "offers a novel approach for utilizing phenotypes—such as disease stage, tumor metastasis, "
                            "treatment response, and survival outcomes—gathered from bulk assays. Its goal is to "
                            "identify the cell subpopulations most closely associated with these phenotypes within "
                            "single-cell data."
                        ], style={'text-align': 'justify'}), width=12),
                    ], className='mb-3'),

                    # html.Br(),
                    html.Img(src='https://github.com/sunduanchen/Scissor/raw/master/Figure_Method.jpg',
                             style={
                                 'height': '850px',
                                 'width': '800px',
                                 'margin-left': '25%'
                             }
                             ),
                    html.Br(),
                    html.B('Workflow of Scissor', style={'margin-left': '45%'}),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(html.A("Learn more about Scissor",
                                       href='https://github.com/sunduanchen/Scissor?tab=readme-ov-file',
                                       target='_blank'))
                    ], className='mb-4'),
                    html.Hr(),
                    html.H5('DEGAS: Diagnostic Evidence Gauge of Single Cells'),
                    html.B('DEGAS '),
                    "introduces a flexible deep transfer learning framework aimed at prioritizing cells in relation"
                    " to disease by transferring disease information from patients to cells.",
                    html.Br(),
                    html.Img(src='https://github.com/tsteelejohnson91/DEGAS/blob/master/figures/DEGAS.png?raw=true',
                             style={
                                 'height': '500px',
                                 'margin-left': '25%'
                             }
                             ),
                    html.Br(),
                    html.B('Workflow of DEGAS', style={'margin-left': '45%'}),
                    html.Br(),
                    html.A(href='https://github.com/tsteelejohnson91/DEGAS',
                           children="Explore DEGAS further", target="'_blank"),
                    html.Hr(),
                    html.H5('scAB: Multiresolution Cell State Detection'),
                    html.B('scAB '),
                    "stands out as a novel computational method that integrates single-cell genomics data with "
                    "clinically annotated bulk sequencing data, employing a knowledge- and graph-guided matrix "
                    "factorization model to detect multiresolution cell states of clinical significance.",
                    html.Br(),
                    html.Img(src='https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/nar'
                                 '/50/21/10.1093_nar_gkac1109/1/gkac1109fig1.jpeg?Expires=1714354520&Signature=D0vCpp'
                                 'CnUrL1N6fs~k20lHZUeXJWIRfj8iH7Slzj5e01a24Q2dd6X8YlcaKm5WndDArQMMYiT90gXetd5WFphT'
                                 'it8YGenyIFxG7NFHiFaYWMeB8IdDLGw7FIbwRBnIyJpDVM61sfnZ5qQMb97ZStxxIHm1GuYuNdbRQ-ZO'
                                 '0HQESxQVT2XdBl56fgnJnEw2JlzDeWgoqGhYvQX1KQBfd2Qj0Lf2J7Nm5PnKVl9LE4B~GCIS-evJAfBiDc'
                                 'JoMY8Qr14bR59YtqBOd1SqkBhachht~2jd6glTo~rMPNAHAJHYtzW7i2ET7L~om1'
                                 'TYkMTdFbYQrm48XvbDRmaphv6uB7EA__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA',
                             style={
                                 'height': '800px',
                                 'margin-left': '20%'
                             }
                             ),
                    html.Br(),
                    html.B('Workflow of scAB', style={'margin-left': '45%'}),
                    html.Br(),
                    html.A(href='https://github.com/jinworks/scAB',
                           children="Discover more about scAB", target="'_blank"),
                    html.Hr(),
                    html.H5('Beyondcell: Targeting Cancer Therapeutic Heterogeneity'),
                    html.B('Beyondcell '),
                    "delineates a method for identifying drug vulnerabilities using scRNA-seq and Spatial"
                    " Transcriptomics data, focusing on classifying cells/spots into Therapeutic Clusters "
                    "(TCs) to analyze drug-related commonalities.",
                    html.Br(),
                    html.Img(src='https://github.com/cnio-bu/beyondcell/raw/master/.img/workflow_tutorial.png',
                             style={
                                 'height': '200px',
                                 'margin-left': '20%'
                             }
                             ),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Img(src='https://github.com/cnio-bu/beyondcell/raw/master/.img/drug_signatures.png',
                             style={
                                 'height': '200px',
                                 'margin-left': '18%'
                             }
                             ),
                    html.Br(),
                    html.B('Workflow of Beyondcell', style={'margin-left': '45%'}),
                    html.Br(),
                    html.A(href='https://github.com/cnio-bu/beyondcell',
                           children="Learn more about Beyondcell", target="'_blank"),

                ], style={'font-family': 'Georgia, serif', 'padding': '20px'}
            ), )
    ], style={'margin-top': '5%', 'margin-bottom': '10%', 'margin-right': '5%', 'margin-left': '5%'}),
