# -*- coding: utf-8 -*-
# @Date : 1.19.2024
# @Author : LingLUO

from dash import html


def homepage_layout():
    return html.Div(id='homepage-content', children=[
        html.Img(
            src='./assets/Workflow.png',
            style={
                'display': 'block',
                'margin-left': 'auto',
                'margin-right': 'auto',
                'width': '80%',
                'margin-top': '100px'
            }
        )
    ], style={'text-align': 'center', 'margin-top': '-5%'}),
