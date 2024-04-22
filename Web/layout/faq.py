from dash import html, dcc
import dash_bootstrap_components as dbc


def faq_layout():
    return html.Div(children=[
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4('Contact Us', className='text-center',
                            style={'font-family': 'Georgia, serif', 'margin-bottom': '20px'}),
                    dbc.Row([
                        html.P([
                            """
                            Please don't hesitate to contact us if you have any questions or suggestions about TiRank.
                            """
                        ])
                    ]),


                ], style={'font-family': 'Georgia, serif', 'padding': '20px'}
            ), )
    ], style={'margin-top': '5%', 'margin-bottom': '10%', 'margin-right': '5%', 'margin-left': '5%'}),
