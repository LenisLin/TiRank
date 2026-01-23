# -*- coding: utf-8 -*-
# @Date : 1.15.2024
# @Author : LingLUO
import os.path

import torch
import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from layout.homepage import homepage_layout
from layout.upload import upload_layout
from layout.preprocess import preprocess_layout
from layout.tirank import tirank_layout
from layout.degpe import degpe_layout
from layout.others import others_layout
from layout.tutorial import tutorial_layout
from tirankWeb.loaddata import load_bulk_exp_, load_bulk_clinical_, load_st_data_, load_sc_data_
from components.modes import none_state, train_advanced_card, \
    predict_advanced_card, qc_violins_layout, leiden_cluster_layout, patho_label_layout, tirank_pred_score_distribution, \
    train_epoch_loss_plot, umap_of_tirank_label_score, umap_of_tirank_pred_score, \
    spatial_of_tirank_label_score, spatial_of_tirank_pred_score
from tirank.LoadData import *
from tirank.SCSTpreprocess import *
from tirankWeb.SCSTpreprocess import clustering_, compute_similarity_, FilteringAnndata_
from tirank.Imageprocessing import GetPathoClass
from tirankWeb.dataloader import generate_val_, pack_data_
from tirankWeb.GPextractor import GenePairExtractor
from tirankWeb.Model import initial_model_para_
from tirankWeb.TrainPre import tune_hyperparameters_, predict_
from tirankWeb.Visualization import plot_score_distribution_, deg_analysis_, deg_volcano_, pathway_enrichment, \
    plot_score_umap_, plot_label_distribution_among_conditions_

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
app.title = 'TiRank'
app._favicon = 'TiRank_white.png'

bg_color = "#333333"
font_color = "#F3F6FA"

mode_global = 'Cox'
infer_mode_global = 'ST'
col_1_global = []
regression_col_1_global = []
bio_col_1_global = []
st_data = None
sc_data = None

tabs_styles1 = {
    'margin-top': '-50px',
    'width': '1350px',
    'margin-left': '200px',
}
tabs_styles2 = {
    'margin-top': '-50px',
    'width': '506px',
    'margin-left': '944px',
}
tab_style = {
    'borderBottom': '0px solid #0D1A51',
    'borderTop': '0px solid #d6d6d6',
    'borderLeft': '0px solid #d6d6d6',
    'borderRight': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'backgroundColor': '#333333',
}

tab_selected_style = {
    'borderTop': '0px solid #d6d6d6',
    'borderBottom': '3px solid #ffffff',
    'borderLeft': '0px solid #d6d6d6',
    'borderRight': '1px solid #d6d6d6',
    'backgroundColor': '#333333',
    'color': 'white',
    'padding': '6px'
}

# 定义GitHub图标样式
github_logo_style = {
    'height': '50px',
}

# 创建布局
main_layout = html.Div(id='main_page', children=[
    dcc.Location(id='url', refresh=False),
    dbc.Navbar(
        dbc.Container(
            [
                html.Div([
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                dbc.Row(
                                    [
                                        dbc.Col(html.Img(src='./assets/TiRank_white.png', height="50px",
                                                         style={'margin-left': '50px', 'border-radius': '200px'
                                                                }),
                                                ),
                                    ],
                                    align="center",
                                    className="g-0",
                                ),
                            ]),
                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Col(dbc.NavbarBrand("TiRank", className="ms-2"),
                                        style={'margin-right': '150px',
                                               }),
                            ])
                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Nav([
                                    dbc.NavItem(dbc.NavLink("Home", id='home_nav', active=True, href="home"),
                                                style={'margin-right': '80px', 'font-size': '20px'}),
                                    dbc.NavItem(
                                        dbc.NavLink("UploadData", id='upload_nav', active=False, href="upload"),
                                        style={'margin-right': '80px', 'font-size': '20px'}),
                                    dbc.NavItem(
                                        dbc.NavLink("Preprocessing", id='preprocess_nav', active=False,
                                                    href="preprocess"),
                                        style={'margin-right': '80px', 'font-size': '20px'}),
                                    # dbc.NavItem(
                                    #     dbc.NavLink("Analysis", id='analysis_nav', active=False, href="analysis"),
                                    #     style={'margin-right': '80px', 'font-size': '20px'}),
                                    dbc.NavItem([
                                        dbc.DropdownMenu(
                                            label=dbc.Label('Analysis', id='analysis_id', style={}),
                                            children=[
                                                dbc.DropdownMenuItem(
                                                    dbc.NavLink("TiRank", id='tirank', active=False,
                                                                href="tirank", style={'color': '#000000'}),
                                                    id='option1', n_clicks=0),
                                                dbc.DropdownMenuItem(
                                                    dbc.NavLink("Differential expression genes & Pathway enrichment",
                                                                id='degpe', active=False, href="degpe",
                                                                style={'color': '#000000'}), id='option2',
                                                    n_clicks=0),
                                                dbc.DropdownMenuItem(
                                                    dbc.NavLink("Other methods", id='others', active=False,
                                                                href="others", style={'color': '#000000'}),
                                                    id='option3', n_clicks=0),
                                            ],
                                            id="dropdown-analysis",
                                            nav=True,
                                            in_navbar=True,
                                        )
                                    ],
                                        style={'margin-right': '80px', 'font-size': '20px'}
                                    ),
                                    dbc.NavItem(
                                        dbc.NavLink("Tutorial", id='tutorial_nav', active=False, href="tutorial"),
                                        style={'margin-right': '80px', 'font-size': '20px'}),
                                    dbc.NavItem(dbc.NavLink("FAQ", id='faq_nav', active=False,
                                                            href="https://github.com/LenisLin/TiRank"),
                                                style={'margin-right': '150px', 'font-size': '20px'}),
                                ])
                            ]),
                        ),
                        dbc.Col(
                            html.Div([
                                html.A(
                                    html.Img(src='./assets/github.png', style=github_logo_style),
                                    href='https://github.com/LenisLin/TiRank',
                                    target='_blank',  # 在新标签页中打开链接
                                ),
                            ], style={'margin-left': 'auto'}),
                        )
                    ], className="g-0", ),
                ]),
            ],
            fluid=True,
        ),
        color=bg_color,
        dark=True,
    ),
    html.Div(
        html.A(
            html.Img(src='./assets/github.png', style=github_logo_style),
            href='https://github.com/LenisLin/TiRank',
            target='_blank',
            style={'align-self': 'center'}
        ), style={'margin-right': '1%',
                  'margin-top': '-50px'}
    ),
    html.Div(id='update_page_source'),

    # 结束栏
    html.Div(children=[
        html.Footer(children=[
            html.Div(children=[
                html.P('© 2024 tirank. All rights reserved.', style={'margin': '0'}),
                html.P('Contact us: luoling2001@163.com', style={'margin': '0'})
            ], style={'text-align': 'center', 'color': '#555', 'padding': '10px'})
        ], style={'background-color': '#333333', 'position': 'fixed', 'bottom': '0', 'width': '100%',
                  'text-align': 'center'})
    ])
])

app.layout = main_layout
app.config['suppress_callback_exceptions'] = True
app.validation_layout = html.Div([
    main_layout,
    none_state(),
    homepage_layout(),
    upload_layout(),
    preprocess_layout(),
    tirank_layout(),

])


@app.callback(
    Output('update_page_source', 'children'),
    Input('url', 'pathname'),
)
def update_page(pathname):
    if pathname == '/':
        return homepage_layout()
    elif pathname == '/home':
        return homepage_layout()
    elif pathname == '/upload':
        return upload_layout()
    elif pathname == '/preprocess':
        return preprocess_layout()
    elif pathname == '/tirank':
        return tirank_layout()
    elif pathname == '/degpe':
        return degpe_layout()
    elif pathname == '/others':
        return others_layout()
    elif pathname == '/tutorial':
        return tutorial_layout()


# 给自己写笑了 ioi
@app.callback(
    Output('home_nav', 'active'),
    Output('upload_nav', 'active'),
    Output('preprocess_nav', 'active'),
    Output('tutorial_nav', 'active'),
    Output('faq_nav', 'active'),
    Output('home_nav', 'n_clicks'),
    Output('upload_nav', 'n_clicks'),
    Output('preprocess_nav', 'n_clicks'),
    Output('tirank', 'n_clicks'),
    Output('degpe', 'n_clicks'),
    Output('others', 'n_clicks'),
    Output('tutorial_nav', 'n_clicks'),
    Output('faq_nav', 'n_clicks'),
    Output('analysis_id', 'style'),
    Input('home_nav', 'n_clicks'),
    Input('upload_nav', 'n_clicks'),
    Input('preprocess_nav', 'n_clicks'),
    Input('tirank', 'n_clicks'),
    Input('degpe', 'n_clicks'),
    Input('others', 'n_clicks'),
    Input('tutorial_nav', 'n_clicks'),
    Input('faq_nav', 'n_clicks'),
    prevent_initial_call=True,
)
def update_nav(home, upload, preprocess, tirank, degpe, others, tutorial, faq):
    if None:
        return True, False, False, False, False, 0, 0, 0, 0, 0, 0, 0, 0, {}
    elif home:
        return True, False, False, False, False, 0, 0, 0, 0, 0, 0, 0, 0, {}
    elif upload:
        return False, True, False, False, False, 0, 0, 0, 0, 0, 0, 0, 0, {}
    elif preprocess:
        return False, False, True, False, False, 0, 0, 0, 0, 0, 0, 0, 0, {}
    elif tirank:
        return False, False, False, False, False, 0, 0, 0, 0, 0, 0, 0, 0, {'color': '#FFFFFF'}
    elif degpe:
        return False, False, False, False, False, 0, 0, 0, 0, 0, 0, 0, 0, {'color': '#FFFFFF'}
    elif others:
        return False, False, False, False, False, 0, 0, 0, 0, 0, 0, 0, 0, {'color': '#FFFFFF'}
    elif tutorial:
        return False, False, False, True, False, 0, 0, 0, 0, 0, 0, 0, 0, {}
    elif faq:
        return False, False, False, False, True, 0, 0, 0, 0, 0, 0, 0, 0, {}


# upload page
@app.callback(
    Output("download-em", "data"),
    Input("download_for_em", "n_clicks"),
    State('expression-matrix', 'value'),
    prevent_initial_call=True,
)
def download_em(n_clicks, url):
    if url is not None and os.path.exists(url):
        return dcc.send_data_frame(pd.read_csv(url).to_csv, "expression matrix.csv")


@app.callback(
    Output("download-ci", "data"),
    Input("download_for_ci", "n_clicks"),
    State('clinical-info', 'value'),
    prevent_initial_call=True,
)
def download_ci(n_clicks, url):
    return dcc.send_data_frame(pd.read_csv(url).to_csv, "clinical information.csv")


@app.callback(
    Output("download-st", "data"),
    Input("download_for_st", "n_clicks"),
    State('st-info', 'value'),
    prevent_initial_call=True,
)
def download_st(n_clicks, url):
    global st_data
    if st_data is None:
        return dcc.send_data_frame(load_st_data_(url).to_csv, 'processed spatial data.csv')
    else:
        return dcc.send_data_frame(st_data.to_csv, 'processed spatial data.csv')


@app.callback(
    Output("download-sc", "data"),
    Input("download_for_sc", "n_clicks"),
    State('sc-info', 'value'),
    prevent_initial_call=True,
)
def download_sc(n_clicks, url):
    global sc_data
    if sc_data is None:
        return dcc.send_data_frame(load_sc_data_(url).to_csv, 'processed single-cell data.csv')
    else:
        return dcc.send_data_frame(sc_data.to_csv, 'processed single-cell data.csv')


@app.callback(
    Output('expression-matrix', 'value', allow_duplicate=True),
    Output('clinical-info', 'value', allow_duplicate=True),
    Output('st-info', 'value'),
    Input('load-example-switch-st', 'value'),
    prevent_initial_call=True,
)
def switch_if_upload_st_data(is_switch):
    global col_1_global
    global bio_col_1_global
    global regression_col_1_global
    if 1 in is_switch:
        col_1_global = list(pd.read_csv('./data/ExampleData/CRC_ST_Prog/GSE39582_clinical_os.csv').columns)
        bio_col_1_global = col_1_global
        regression_col_1_global = col_1_global
        return ('./data/ExampleData/CRC_ST_Prog/GSE39582_exp_os.csv',
                './data/ExampleData/CRC_ST_Prog/GSE39582_clinical_os.csv',
                './data/ExampleData/CRC_ST_Prog/SN048_A121573_Rep1')
    else:
        return None, None, None


@app.callback(
    Output('expression-matrix', 'value', allow_duplicate=True),
    Output('clinical-info', 'value', allow_duplicate=True),
    Output('sc-info', 'value'),
    Input('load-example-switch-sc', 'value'),
    prevent_initial_call=True,
)
def switch_if_upload_sc_data(is_switch):
    global col_1_global
    global regression_col_1_global
    global bio_col_1_global
    if 1 in is_switch:
        col_1_global = list(pd.read_csv('./data/ExampleData/SKCM_SC_Res/Liu2019_meta.csv').columns)
        bio_col_1_global = col_1_global
        regression_col_1_global = col_1_global
        col_1_global = list(pd.read_csv('./data/ExampleData/SKCM_SC_Res/Liu2019_meta.csv').columns)
        return ('./data/ExampleData/SKCM_SC_Res/Liu2019_exp.csv',
                './data/ExampleData/SKCM_SC_Res/Liu2019_meta.csv',
                './data/ExampleData/SKCM_SC_Res/GSE120575.h5ad')
    else:
        return None, None, None


@app.callback(
    Output("modal-centered", "is_open", allow_duplicate=True),
    Input("close-centered", "n_clicks"),
    [State("modal-centered", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output("confirm-danger-btd", "is_open", allow_duplicate=True),
    Input("confirm-danger-btd-close", "n_clicks"),
    [State("confirm-danger-btd", "is_open")],
    prevent_initial_call=True
)
def toggle_modal_confirm_danger_btd(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output("confirm-gene-empty-danger-btd", "is_open", allow_duplicate=True),
    Input("confirm-gene-empty-danger-btd-close", "n_clicks"),
    [State("confirm-gene-empty-danger-btd", "is_open")],
    prevent_initial_call=True
)
def toggle_modal_confirm_danger_btd(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output("confirm-done-btd", "is_open", allow_duplicate=True),
    Input("confirm-done-btd-close", "n_clicks"),
    [State("confirm-done-btd", "is_open")],
    prevent_initial_call=True
)
def toggle_modal_confirm_danger_btd(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output("confirm-st-btd", "is_open", allow_duplicate=True),
    Input("confirm-st-btd-close", "n_clicks"),
    [State("confirm-st-btd", "is_open")],
    prevent_initial_call=True
)
def toggle_modal_confirm_danger_btd(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output("confirm-sc-btd", "is_open", allow_duplicate=True),
    Input("confirm-sc-btd-close", "n_clicks"),
    [State("confirm-sc-btd", "is_open")],
    prevent_initial_call=True
)
def toggle_modal_confirm_danger_btd(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    [Output('view_for_em', 'n_clicks'),
     Output('upload_dataframe_view', 'data', allow_duplicate=True),
     Output('upload_dataframe_view', 'columns', allow_duplicate=True),
     Output('modal-centered', 'is_open', allow_duplicate=True)],
    [Input('view_for_em', 'n_clicks'),
     State('expression-matrix', 'value')],
    prevent_initial_call=True
)
def view_exp_matrix(n_clicks, exp_url):
    if n_clicks != 0 and exp_url is not None:
        if os.path.isfile(exp_url):
            return 0, load_bulk_exp_(exp_url).iloc[:, 0:7].to_dict('records'), [{"name": ' ', "id": 'Unnamed: 0'}] + [
                {"name": i, "id": i} for i in
                load_bulk_exp_(exp_url).iloc[:, 1:7].columns], False
        else:
            return 0, None, None, True
    elif n_clicks != 0 and exp_url is None:
        return 0, None, None, True
    else:
        return 0, None, None, False


@app.callback(
    [Output('view_for_ci', 'n_clicks'),
     Output('upload_dataframe_view', 'data', allow_duplicate=True),
     Output('upload_dataframe_view', 'columns', allow_duplicate=True),
     Output('modal-centered', 'is_open', allow_duplicate=True)],
    [Input('view_for_ci', 'n_clicks'),
     State('clinical-info', 'value')],
    prevent_initial_call=True
)
def view_exp_matrix(n_clicks, exp_url):
    global col_1_global
    global bio_col_1_global
    global regression_col_1_global
    if n_clicks != 0 and exp_url is not None:
        if os.path.isfile(exp_url):
            col_1_global = list(pd.read_csv(exp_url).columns)
            bio_col_1_global = col_1_global
            regression_col_1_global = col_1_global
            return 0, load_bulk_clinical_(exp_url).iloc[:, 0:8].to_dict('records'), [{"name": ' ',
                                                                                      "id": 'Unnamed: 0'}] + [
                                                                                        {"name": i, "id": i} for i in
                                                                                        load_bulk_clinical_(
                                                                                            exp_url).iloc[:,
                                                                                        1:8].columns], False
        else:
            return 0, None, None, True
    elif n_clicks != 0 and exp_url is None:
        return 0, None, None, True
    else:
        return 0, None, None, False


@app.callback(
    [Output('view_for_st', 'n_clicks'),
     Output('upload_dataframe_view', 'data', allow_duplicate=True),
     Output('upload_dataframe_view', 'columns', allow_duplicate=True),
     Output('modal-centered', 'is_open', allow_duplicate=True)
     ],
    [Input('view_for_st', 'n_clicks'),
     State('st-info', 'value')],
    prevent_initial_call=True
)
def view_st_matrix(n_clicks, exp_url):
    global st_data
    if n_clicks != 0 and exp_url is not None:
        if os.path.exists(exp_url):
            st_data = load_st_data_(exp_url)
            return 0, st_data.iloc[:, 0:6].to_dict('records'), [{"name": ' ', "id": 'index'}] + [
                {"name": i, "id": i} for i in
                load_st_data_(exp_url).iloc[:, 1:6].columns], False
        else:
            return 0, None, None, True
    elif n_clicks != 0 and exp_url is None:
        return 0, None, None, True
    else:
        return 0, None, None, False


@app.callback(
    [Output('view_for_sc', 'n_clicks'),
     Output('upload_dataframe_view', 'data', allow_duplicate=True),
     Output('upload_dataframe_view', 'columns', allow_duplicate=True),
     Output('modal-centered', 'is_open', allow_duplicate=True)
     ],
    [Input('view_for_sc', 'n_clicks'),
     State('sc-info', 'value')],
    prevent_initial_call=True
)
def view_sc_matrix(n_clicks, exp_url):
    global sc_data
    if n_clicks != 0 and exp_url is not None:
        if os.path.exists(exp_url):
            sc_data = load_sc_data_(exp_url)
            return 0, sc_data.iloc[:, 0:7].to_dict('records'), [{"name": ' ', "id": 'index'}] + [
                {"name": i, "id": i} for i in
                load_sc_data_(exp_url).iloc[:, 1:7].columns], False
        else:
            return 0, None, None, True
    elif n_clicks != 0 and exp_url is None:
        return 0, None, None, True
    else:
        return 0, None, None, False


def check_bulk(bulk_exp_url, bulk_clinical_url):
    bulk_exp = load_bulk_exp(bulk_exp_url)
    bulk_clinical = load_bulk_clinical(bulk_clinical_url)
    common_elements = bulk_clinical.index.intersection(bulk_exp.columns)
    if len(common_elements) == 0:
        print("The rownames of clinical information was not match with expression profile !")
        return False, True

    bulk_clinical = bulk_clinical.loc[common_elements, :]
    bulk_exp = bulk_exp.loc[:, common_elements]

    # save Data
    with open(os.path.join('./data/', 'bulk_exp.pkl'), 'wb') as f:
        pickle.dump(bulk_exp, f)
    f.close()
    with open(os.path.join('./data/', 'bulk_clinical.pkl'), 'wb') as f:
        pickle.dump(bulk_clinical, f)
    f.close()

    return True, False


@app.callback(
    Output('confirm-done-btd', 'is_open', allow_duplicate=True),
    Output('confirm-danger-btd', 'is_open', allow_duplicate=True),
    Output('check_bulk', 'n_clicks'),
    Input('check_bulk', 'n_clicks'),
    State('expression-matrix', 'value'),
    State('clinical-info', 'value'),
    prevent_initial_call=True
)
def check_bulk_td(n_clicks, exp_v, ci_v):
    if n_clicks != 0 and exp_v is not None and ci_v is not None:
        return *check_bulk(exp_v, ci_v), 0
    else:
        return False, False, 0


@app.callback(
    Output('confirm_st', 'n_clicks'),
    Output('confirm-st-btd', 'is_open', allow_duplicate=True),
    State('st-info', 'value'),
    Input('confirm_st', 'n_clicks'),
    prevent_initial_call=True

)
def confirm_st(scsd_v, n_clicks):
    global infer_mode_global
    if n_clicks != 0 and scsd_v is not None:
        infer_mode_global = 'ST'
        print(infer_mode_global)
        return 0, True
    else:
        return 0, False


@app.callback(
    Output('confirm_sc', 'n_clicks'),
    Output('confirm-sc-btd', 'is_open', allow_duplicate=True),
    State('sc-info', 'value'),
    Input('confirm_sc', 'n_clicks'),
    prevent_initial_call=True

)
def confirm_sc(scsd_v, n_clicks):
    global infer_mode_global
    if n_clicks != 0 and scsd_v is not None:
        infer_mode_global = 'SC'
        print(infer_mode_global)
        return 0, True
    else:
        return 0, False


# # preprocess


# @app.callback(
#     Output("selected-mode", "children"),
#     Input("mode-dropdown", "value")
# )
# def update_selected_text(selected_value):
#     global mode_global
#     global col_1_global
#     global regression_col_1_global
#     global bio_col_1_global
#     if selected_value is None:
#         return none_state()
#     elif selected_value == 1:
#         mode_global = 'Cox'
#         return cox(col_1_global)
#     elif selected_value == 2:
#         mode_global = 'Regression'
#         return regression(regression_col_1_global)
#     elif selected_value == 3:
#         mode_global = 'Classification'
#         return classification(bio_col_1_global)


# @app.callback(
#     Output("filtering-card", "children"),
#     Input("preprocessing-checklist", "value"),
#     prevent_initial_call=True,
# )
# def filtering_layout(selected_value):
#     if 1 in selected_value:
#         return filtering_card()


@app.callback(
    Output('preprocessing-res-col', 'children', allow_duplicate=True),
    Input('preprocess-perform', 'n_clicks'),

    Input('datasplit-perform', 'n_clicks'),
    Input('preprocess-res-radioitems', 'value'),

    State('log-radios-row', 'value'),
    State('normalization-radios-row', 'value'),

    State('mode-dropdown', 'value'),
    State('cox-time', 'value'),
    State('cox-status', 'value'),

    State('regression-var', 'value'),

    State('classification-var', 'value'),

    State('tvg-dropdown', 'value'),
    State('pvt-dropdown', 'value'),
    State('tgp-dropdown', 'value'),
    prevent_initial_call=True,
)
def update_preprocess_res(bt1, bt2, bt3, is_log, is_norm,
                          mode, cox_time, cox_status, reg_val, class_val, tvg, pvt, tgp):
    global mode_global
    triggered_id = ctx.triggered_id
    print(triggered_id)
    if triggered_id == 'preprocess-perform':
        preprocess(is_log, is_norm)
    elif triggered_id == 'datasplit-perform' and mode == 1:
        mode_global = 'Cox'
        cox_data_split(cox_time, cox_status)
        gen_pair_transformation(mode, tvg, pvt, tgp)
    elif triggered_id == 'datasplit-perform' and mode == 2:
        mode_global = 'Regression'
        regression_data_split(reg_val)
        gen_pair_transformation(mode, tvg, pvt, tgp)
    elif triggered_id == 'datasplit-perform' and mode == 3:
        mode_global = 'Classification'
        class_data_split(class_val)
        gen_pair_transformation(mode, tvg, pvt, tgp)
    elif triggered_id == 'preprocess-res-radioitems' and bt3 == 'bulk_gene_pair_heatmap':
        return dbc.Col(
            html.Img(
                src='./assets/bulk gene pair heatmap.png',
                style={
                    'width': '700px',
                    'height': '600px',
                    'margin': 'auto',
                    'display': 'flex',
                    'align-items': 'center',
                    'justify-content': 'center'
                    # Ensure the image is centered inside the column
                }
            ), style={
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center'  # Ensure the image and column content is centered
            },
        )
    elif triggered_id == 'preprocess-res-radioitems' and bt3 == 'sc_gene_pair_heatmap':
        return dbc.Col(
            html.Img(
                src='./assets/sc gene pair heatmap.png',
                style={
                    'width': '700px',
                    'height': '600px',
                    'margin': 'auto',
                    'display': 'flex',
                    'align-items': 'center',
                    'justify-content': 'center'
                    # Ensure the image is centered inside the column
                }
            )
        )


def select_view_res(v):
    if v == 'bulk_gene_pair_heatmap':
        print(v)
        return dbc.Col(
            html.Img(
                src='./assets/bulk gene pair heatmap.png',
                style={
                    'width': '700px',
                    'height': '600px',
                    'margin': 'auto',
                    'display': 'flex',
                    'align-items': 'center',
                    'justify-content': 'center'
                    # Ensure the image is centered inside the column
                }
            ),
        )
    elif v == 'sc_gene_pair_heatmap':
        return dbc.Col(
            html.Img(
                src='./assets/sc gene pair heatmap.png',
                style={
                    'width': '700px',
                    'height': '600px',
                    'margin': 'auto',
                    'display': 'flex',
                    'align-items': 'center',
                    'justify-content': 'center'
                    # Ensure the image is centered inside the column
                }
            )
        )


def preprocess(is_log, is_norm):
    global infer_mode_global
    infer_model = infer_mode_global

    f = open('./data/anndata.pkl', 'rb')
    scAnndata = pickle.load(f)
    f.close()

    if infer_model == 'ST':
        print('[preprocessing] do filtering...')
        scAnndata = FilteringAnndata_(scAnndata, max_count=35000, min_count=5000, MT_propor=10,
                                      min_cell=10, imgPath='./img/')
    elif infer_model == 'SC':
        print('[preprocessing] do filtering...')
        scAnndata = FilteringAnndata_(scAnndata, max_count=35000, min_count=5000, MT_propor=10,
                                      min_cell=10, imgPath='./img/')

    scAnndata = Normalization(scAnndata)

    scAnndata = Logtransformation(scAnndata)

    scAnndata = clustering_(scAnndata, infer_mode=infer_model, savePath='./img/')

    compute_similarity_(savePath='./data/', ann_data=scAnndata, calculate_distance=False)

    if infer_model == 'ST':
        print('[preprocessing] do patho_cluster...')
        pretrain_path = './data/pretrainModel/ctranspath.pth'
        n_patho_cluster = 6
        scAnndata = GetPathoClass(
            adata=scAnndata,
            pretrain_path=pretrain_path,
            n_clusters=n_patho_cluster,
            image_save_path=os.path.join('./assets')
        )

    with open('./data/scAnndata.pkl', 'wb') as f:
        pickle.dump(scAnndata, f)
    f.close()

    return dbc.Col([
        html.Img(
            src='./assets/white_bg.png',
            style={'width': '700px', 'height': '600px',
                   'margin': 'auto',
                   'display': 'flex',
                   'align-items': 'center'}
        )
    ])


def cox_data_split(cox_time, cox_status):
    # data split
    generate_val_('./data/', validation_proportion=0.15, mode='Cox')
    return [
        html.Img(
            src='./assets/white_bg.png',
            style={'width': '700px', 'height': '600px',
                   'margin': 'auto',
                   'display': 'flex',
                   'align-items': 'center'}
        )
    ]


def regression_data_split(reg_val):
    # data split
    generate_val_('./data/', validation_proportion=0.15, mode='Regression')
    return [
        html.Img(
            src='./assets/white_bg.png',
            style={'width': '700px', 'height': '600px',
                   'margin': 'auto',
                   'display': 'flex',
                   'align-items': 'center'}
        )
    ]


def class_data_split(class_val):
    # data split
    generate_val_('./data/', validation_proportion=0.15, mode='Classification')
    return [html.Img(
        src='./assets/white_bg.png',
        style={'width': '700px', 'height': '600px',
               'margin': 'auto',
               'display': 'flex',
               'align-items': 'center'}
    )]


def gen_pair_transformation(mode, tvg, pvt, tgp):
    mode_dict = {1: 'Cox', 2: 'Regression', 3: 'Classification'}
    if tvg:
        tvg_ = tvg
    else:
        tvg_ = 2000

    if pvt:
        pvt_ = pvt
    else:
        pvt_ = 0.05

    if tgp:
        tgp_ = tgp
    else:
        tgp_ = 1000

    g_p_extractor = GenePairExtractor(
        save_path='./data/',
        analysis_mode=mode_dict[mode],
        top_var_genes=tvg_,
        top_gene_pairs=tgp_,
        p_value_threshold=pvt_,
        padj_value_threshold=None,
        max_cutoff=0.8,
        min_cutoff=-0.8,
    )

    g_p_extractor.load_data()
    g_p_extractor.run_extraction()
    if g_p_extractor.is_empty == 1:
        return dbc.Col([html.Img(
            src='./assets/white_bg.png',
            style={'width': '700px', 'height': '600px',
                   'margin': 'auto',
                   'display': 'flex',
                   'align-items': 'center'}
        )])
    g_p_extractor.save_data()
    return dbc.Col([html.Img(
        src='./assets/white_bg.png',
        style={'width': '700px', 'height': '600px',
               'margin': 'auto',
               'display': 'flex',
               'align-items': 'center'}
    )])


@app.callback(
    Output('preprocessing-res-row', 'children'),
    Input('preprocessing-res-radioitems', 'value'),
    prevent_initial_call=True,
)
def preprocessing_res(radio_value):
    if radio_value == 'qc_violins':
        return qc_violins_layout()
    elif radio_value == 'leiden_cluster':
        return leiden_cluster_layout()
    elif radio_value == 'patho_label':
        return patho_label_layout()
    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '700px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


# analysis-tirank
@app.callback(
    Output('advanced-train-turn-row', 'children'),
    Input('advanced-train-checklist', 'value')
)
def train_advanced_turn(advanced_value):
    if 1 in advanced_value:
        return train_advanced_card()


@app.callback(
    Output('advanced-predict-turn-row', 'children'),
    Input('advanced-predict-checklist', 'value')
)
def predict_advanced_turn(advanced_value):
    if 1 in advanced_value:
        return predict_advanced_card()


@app.callback(
    Output('train_id', 'n_clicks'),
    Output('tirank-res-row', 'children', allow_duplicate=True),
    State('device-dropdown', 'value'),
    State('advanced-train-checklist', 'value'),
    State('nhead', 'value'),
    State('n_output', 'value'),
    State('nhid2', 'value'),
    State('nhid1', 'value'),
    State('nlayer', 'value'),
    State('dropout', 'value'),
    State('n_trails', 'value'),
    Input('train_id', 'n_clicks'),
    prevent_initial_call=True,
)
def train(device, advanced, nhead, n_output, nhid2, nhid1, nlayers, dropout, n_trails, n_clicks):
    global mode_global
    global infer_mode_global
    infer_mode = infer_mode_global
    save_path = './data/'
    pack_data_(save_path, mode=mode_global, infer_mode=infer_mode, batch_size=1024)

    if device is None:
        device_ = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_ = device

    url_ = './data/'

    encoder_type = 'MLP'

    if n_clicks == 1:
        if advanced is None:
            nhead_ = 2,
            nhid1_ = 96,
            nhid2_ = 8,
            n_output_ = 32,
            nlayers_ = 2,
            n_trails_ = 20,
            dropout_ = 0.5
        else:
            if nhead is not None:
                nhead_ = nhead
            else:
                nhead_ = 2
            if nhid1 is not None:
                nhid1_ = nhid1
            else:
                nhid1_ = 96
            if nhid2 is not None:
                nhid2_ = nhid2
            else:
                nhid2_ = 8
            if n_output is not None:
                n_output_ = n_output
            else:
                n_output_ = 32
            if nlayers is not None:
                nlayers_ = nlayers
            else:
                nlayers_ = 3
            if n_trails is not None:
                n_trails_ = n_trails
            else:
                n_trails_ = 5
            if dropout is not None:
                dropout_ = dropout
            else:
                dropout_ = 0.5

        initial_model_para_(save_path=url_, nhead=nhead_, nhid1=nhid1_, nhid2=nhid2_, n_output=n_output_,
                            nlayers=nlayers_, n_pred=1, dropout=dropout_, mode=mode_global,
                            encoder_type=encoder_type, infer_mode=infer_mode)
        tune_hyperparameters_(
            # Parameters Path
            save_path=url_,
            device=device_,
            n_trials=n_trails_,
        )  # optional parameters: n_trials

    return 0, train_epoch_loss_plot()


@app.callback(
    Output('predict-button', 'n_clicks'),
    Output('tirank-res-row', 'children', allow_duplicate=True),
    State('do-reject-checklist', 'value'),
    State('advanced-predict-checklist', 'value'),
    State('tolerance', 'value'),
    State('reject_model', 'value'),
    Input('predict-button', 'n_clicks'),
    prevent_initial_call=True,

)
def predict(reject, advanced, tolerance, reject_mode, n_clicks):
    global mode_global
    global infer_mode_global
    if n_clicks == 1:
        if 1 in reject:
            if 1 in advanced:
                predict_('./data/', mode=mode_global, do_reject=True, tolerance=tolerance, reject_mode=reject_mode)
            else:
                predict_('./data/', mode=mode_global, do_reject=True, tolerance=0.05, reject_mode='GMM')
        else:
            predict_('./data/', mode=mode_global, do_reject=False, tolerance=tolerance)

        plot_score_distribution_('./data/')
        plot_score_umap_('./data/', infer_mode_global)
        # plot_label_distribution_among_conditions_('./data/', group="leiden_clusters")

    return 0, tirank_pred_score_distribution()


@app.callback(
    Output('tirank-res-row', 'children', allow_duplicate=True),
    Input('tirank-res-radioitems', 'value'),
    prevent_initial_call=True,
)
def tirank_res(radio_value):
    if radio_value == 'tiRank_pred_score_distribution':
        return tirank_pred_score_distribution()
    elif radio_value == 'epoch-loss':
        return train_epoch_loss_plot()
    elif radio_value == 'UMAP_of_TiRank_Label_Score':
        return umap_of_tirank_label_score()
    elif radio_value == 'UMAP of TiRank Pred Score':
        return umap_of_tirank_pred_score()
    elif radio_value == 'Spatial of TiRank Pred Score':
        return spatial_of_tirank_pred_score()
    elif radio_value == 'Spatial of TiRank Label Score':
        return spatial_of_tirank_label_score()

    else:
        return dbc.Col(html.Img(src='./assets/white_bg.png',
                                style={'width': '700px', 'height': '700px', 'margin': 'auto',
                                       'display': 'flex', 'align-items': 'center'}))


@app.callback(
    Output('deg-plot', 'n_clicks'),
    Output('degpe_dataframe_view', 'data'),
    State('logfc-dropdown', 'value'),
    State('pvalue-dropdown', 'value'),
    Input('deg-plot', 'n_clicks'),
    prevent_initial_call=True,

)
def deg_plot(logfc, pvalue, n_clicks):
    fc_threshold = 2
    p_value_threshold = 0.05
    if logfc is not None:
        fc_threshold = eval(logfc)
    if pvalue is not None:
        p_value_threshold = eval(pvalue)
    if n_clicks == 1:
        deg_analysis_('./data/', fc_threshold=fc_threshold, Pvalue_threshold=p_value_threshold, do_p_adjust=True)
        deg_volcano_('./data/', fc_threshold=fc_threshold, Pvalue_threshold=p_value_threshold, do_p_adjust=True)
    return 0, pd.read_csv('data/Differentially expressed genes data frame.csv').to_dict('records')


# @app.callback(
#     Output('depge-row', 'children', allow_duplicate=True),
#     Input('degpe-res-radioitems', 'value'),
#     prevent_initial_call=True,
# )
# def depge_res(radio_value):
#     if radio_value == 'deg_volcano_plot':
#         return deg_volcano_plot()
#     elif radio_value == 'leiden_cluster':
#         return leiden_cluster_layout()
#     else:
#         return dbc.Col(html.Img(src='./assets/white_bg.png',
#                                 style={'width': '700px', 'height': '700px', 'margin': 'auto',
#                                        'display': 'flex', 'align-items': 'center'}))


@app.callback(
    Output("download-res-degpe", "data"),
    Input("download_for_res_degpe", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(pd.read_csv('./data/Differentially expressed genes data frame.csv').to_csv,
                               "Differentially expressed genes data frame.csv")


# enrichment-run
@app.callback(
    Output('enrichment-run', 'n_clicks'),
    Input('enrichment-run', 'n_clicks'),
    prevent_initial_call=True,
)
def enrichment_run(n_clicks):
    if n_clicks == 1:
        pathway_enrichment('./data/', database=["GO_Biological_Process_2023"])
        return 0


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050)

