
import os

import numpy as np

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

from scipy.special import erf

import pandas as pd
# import palettable
import seaborn as sns
cp = sns.color_palette()

from sklearn.neighbors.kde import KernelDensity

#%%
# loading data and setting prelims:

rsi_df = pd.read_csv('data.csv')

r = rsi_df['0'].values

d = 2
beta = 0.05
n = 1
alpha_tcc = (np.log(r)+beta*n*(d**2))/(-n*d)
gard_tcc_per_nd =  (alpha_tcc+beta*d)
rxdose_tcc = np.array(33/(alpha_tcc+beta*d))

t = np.arange(0,5,.1)
rsi_interval = np.round(np.arange(0.01,.81,.01),2)
dose_range = np.round(np.arange(40,82,2),0)

#%% dependency functions

def construct_kde(array, bandwidth=None):
    if bandwidth == None:
        bw = 1.2*array.std()*np.power(array.size,-1/5)
    else:
        bw = bandwidth
    kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    kde.fit(array.reshape(-1,1))
    x = np.linspace(array.min(),array.max(),200)
    log_dens=kde.score_samples(x.reshape(-1,1))
    kdens=np.exp(log_dens)
    return x,kdens

def cumulative_hist(hist_obj):
    counts = hist_obj[0]
    cdf_hist_obj = np.zeros(len(counts))
    total_sum = np.sum(counts)
    for i in range(len(counts)):
        cdf_hist_obj[i]  = np.sum(counts[:i])/total_sum
    return cdf_hist_obj,hist_obj[1]

""" weibull fits: """

def S1(t):
    lambda_ = 0.028907
    rho_ = 0.641806
    y = np.exp(-1*np.power(lambda_*t,rho_))
    return y

def S2(t):
    lambda_ = 0.263115
    rho_ = 1.161710
    y = np.exp(-1*np.power(lambda_*t,rho_))
    return y

wb1 = dict(x=t,y=S1(t),xaxis='x2',yaxis='y2',
           line = dict(color='rgb(.5,.5,.5)'))
wb2 = dict(x=t,y=S2(t),xaxis='x2',yaxis='y2',
           line = dict(color='rgb(.5,.5,.5)'))


""" exponential fits: """
#
# def exp_rsi(t,X):
#     mu = 4.11
#     gamma =  -6.55
#     beta_0 = np.exp(-1*mu)
#     return np.exp(-beta_0*t*np.exp(-gamma*X))
#
# def exp_gard(t,G):
#     mu = 0.362
#     gamma =  0.048
#     beta_0 = np.exp(-1*mu)
#     return np.exp(-beta_0*t*np.exp(-gamma*G))

def plc_gard33(t,G33):
    """t = time in yrs, G33 = bool"""
    if G33:
        lc_gard33 = S1(t)
    else:
        lc_gard33 = S2(t)
    return lc_gard33

#%%

"""ntcp functions: """

def risk_p(dose):
    d = np.maximum(dose-18,0)
    r = d*0.068
    return r

def prob_pneumonitis(MLD):
    fx_to_lung = 8.5
    MLDh = MLD
    MLDl = 0
    b0 = -3.87
    b1 = 0.126
    prob_h = np.exp(b0+b1*MLDh)/(1+np.exp(b0+b1*MLDh))
    prob_l = np.exp(b0+b1*MLDl)/(1+np.exp(b0+b1*MLDl))
    return np.round(prob_h - prob_l,3)

def risk_e(dose):
    d = np.maximum(dose-40,0)
    r = 0.026*d
    return r

def prob_esoph(dose_h):
    if dose_h < 7.5:
        r = .01/100
    elif (dose_h >= 7.5) & (dose_h <=20):
        r = (0.08*dose_h - 0.59)/100
    elif dose_h >20:
        EUD1 = dose_h
        EUD2 = 0
        TD50 = 47
        m = 0.36
        t1 = (EUD1-TD50)/(m*TD50)
        t2 = (EUD2-TD50)/(m*TD50)
        y = (erf(t1)-erf(t2))/2
        r = np.round(y,3)
    return r

p_mce_1gy = .074

def pfs_rsi(t,rval, dose):
    lc_rsi = exp_rsi(t,rval)
    risk_e_l = (risk_p(dose)+risk_e(dose))/100
    risk_mce = 1+p_mce_1gy*dose/14
    pfs = np.power(lc_rsi,risk_mce*np.exp(risk_e_l))
    return pfs

def pfs_gard(t,G, dose):
    """t = time in yrs, G = GARD"""
    lc_gard = exp_gard(t,G)
    risk_e_l = (risk_p(dose)+risk_e(dose))/100
    risk_mce = 1+p_mce_1gy*dose/14
    pfs = np.power(lc_gard,risk_mce*np.exp(risk_e_l))
    return pfs

def pfs_gard_man(t,G, H,L,E):
    """t = time in yrs, G = GARD"""
    lc_gard = exp_gard(t,G)

    risk_e_l = prob_pneumonitis(L)+prob_esoph(E)
    risk_mce = 1+p_mce_1gy*H
    pfs = np.power(lc_gard,risk_mce*np.exp(risk_e_l))
    return pfs

def pfs_gard33(t, G33, dose):
    """plc_gard33 = S(t) local control function w/ WB model,
        dose = total dose"""

    lc_gard33 = plc_gard33(t,G33)
    risk_e_l = (risk_p(dose)+risk_e(dose))/100
    risk_mce = 1+p_mce_1gy*dose/14
    pfs = np.power(lc_gard33,risk_mce*np.exp(risk_e_l))

    return pfs

def pfs_gard33_man(t,G33, H,L,E):
    """t = time in yrs, G = GARD """
    lc_gard33 = plc_gard33(t,G33)

    risk_e_l = prob_pneumonitis(L)+prob_esoph(E)
    risk_mce = 1+p_mce_1gy*H
    pfs = np.power(lc_gard33,risk_mce*np.exp(risk_e_l))
    return pfs

#%%

""" histogram & distribution trace objects: """

bins_rsi = list(np.arange(0,.81,.03))
rsi_hist_obj = np.histogram(r,bins=bins_rsi,density=True)
bar_rsi = go.Bar(name = 'RSI',
                x=rsi_hist_obj[1][:-2], y=rsi_hist_obj[0],
                width=.02,
                marker = {'color':'rgb(.1,.7,.35)'},
                opacity=.4,
                xaxis='x2',yaxis='y2')

bins_rxdose = list(np.arange(21,141,5))+[240]
rxdose_hist_obj = np.histogram(rxdose_tcc,bins=bins_rxdose,density=True)

bar_rxdose = go.Bar(name  = 'Rx-RSI (Gy)',
                    x=rxdose_hist_obj[1][:-2],y=rxdose_hist_obj[0],
                    width = 4,
                    # marker = {'color':'rgb(.4,.1,.5)'},
                    marker = {'color':'rgb(.1,.6,.85)'},
                    opacity=.4,
                    xaxis='x1', yaxis='y1')

hist_rsi = go.Histogram(name='RSI',
                        x=r,
                        xbins=dict(start=0, end=.81,size= .03),
                        autobinx = False,
                        histnorm='probability density', marker = {'color':'rgb(.1,.75,.3)'}, opacity=.5,
                        xaxis='x2',yaxis='y2')

hist_rxdose = go.Histogram(x=rxdose_tcc,nbinsx=60,
                           histnorm='probability density',
                           opacity=.6,
                           # marker = {'color':'rgb(.1,.6,.3)'},
                           marker = {'color':'rgb(.1,.6,.85)'}, # lighter blue
                           xaxis='x1',yaxis='y1',
                           name='Rx-RSI (Gy)')

rxdose_kde = construct_kde(rxdose_tcc)
dist_rxdose = go.Scatter(x=rxdose_kde[0],y=rxdose_kde[1],
                xaxis='x1',yaxis='y1',
                hoverinfo='x',
                line=dict(color='rgb(.1,.6,.85)',width=2.5),
                showlegend=False)

rsi_kde = construct_kde(r)
dist_rsi = go.Scatter(x=rsi_kde[0],y=rsi_kde[1],
                xaxis='x2',yaxis='y2',
                line=dict(color='rgb(.1,.75,.3)'),
                showlegend=False)

"""   plot layouts:   """
#%%
rsi_range_x = [0, .8]
rsi_range_y = [0,5.5]
gard_range_x=[0,150]
gard_range_y=[0,0.033]

axes_color = 'rgb(.4,.4,.4)'
axes_width = 1
frame_on = True

layout_hists = go.Layout(
    xaxis1=dict(
        title= "RxRSI (Gy)",
        domain=[0,1],
        range=gard_range_x,
        anchor='y1',
        linecolor='rgb(.4,.4,.4)',
        linewidth=axes_width,
        mirror=frame_on,
        tickmode='array',
        tickvals = list(range(0,160,10)),
        ticktext= [0,'',20,'',40,'',60,'',80,'',100,'',120,'','',''],
        ticklen=4),
    yaxis1=dict(
        domain=[0,1],
        range=gard_range_y,
        anchor='x1',
        linecolor=axes_color,
        linewidth=axes_width,
        mirror=frame_on,
        showticklabels=False,
        showgrid=False),
    xaxis2=dict(
        title='RSI',
        titlefont={'size':12},
        linecolor=axes_color,
        domain=[0.63, .98],range=rsi_range_x,
        anchor='y2',
        tickmode='array',
        tickvals = list(np.arange(0,.9,.1)),
        ticktext= [0,'','.2','','.4','','.6','','.8'],
        ticklen=3,
        tickfont={'size':10}),
    yaxis2=dict(
        showticklabels=False,
        linecolor=axes_color,
        domain=[0.65, 1],range = rsi_range_y,
        anchor='x2',
        showgrid=False),
    width=500,
    height=500,
    margin=go.layout.Margin(
        l=40,
        b=60,
        t=50,
        r=130,
        pad=0.1),
    legend=dict(x=1.05,y=.2,
        font={'size':11},
        bgcolor= 'rgb(255,255,255)',
        bordercolor=axes_color,
        borderwidth=1),
    paper_bgcolor='rgb(.97,.97,.97)',
    title=go.layout.Title(text='[Title Here]',x=0.1,yref='container',y=.95)
)

layout_outcome_curves = go.Layout(
    xaxis=dict(
        title='Time (yrs)',
        range=[0, 5.2],
        showgrid = False,
        linecolor=axes_color,
        linewidth=axes_width,
        mirror =  True,
        tickmode='array',
        tickvals = list(range(6)),
        ticktext= list('012345'),
        ticklen=4),
    yaxis=dict(
        title = 'Probability',
        range=[0, 1],
        showgrid=False,
        linecolor=axes_color,
        linewidth=axes_width,
        mirror  = True,
        tickmode='array',
        tickvals = np.arange(0,1.1,.1).tolist(),
        ticktext= [0,'','.2','','.4','','.6','','.8','','1'],
        ticklen=4),
    width=400,
    height=500,
    margin=go.layout.Margin(
        l=70,
        r=40,
        b=70,
        t=60,
        pad=.1),
    legend=dict(x=.1,y=.1),
    paper_bgcolor='rgb(.97,.97,.97)',
    # plot_bgcolor = 'rgb(250,250,250)',
    # paper_bgcolor='rgb(20,60,110)',
    # font=dict(color='rgb(230,230,230)'),
    title=go.layout.Title(text='[Title Here...]',x=0.18,yref='container',y=.93)
)

#%%

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


#%%

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

#%%

marks={.01:'0'}
for k in rsi_interval[4::5]:
    marks[str(k)]=str(k)
#%%
app.layout = html.Div(
    [
        # top row
        html.Div([
            html.Div(
                dcc.Graph(id='histograms'),
                style={'display':'inline-block',
                       'border-style':'solid',
                       'border-width':2,
                       'border-color':'rgb(120,120,120)'}),
            html.Div(
                dcc.Graph(id='outcome-curves'),
                style={'display':'inline-block',
                       'border-style':'solid',
                       'border-width':2,
                       'border-color':'rgb(120,120,120)'}),
            html.Div(id='table-container',children=
                [
                    html.Table(
                        [
                            html.Thead([
                                html.Tr(html.Th('Outcomes',colSpan=2,style={'background-color':'rgb(240,240,240)',
                                                                            'textAlign':'center',
                                                                            'padding':5}))
                            ], style={'height':'20px'}),
                            html.Colgroup([
                                html.Col(style={'backgroundColor':'rgb(240,240,240)','width':120}),
                                html.Col(style={'width':120})
                            ]),
                            html.Tbody([
                                html.Tr([html.Th('RSI',style={'paddingLeft':10}), html.Td(id='rsi-output',style={'textAlign':'center'})]),
                                html.Tr([html.Th('Dose',style={'paddingLeft':10}), html.Td(id='dose-output',style={'textAlign':'center'})]),
                                html.Tr([html.Th('RxRSI',style={'paddingLeft':10}),html.Td(id='rxdose-output',style={'textAlign':'center'})]),
                                html.Tr([html.Th('GARD (Tx)',style={'paddingLeft':10}),html.Td(id='gard-output',style={'textAlign':'center'})]),

                                html.Tr(html.Th('Normal Tissue Doses',style={'textAlign':'center','fontSize':15,'padding':5},colSpan=2)),
                                html.Tr([html.Th(['Heart'],style={'paddingLeft':10}), html.Td(id='heart-dose-output',style={'textAlign':'center'})]),
                                html.Tr([html.Th('Lung',style={'paddingLeft':10}), html.Td(id='lung-dose-output',style={'textAlign':'center'})]),
                                html.Tr([html.Th('Esophagus',style={'paddingLeft':10}), html.Td(id='esoph-dose-output',style={'textAlign':'center'})]),

                                html.Tr(html.Th('Outcomes',style={'textAlign':'center','fontSize':15,'padding':5},colSpan=2)),
                                html.Tr([html.Th('5-yr Predicted EFS',style={'padding':10}), html.Td(id='pefs-output',style={'textAlign':'center'})])

                            ],style={'fontSize':13})
                        ], style={'margin-top':10}
                    )
                ], style={'float':'right','height':500,'margin-right':30,
                          'border-style':'hidden',
                          'border-width':2,
                          'border-color':'rgb(120,120,120)'}
            )
        ], style = {'margin-left':40, 'width':1200,'display':'block'}),

        ####### bottom row #######
        html.Div([
            # bottom left
            html.Div(children=[
                html.H6(children='RSI and Dose Selection',style={'fontSize':20}),
                html.Div([
                    html.Div(id='display-selected-rsi'),
                    html.Div(
                        dcc.Slider(
                            id = 'rsi-slider',
                            min=rsi_interval.min(),
                            max = rsi_interval.max(),
                            value = 0.20,
                            marks = marks,
                            step = 0.01,
                            updatemode='drag'
                        )
                    ),
                    html.Br(),
                    html.Br(),
                    html.Div(id='display-selected-dose'),
                    html.Div(
                        dcc.Slider(
                            id='dose-slider',
                            min=dose_range.min(),
                            max=dose_range.max(),
                            value = 60,
                            marks = {str(k):str(k) for k in dose_range},
                            step=1,
                            updatemode='drag'
                        )
                    )
                ], style = {'fontSize':14}),

                html.Div(
                    dcc.Graph(id='gard-dose-plot'),
                    style={'display':'inline-block',
                           'border-style':'solid',
                           'border-width':1,
                           'border-color':'rgb(120,120,120)',
                           'marginTop':50}
                ),
                html.Div(
                    dcc.Graph(id='cdf-hist-plot'),
                    style={'display':'inline-block',
                           'border-style':'solid',
                           'border-width':1,
                           'border-color':'rgb(120,120,120)',
                           'marginTop':20}
                )


            ], style={'width':450,'display':'inline-block'}),

            # bottom right
            html.Div(children=[
                    html.H6('Normal Tissue Dose',style={'fontSize':20}),

                    html.Table([
                        html.Thead([
                            html.Tr(children=[html.Th('Site',style={'paddingLeft':'10px','paddingBottom':'5px','paddingTop':'5px'}),
                                              html.Th('Use Total Dose', style={'padding':'5px'}),
                                              html.Th('Manually Enter',style={'padding':'5px'}),
                                              html.Th('',style={'padding':'5px'})],
                                    style={'fontSize':14,'height':40,
                                           'color':'rgb(245,245,245)','backgroundColor':'rgb(120,120,120)'})
                        ]),
                        html.Colgroup([
                            html.Col(style={'width':80,'backgroundColor':'rgb(240,240,240)'}),
                            html.Col(style={'width':90}),
                            html.Col(style={'width':100}),
                            html.Col(style={'width':115})
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Th('All',
                                        style={'paddingLeft':'10px','paddingTop':'3px','paddingBottom':'3px','backgroundColor':'#ffffff'}),
                                html.Td([
                                    dcc.RadioItems(id = 'apply-all',
                                                  options=[{'label': '', 'value': 'auto'},
                                                  {'label': '', 'value': 'manual'}],
                                                  labelStyle={'display': 'inline-block','paddingRight':'60px'},
                                                  value='',
                                                  style={'display':'inline-block'})
                                    ], colSpan=2 , style={'paddingBottom':5,'paddingTop':5,'paddingLeft':'15px'}),
                                html.Td([
                                    html.Button(
                                        id='clear-button',
                                        n_clicks=0,
                                        children=html.Label('X',style={'position':'relative','top':'-10px','z-index':-1}),
                                        style={'width':20,'height':20,
                                               'padding':0,
                                               'fontSize':11,
                                               'fontWeight':'bold',
                                               # 'color':'rgb(180,20,20)',
                                               'backgroundColor':'#d8d8d8',
                                               'vertical-align':'text-bottom',
                                               'z-index':1
                                               }
                                    )
                                ], style={'paddingBottom':5,'paddingTop':5})
                            ], style={'height':'30px'}
                            ),

                            html.Tr([
                                html.Th(['Heart',html.Br(),'mean dose'],style={'paddingLeft':'10px'}),

                                html.Td([
                                    html.Div(id='heart_radio-table-cell',children=[
                                        dcc.RadioItems(id = 'entry-method-heart',
                                                      options=[{'label': ' ', 'value': 'auto'},
                                                      {'label': ' ', 'value': 'manual'}],
                                                      labelStyle={'display': 'inline-block','paddingRight':'60px'},
                                                      value='',
                                                      style={'display':'inline-block'},
                                                      ),
                                    ]),
                                    html.Div(id='h-repl-radio',hidden=True)
                                ],colSpan=2),

                                html.Td([
                                    html.Div(id='heart_input-container',children=[
                                        dcc.Input(id = 'heart-dose-input',
                                                  placeholder='% Total Dose',
                                                  type='number',value='',
                                                  n_blur=0, n_submit=0,
                                                  debounce=True,
                                                  min=0.0, max=100.0, step=1,
                                                  style={'width':'100px','display':'inline-block'})
                                    ], hidden=True),
                                    html.Div(id='heart-default-percent',hidden=True)
                                ], style={'paddingRight':15})
                            ]),

                            html.Tr([
                                html.Th(['Lung',html.Br(),'mean dose'],style={'paddingLeft':'10px'}),
                                html.Td([
                                    html.Div(id='lung_radio-container',children=[
                                        dcc.RadioItems(id = 'entry-method-lung',
                                                      options=[{'label': '', 'value': 'auto'},
                                                      {'label': '', 'value': 'manual'}],
                                                      labelStyle={'display': 'inline-block','paddingRight':'60px'},
                                                      value='',
                                                      style={'display':'inline-block'})
                                    ]),
                                    html.Div(id='l-repl-radio', hidden=True)
                                ], colSpan=2),

                                html.Td([
                                    html.Div(id='lung_input-container',children=[
                                        dcc.Input(id = 'lung-dose-input',
                                                  placeholder='% Total Dose',
                                                  type='number',value='',
                                                  debounce=True,
                                                  min=0.0, max=100.0, step=1,
                                                  n_blur=0, n_submit=0,
                                                  style={'width':'100px','display':'inline-block'})
                                    ], hidden=True),
                                    html.Div(id='lung-default-percent',hidden=True)
                                ])

                            ]),
                            html.Tr([
                                html.Th(['Esophagus',html.Br(),'mean dose'],style={'paddingLeft':'10px'}),
                                html.Td([
                                    html.Div(id='esoph_radio-container',children=[
                                        dcc.RadioItems(id = 'entry-method-esoph',
                                                      options=[{'label': '', 'value': 'auto'},
                                                      {'label': '', 'value': 'manual'}],
                                                      labelStyle={'display': 'inline-block','paddingRight':'60px'},
                                                      value='',
                                                      style={'display':'inline-block'})
                                    ]),
                                    html.Div(id='e-repl-radio',hidden=True)
                                ], colSpan=2),
                                html.Td([
                                    html.Div(id='esoph_input-container',children=[
                                        dcc.Input(id = 'esoph-dose-input',
                                                  placeholder='% Total Dose',
                                                  type='number', value='',
                                                  debounce=True,
                                                  min=0.0, max=100.0, step=1,
                                                  n_blur=0, n_submit=0,
                                                  style={'width':'100px','display':'inline-block'})
                                    ], hidden=True),
                                    html.Div(id='esoph-default-percent',hidden=True)
                                ])
                            ])
                        ], style={'fontSize':13})
                    ]),

            ], style={'width':500,'margin-left':10,'margin-right':40,'float':'right'}
             ),
             # draft of alternate output table #
            # html.Div([
            #     html.Table(
            #         [
            #             html.Thead([
            #                 html.Tr(html.Th('Outcomes',colSpan=2,style={'background-color':'rgb(240,240,240)',
            #                                                             'textAlign':'center',
            #                                                             'padding':5}))
            #             ], style={'height':'20px'}),
            #             html.Colgroup([
            #                 html.Col(style={'backgroundColor':'rgb(240,240,240)','width':120}),
            #                 html.Col(style={'width':120})
            #             ]),
            #             html.Tbody([
            #                 html.Tr([html.Th('RSI',style={'paddingLeft':10}), html.Td(id='____rsi-output',style={'textAlign':'center'})]),
            #                 html.Tr([html.Th('Dose',style={'paddingLeft':10}), html.Td(id='_____dose-output',style={'textAlign':'center'})]),
            #                 html.Tr([html.Th('RxDose',style={'paddingLeft':10}),html.Td(id='______rxdose-output',style={'textAlign':'center'})]),
            #                 html.Tr([html.Th('GARD (Tx)',style={'paddingLeft':10}),html.Td(id='_____gard-output',style={'textAlign':'center'})]),
            #
            #                 html.Tr(html.Th('Normal Tissue Doses',style={'textAlign':'center','fontSize':15,'padding':5},colSpan=2)),
            #                 html.Tr([html.Th('Heart',style={'paddingLeft':10}), html.Td(id='_____heart-dose-output',style={'textAlign':'center'})]),
            #                 html.Tr([html.Th('Lung',style={'paddingLeft':10}), html.Td(id='_____lung-dose-output',style={'textAlign':'center'})]),
            #                 html.Tr([html.Th('Esophagus',style={'paddingLeft':10}), html.Td(id='_____esoph-dose-output',style={'textAlign':'center'})]),
            #
            #                 html.Tr(html.Th('Outcomes',style={'textAlign':'center','fontSize':15,'padding':5},colSpan=2)),
            #                 html.Tr([html.Th('5-yr Predicted EFS',style={'padding':10}), html.Td(id='_____pefs-output',style={'textAlign':'center'})])
            #
            #             ],style={'fontSize':13})
            #         ], style={'margin-top':10}
            #     )
            # ])
            # end of draft of table  #

        ], style={'width':1050, 'margin-left':50,'margin-top':5, 'display':'block'})

    ], style={'margin-top':40}

)


@app.callback([
    Output('display-selected-rsi','children'),
    Output('display-selected-dose','children')],
    [Input('rsi-slider','value'),
    Input('dose-slider','value')])

def update_slider_text(selected_rsi,selected_dose):
    return 'RSI: {}'.format(np.round(selected_rsi,2)), 'Treatment Dose: {} (Gy)'.format(selected_dose)


""" ********************* update hist plots ********************* """
@app.callback(
    Output('histograms','figure'),
    [Input('rsi-slider','value'),
    Input('dose-slider','value')]
)

def update_hist_figures(selected_rsi,selected_dose):
    rval = selected_rsi
    alpha_val = (np.log(rval)+beta*n*(d**2))/(-n*d)
    rxdose_val = 33/(alpha_val+beta*d)
    dose_val = selected_dose
    gard_val = dose_val*(alpha_val+beta*d)
    G33 = True if (dose_val>=rxdose_val) else False

    gard_tcc = gard_tcc_per_nd*dose_val
    hist_gard = go.Histogram(x=gard_tcc,nbinsx=60,histnorm='probability density',opacity=.6, marker = {'color':'rgb(.1,.6,.3)'},xaxis='x1',yaxis='y1')
    # traces = [hist_rsi,hist_rxdose, dist_rxdose, dist_rsi]
    traces = [bar_rsi, dist_rsi, bar_rxdose, dist_rxdose]

    ticker_color = 'rgb(.95,.6,.2)'
    traces.append(go.Scatter(
        name='Selected RSI',
        xaxis='x2',yaxis='y2',
        line = {'color':ticker_color,'width':1,'dash':'dash'},
        y=np.linspace(0,6,50),
        x=np.full((50),rval))
    )
    traces.append(go.Scatter(
        name = 'RxRSI',
        xaxis='x1', yaxis='y1',
        line = {'color':ticker_color,'width':2.75},
        y=np.linspace(0,gard_range_y[1],50),
        x=np.full((50),rxdose_val))
    )
    traces.append(go.Scatter(
        name = 'Treatment Dose',
        xaxis='x1', yaxis='y1',
        line = {'color':'rgb(.95,.2,.1)','width':2.75},
        y=np.linspace(0,gard_range_y[1],50),
        x=np.full((50),dose_val))
    )
    return {
        'data':traces,
        'layout':layout_hists
    }
"""----------------------------------------------------------------"""


""" ********************* update gard-dose plot ********************* """
@app.callback(
    Output('gard-dose-plot','figure'),
    [Input('rsi-slider','value'),
    Input('dose-slider','value')]
)
def update_gard_dose_figure(selected_rsi,selected_dose):
    rsi_i = selected_rsi
    d = 2
    beta = 0.05
    n = 1
    alpha_i = (np.log(rsi_i)+beta*n*(d**2))/(-n*d)
    gard_nd_i = alpha_i+beta*d

    dose = np.arange(0,102,2)
    gard_i=gard_nd_i*dose

    max_y = 80

    trace3=go.Scatter(
        x=dose,
        y=gard_i,
        line={'color':'rgb(20,20,20)'}
    )
    trace1 = go.Scatter(
        x=dose,
        y=np.full(shape=len(dose),fill_value=33),
        # line=dict(color='rgb(220,60,60)'),
        line=dict(color='rgb(250,160,140)'),
        mode= 'lines',
        stackgroup='one'
    )
    trace2 = go.Scatter(
        x=dose,
        y=np.full(shape=len(dose),fill_value=max_y-33),
        line=dict(color='rgb(120,190,250)'),
        mode= 'lines',
        stackgroup='one'
    )
    trace4 = go.Scatter(
        name = 'Treatment Dose',
        xaxis='x1', yaxis='y1',
        # line = {'color':'rgb(.9,.5,.1)','width':2.5},
        line = {'color':'rgb(255,50,25)','width':2},
        y=np.linspace(0,max_y,50),
        x=np.full((50),selected_dose)
    )
    traces = [trace1,trace2,trace3,trace4]

    layout=go.Layout(
        xaxis=dict(
            title= "Dose",
            range=[0,100],
            linecolor='rgb(.4,.4,.4)',
            linewidth=1,
            mirror=False
            ),
        yaxis=dict(
            title="GARD",
            range=[0,max_y],
            linecolor='rgb(.4,.4,.4)',
            linewidth=1,
            mirror=False
            ),
        width=400,
        height=300,
        margin=go.layout.Margin(
            l=50,
            r=40,
            b=50,
            t=30,
            pad=.1),
        showlegend=False
    )

    return {
        'data':traces,
        'layout':layout
    }

"""----------------------------------------------------------------"""
@app.callback(
    Output('cdf-hist-plot','figure'),
    [Input('rsi-slider','value'),
    Input('dose-slider','value')]
)

def update_cdf_hist(selected_rsi,selected_dose):
    rval = selected_rsi
    alpha_val = (np.log(rval)+beta*n*(d**2))/(-n*d)
    rxdose_val = 33/(alpha_val+beta*d)
    dose_val = selected_dose
    gard_val = dose_val*(alpha_val+beta*d)
    G33 = True if (dose_val>=rxdose_val) else False

    def construct_kde(array, bandwidth=None):
        if bandwidth == None:
            bw = 1.2*array.std()*np.power(array.size,-1/5)
        else:
            bw = bandwidth
        kde = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde.fit(array.reshape(-1,1))
        x = np.linspace(array.min(),array.max(),200)
        log_dens=kde.score_samples(x.reshape(-1,1))
        kdens=np.exp(log_dens)

        total_dens=np.sum(kdens)
        cdf_array=np.zeros(shape=len(x))
        for i in range(len(x)):
            cdf_array[i] = np.sum(kdens[:i])/total_dens
        return x,kdens,cdf_array



    rsi_kde = construct_kde(r)
    dist_rsi = go.Scatter(x=rsi_kde[0],y=rsi_kde[1],
                    xaxis='x2',yaxis='y2',
                    line=dict(color='rgb(.1,.6,.3)'),
                    showlegend=False)

    cdf_rsi_trace = go.Scatter(x=rsi_kde[0],y=rsi_kde[2]*4.5,
                    xaxis='x2',yaxis='y2',
                    line=dict(color='rgb(.1,.6,.3)'),
                    showlegend=False)

    rsi_hist_obj = np.histogram(r,bins=list(np.arange(0.02,.82,.04)),density=True)
    bar_rsi = go.Bar(x=rsi_hist_obj[1][:-2],y=rsi_hist_obj[0],
                 width=.02,
                 marker = {'color':'rgb(.1,.7,.35)'}, opacity=.6,
                 xaxis='x2',yaxis='y2')
    rsi_cdf_hist = cumulative_hist(rsi_hist_obj)
    cdf_bar_rsi = go.Bar(x=rsi_cdf_hist[1][:-2],y=rsi_cdf_hist[0]*4.2,
                        width = .02,
                        marker = {'color':'rgb(.1,.7,.35)'}, opacity=.6,
                        name='rsi',
                        xaxis='x2',yaxis='y2')
    #####
    rxdose_kde=construct_kde(rxdose_tcc)
    dist_rxdose = go.Scatter(x=rxdose_kde[0],y=rxdose_kde[1],
                    xaxis='x1',yaxis='y1',
                    line=dict(color='rgb(.1,.6,.85)'),
                    showlegend=False)
    cdf_rxdose_trace = go.Scatter(x=rxdose_kde[0],y=rxdose_kde[2]*.027,
                    xaxis='x1',yaxis='y1',
                    line=dict(color='rgb(.1,.6,.85)'),
                    showlegend=False)

    rxdose_bins = list(np.arange(21,131,5))+[240]
    rxdose_hist = np.histogram(rxdose_tcc,bins=rxdose_bins,density=True)
    bar_rxdose = go.Bar(x=rxdose_hist[1][:-2],y=rxdose_hist[0],
                        width = 4,
                        marker = {'color':'rgb(.1,.6,.85)'},
                        opacity=.4,name='rxrsi')
    rxdose_cdf_hist = cumulative_hist(rxdose_hist)
    cdf_bar_rxdose = go.Bar(x=rxdose_cdf_hist[1][1:],y=rxdose_cdf_hist[0]*.027,
                        width = 4,
                        marker = {'color':'rgb(.1,.6,.85)'},
                        opacity=.4,name='rxrsi')

    traces = [cdf_rxdose_trace,cdf_bar_rxdose,cdf_rsi_trace,cdf_bar_rsi]

    # traces = [bar_rsi, dist_rsi, bar_rxdose, dist_rxdose]

    rsi_range_x = [0, .8]
    rsi_range_y = [0,5.5]
    gard_range_x=[0,130]
    gard_range_y=[0,0.038]

    ticker_color = 'rgb(.95,.6,.2)'
    traces.append(go.Scatter(
        name='Selected RSI',
        xaxis='x2',yaxis='y2',
        line = {'color':ticker_color,'width':1,'dash':'dash'},
        y=np.linspace(0,6,50),
        x=np.full((50),rval))
    )
    traces.append(go.Scatter(
        name = 'RxRSI',
        xaxis='x1', yaxis='y1',
        line = {'color':ticker_color,'width':2.75},
        y=np.linspace(0,gard_range_y[1],50),
        x=np.full((50),rxdose_val))
    )
    traces.append(go.Scatter(
        name = 'Treatment Dose',
        xaxis='x1', yaxis='y1',
        line = {'color':'rgb(.95,.2,.1)','width':2.75},
        y=np.linspace(0,gard_range_y[1],50),
        x=np.full((50),dose_val))
    )
    frame = False
    axes_color = 'rgb(.5,.5,.5)'

    layout = go.Layout(
        xaxis1=dict(
            title= "Dose (Gy)",
            domain=[0,1],
            range=gard_range_x,
            anchor='y1',
            showgrid=False,
            linecolor=axes_color,
            linewidth=1.4,
            mirror=frame,
            tickmode='array',
            tickvals = list(range(0,160,10)),
            ticktext= [0,'',20,'',40,'',60,'',80,'',100,'',120,'','',''],
            ticklen=4),
        yaxis1=dict(
            domain=[0,1],
            range=gard_range_y,
            anchor='x1',
            linecolor=axes_color,
            linewidth=1.4,
            mirror=frame,
            showticklabels=False,
            showgrid=False),
        xaxis2=dict(
            title='RSI',
            titlefont={'size':12},
            linecolor = axes_color,
            linewidth=1.4,
            showgrid=False,
            domain=[0.05, .5],range=rsi_range_x,
            anchor='y2',
            tickmode='array',
            tickvals = list(np.arange(0,.9,.1)),
            ticktext= [0,'','.2','','.4','','.6','','.8'],
            ticklen=3,
            tickfont={'size':10}),
        yaxis2=dict(
            showticklabels=False,
            linecolor= axes_color,
            linewidth=1.4,
            domain=[0.55, 1],range = rsi_range_y,
            anchor='x2',
            showgrid=False),

        width=400,
        height=400,
        margin=go.layout.Margin(
            l=40,
            b=40,
            t=20,
            r=60,
            pad=0.1
        ),
        # title=go.layout.Title(text='[Title Here]',x=0.1,yref='container',y=.91),
        showlegend=False
    )
    return {
        'data':traces,
        'layout':layout
    }

"""----------------------------------------------------------------"""

""" *************** add rsi, gard, dose to output table *************** """
@app.callback([Output('rsi-output','children'),
               Output('dose-output','children'),
               Output('rxdose-output','children'),
               Output('gard-output','children')],
               [Input('rsi-slider','value'),
                Input('dose-slider','value')]
)
def update_output_table_rsi_gard(selected_rsi,selected_dose):
    rval = np.round(selected_rsi,2)
    alpha_val = (np.log(rval)+beta*n*(d**2))/(-n*d)
    rxdose_val = np.round(33/(alpha_val+beta*d),0)
    dose_val = selected_dose
    gard_val = np.round(dose_val*(alpha_val+beta*d),0)
    G33 = True if (dose_val>=rxdose_val) else False

    return rval, dose_val, rxdose_val, gard_val
"""------------------------------------------------------------------"""


""" ********************* plot outcome curve *********************
------------------------------------------------------------------"""
@app.callback([Output('outcome-curves','figure'),
              Output('pefs-output','children')],
              [Input('rsi-slider','value'),
               Input('dose-slider','value'),
               Input('heart-dose-output','children'),
               Input('lung-dose-output','children'),
               Input('esoph-dose-output','children')]
)
def update_outcome_figure(selected_rsi,selected_dose,hdose,ldose,edose):
    rval = selected_rsi
    alpha_val = (np.log(rval)+beta*n*(d**2))/(-n*d)
    rxdose_val = 33/(alpha_val+beta*d)
    dose_val = selected_dose
    gard_val = dose_val*(alpha_val+beta*d)
    G33 = True if (dose_val>=rxdose_val) else False

    traces = []
    tb_output = []
    normal_tissue_doses = np.array([hdose,ldose,edose])

    # if entry_method == 'auto':
    #     penalized_version = pfs_gard33(t,G33,dose_val)
    # elif (entry_method == 'manual'):
    #         penalized_version= pfs_gard33_man(t,G33,hdose,ldose,edose)

    normal_tissue_doses = np.array([hdose,ldose,edose])
    if (np.any(normal_tissue_doses=='')) or (np.any(normal_tissue_doses==None)):
        all_doses_entered = False
    else:
        all_doses_entered = True

    if all_doses_entered:
        pfs_array = pfs_gard33_man(t,G33,hdose,ldose,edose)
        traces.append(go.Scatter(
            name='Predicted-EFS',
            # line = {'color':'rgb(.4,.1,.5)'},
            line = {'color':'rgb(.8,.2,.1)'},
            x=t,
            y=pfs_array,
            visible=True,
            showlegend=True))
        tb_output.append(np.round(pfs_array[49],2))
        vis=True
    else:
        tb_output.append('')
        vis=False

    traces.append(go.Scatter(
        name='Predicted-LC',
        line = dict(color='rgb(.4,.1,.5)'),
        x=t,
        y=plc_gard33(t,G33),
        visible=vis,
        showlegend=True))

    return [{'data':traces,'layout':layout_outcome_curves}] + tb_output
"""------------------------------------------------------------------"""


""" ********************* add doses output table *********************
----------------------------------------------------------------------"""
@app.callback([Output('heart-dose-output','children'),
               Output('lung-dose-output','children'),
               Output('esoph-dose-output','children')],
                [Input('dose-slider','value'),
                 Input('entry-method-heart','value'),
                 Input('entry-method-lung','value'),
                 Input('entry-method-esoph','value'),
                 Input('heart-dose-input','value'),
                 Input('lung-dose-input','value'),
                 Input('esoph-dose-input','value')])
def update_output_table_doses(selected_dose,h_entry_method,l_entry_method,e_entry_method, hdose, ldose, edose):

    dose_val = selected_dose
    entry_methods = [h_entry_method,l_entry_method,e_entry_method]
    fx_total = [14,8.5,4]
    manual_percent= [hdose, ldose, edose]
    site_dose_outputs = []

    for i in range(len(entry_methods)):
        if (entry_methods[i]=='auto'):
            site_dose_outputs.append(np.round(selected_dose/fx_total[i],1))
        elif (entry_methods[i] == 'manual'):
            site_dose_outputs.append(np.round(selected_dose*manual_percent[i]/100,1))
        else:
            site_dose_outputs.append('')

    return site_dose_outputs
"""--------------------------------------------------------"""

"""***************************** reset dose inputs ***************************
------------------------------------------------------------------------------"""
@app.callback([Output('heart-dose-input','value'),
               Output('lung-dose-input','value'),
               Output('esoph-dose-input','value')],
              [Input('entry-method-heart','value'),
               Input('entry-method-lung','value'),
               Input('entry-method-esoph','value')],
              [State('heart-dose-input','n_blur'),State('heart-dose-input','n_submit'),State('heart-dose-input','value'),
              State('lung-dose-input','n_blur'),State('lung-dose-input','n_submit'),State('lung-dose-input','value'),
              State('esoph-dose-input','n_blur'),State('esoph-dose-input','n_submit'),State('esoph-dose-input','value')]
)
def reset_dose_input(h_type,l_type,e_type,h_nblur,h_nsubmit,h_cur_val,l_nblur,l_nsubmit,l_cur_val,e_nblur,e_nsubmit,e_cur_val):
    entry_methods  = [h_type,l_type,e_type]
    nsubmits = [h_nsubmit,l_nsubmit,e_nsubmit]
    nblurs = [h_nblur,l_nblur,e_nblur]
    cur_vals = [h_cur_val,l_cur_val,e_cur_val]

    output_vals = []
    for i in range(3):
        if  (entry_methods[i] != 'manual') and ((nblurs[i]>0) or (nsubmits[i]>0)):
            new_val = ''
            output_vals.append(new_val)
        else:
            output_vals.append(cur_vals[i])

    return output_vals
"""----------------------------------------------------------------------"""



""" *************** update disable property of dose inputs ***************** """

@app.callback([Output('heart-dose-input','disabled'),
               Output('lung-dose-input','disabled'),
               Output('esoph-dose-input','disabled'),

               Output('heart_input-container','hidden'),
               Output('lung_input-container','hidden'),
               Output('esoph_input-container','hidden'),

               Output('heart-default-percent','hidden'),
               Output('lung-default-percent','hidden'),
               Output('esoph-default-percent','hidden'),

               Output('heart-default-percent','children'),
               Output('lung-default-percent','children'),
               Output('esoph-default-percent','children'),
               ],
              [Input('entry-method-heart','value'),
               Input('entry-method-lung','value'),
               Input('entry-method-esoph','value')])

def update_dose_input_editable(h_entry_method,l_entry_method,e_entry_method):
    entry_methods = [h_entry_method,l_entry_method,e_entry_method]
    inputs_disabled = []
    inputs_hidden = []
    auto_pp_hidden = []
    auto_pp_value = []
    defaults = ['7','12','25']
    for i in range(len(entry_methods)):
        if (entry_methods[i] == 'auto'):
            inputs_disabled.append(True)
            inputs_hidden.append(True)
            auto_pp_hidden.append(False)
            auto_pp_value.append('{}%'.format(defaults[i]))
        elif entry_methods[i] == 'manual':
            inputs_disabled.append(False)
            inputs_hidden.append(False)
            auto_pp_hidden.append(True)
            auto_pp_value.append('')
        elif entry_methods[i] == '':
            inputs_disabled.append(True)
            inputs_hidden.append(True)
            auto_pp_hidden.append(True)
            auto_pp_value.append('')

    # disable_h = True if h_entry_method == 'auto' else False
    # disable_l = True if l_entry_method == 'auto' else False
    # disable_e = True if e_entry_method == 'auto' else False
    #
    # hidden_e = True if e_entry_method == 'auto' else False

    # return [disable_h,disable_l,disable_e,hidden_e]
    return inputs_disabled + inputs_hidden + auto_pp_hidden + auto_pp_value
"""----------------------------------------------------------------------"""


""" ********************* entry method-apply to all sites ********************* """

@app.callback(
    [Output('entry-method-heart','value'),
     Output('entry-method-lung','value'),
     Output('entry-method-esoph','value'),
     Output('heart_radio-table-cell','hidden'),
     Output('lung_radio-container','hidden'),
     Output('esoph_radio-container','hidden'),
     Output('h-repl-radio','children'),Output('h-repl-radio','hidden'),
     Output('l-repl-radio','children'),Output('l-repl-radio','hidden'),
     Output('e-repl-radio','children'),Output('e-repl-radio','hidden')],
    [Input('apply-all','value')])

def update_all_dose_radios(entry_method_all):
    repl_radio_left= dcc.RadioItems(options=[{'label': '', 'value': 'a'}],
                                        labelStyle={'display': 'inline-block','paddingRight':'60px'},
                                        value='a', style={'display':'inline-block'})
    repl_radio_right= dcc.RadioItems(options=[{'label': '', 'value': 'a'}],
                                        labelStyle={'display': 'inline-block','paddingLeft':'78px'},
                                        value='a', style={'display':'inline-block'})
    if (entry_method_all == 'auto'):
        val = 'auto'
        hidden=True
        repl_radio = repl_radio_left
        repl_hidden = False
    elif (entry_method_all == 'manual'):
        val = 'manual'
        hidden=True
        repl_radio = repl_radio_right
        repl_hidden = False
    else:
        val = ''
        hidden=False
        repl_radio=''
        repl_hidden=True
    return [val]*3 + [hidden]*3 + [repl_radio, repl_hidden]*3
"""----------------------------------------------------------------------"""



"""----------------------- clear button ------------------------------"""
@app.callback([Output('apply-all','value')],
              [Input('clear-button','n_clicks')])
def  reset_radio_buttons(n_clicks):
    return ['cleared']
"""----------------------------------------------------------------------"""




##################

"""rsi and gard continuous outcome curves"""
    # traces.append(go.Scatter(
    #     name='plc_rsi',
    #     xaxis='x2',yaxis='y2',
    #     line = {'color':ticker_color},
    #     x=t,y=exp_rsi(t,rval),
    #     visible=False))
    # traces.append(go.Scatter(
    #     name='plc_gard',
    #     xaxis='x2',
    #     yaxis='y2',
    #     line = dict(color='rgb(.8,.1,.1)'),
    #     x=t,
    #     y=exp_gard(t,gard_val),
    #     visible=False))
    # traces.append(go.Scatter(
    #     name='penalized-GARD',
    #     xaxis='x2',yaxis='y2',
    #     line = {'color':'rgb(.4,.1,.5)'},
    #     x=t,
    #     y=penalized_version,
    #     visible=False))





if __name__ == '__main__':
    app.run_server(debug=True)
