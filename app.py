
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

#%%

rsi_df = pd.read_csv('data.csv')

r = rsi_df['0'].values

#%%
#
# T = lc_df['td_yrs'].values
# E = lc_df['LocalFailure'].values
#
# gard33=lc_df['gard33']
# idx = (gard33 == True)
# T1 = T[idx]
# T2 = T[~idx]
# E1 = E[idx]
# E2 = E[~idx]
#
# kmf_gt33 = KaplanMeierFitter()
# kmf_gt33.fit(T1, event_observed = E1, label = 'KM GARD>33')
# kmf_lt33 = KaplanMeierFitter()
# kmf_lt33.fit(T2, event_observed = E2, label = 'KM GARD<33')
# kmf_all = KaplanMeierFitter()
# kmf_all.fit(T, event_observed = E, label = 'KM All')
#
# from lifelines import WeibullFitter
#
# wf0 = WeibullFitter()
# wf0.fit(T, E)
# wf1 = WeibullFitter()
# wf1.fit(T1, E1)
# wf2 = WeibullFitter()
# wf2.fit(T2, E2)
#
# def S1(t):
#     return wf1.predict(t)
# def S2(t):
#     return wf2.predict(t)

#%%
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

# print(wf1.summary, wf2.summary)

#%%

""" exponential fits: """

def exp_rsi(t,X):
    mu = 4.11
    gamma =  -6.55
    beta_0 = np.exp(-1*mu)
    return np.exp(-beta_0*t*np.exp(-gamma*X))

def exp_gard(t,G):
    mu = 0.362
    gamma =  0.048
    beta_0 = np.exp(-1*mu)
    return np.exp(-beta_0*t*np.exp(-gamma*G))

#%%

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
#%%
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
#%%
p_mce_1gy = .074

def pfs_gard(t,G, dose):
    """t = time in yrs, G = GARD"""
    lc_gard = exp_gard(t,G)
    risk_e_l = (risk_p(dose)+risk_e(dose))/100
    risk_mce = 1+p_mce_1gy
    pfs = np.power(lc_gard,risk_mce*np.exp(risk_e_l))
    return pfs


def pfs_gard_man(t,G, H,L,E):
    """t = time in yrs, G = GARD"""
    lc_gard = exp_gard(t,G)

    risk_e_l = prob_pneumonitis(L)+prob_esoph(E)
    risk_mce = 1+p_mce_1gy*H
    pfs = np.power(lc_gard,risk_mce*np.exp(risk_e_l))
    return pfs

def pfs_rsi(t,rval, dose):
    lc_rsi = exp_rsi(t,rval)
    risk_e_l = (risk_p(dose)+risk_e(dose))/100
    risk_mce = 1+p_mce_1gy
    pfs = np.power(lc_rsi,risk_mce*np.exp(risk_e_l))
    return pfs

#%%

# r = nsclc.rsi.values
d = 2
beta = 0.05
n = 1
alpha_tcc = (np.log(r)+beta*n*(d**2))/(-n*d)
rxdose_tcc = np.array(33/(alpha_tcc+beta*d))

t = np.arange(0,5,.1)
rsi_interval = np.round(np.arange(0.01,.81,.01),2)
total_dose = 60
dose_range = np.round(np.arange(40,82,2),0)

# hist trace objects:

hist_rsi = go.Histogram(x=r, nbinsx=40,histnorm='probability density',
                        opacity=.6, xaxis='x3',yaxis='y3')

hist_rxdose = go.Histogram(x=rxdose_tcc,nbinsx=80,
                           histnorm='probability density',
                           opacity=.6, marker = {'color':'rgb(.1,.6,.3)'},
                           xaxis='x1',yaxis='y1')

wb1 = dict(x=t,y=S1(t),xaxis='x2',yaxis='y2',
           line = dict(color='rgb(.5,.5,.5)'))
wb2 = dict(x=t,y=S2(t),xaxis='x2',yaxis='y2',
           line = dict(color='rgb(.5,.5,.5)'))

"""   plotting layout:   """

rsi_range_x = [0, np.max(r)]
rsi_range_y = [0,6]
rxdose_range_x=[0,140]
rxdose_range_y=[0,0.03]

whole_plot_width = 900

layout = go.Layout(
    xaxis1=dict(
        title='Rx Dose',
        zeroline=True,
        domain=[0, 0.45],range=rxdose_range_x,anchor='y1',
        tickmode='array',
        tickvals = list(range(0,150,10)),
        ticktext= [0,'',20,'',40,'',60,'',80,'',100,'',120,'',''],
        ticklen=4),
    yaxis1=dict(
        showticklabels=False,
        domain=[0, 1],range=rxdose_range_y,anchor='x1',
        showgrid=False),
    xaxis2=dict(
        title='Time (yrs)',
        domain=[0.55, 1],range=[0, 5],
        anchor='y2',
        showgrid = False),
    yaxis2=dict(
        domain=[0, 1],range=[0, 1],
        anchor='x2',
        showgrid=False),
    xaxis3=dict(
        zeroline=True,
        domain=[0.25, 0.45],range=rsi_range_x,anchor='y3'),
    yaxis3=dict(
        showticklabels=False,
        domain=[0.65, 1],range = rsi_range_y,anchor='x3',
        showgrid=False),
    width=whole_plot_width,
    height=500,
    margin=go.layout.Margin(
        l=80,
        b=50,
        t=50,
        pad=4
    ),
    annotations=[
        dict(
            x=.1,
            y=5,
            xref='x3',
            yref='y3',
            text='RSI',
            showarrow=False
        )
    ],
    showlegend=False
)

#%%

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

#%%

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#%%

marks={.01:'0.0'}
for k in rsi_interval[4::5]:
    marks[str(k)]=str(k)
#%%
app.layout = html.Div([

    html.Div([
        dcc.Graph(id='graph')
    ]),

    html.Div([

        html.Div([

            html.Div(id='display-selected-rsi'),

            dcc.Slider(
                id = 'rsi-slider',
                min=rsi_interval.min(),
                max = rsi_interval.max(),
                value = rsi_interval.min(),
                marks = marks,
                step = 0.01,
                updatemode='drag'),

            html.Br(),
            html.Br(),

            html.Div(id='display-selected-dose'),

            dcc.Slider(
                id='dose-slider',
                min=dose_range.min(),
                max=dose_range.max(),
                value = dose_range.min(),
                marks = {str(k):str(k) for k in dose_range},
                step=1,
                updatemode='drag'
            )
        ], style = {'width':500, 'marginLeft':80, 'marginTop':20, 'fontSize':14,'display': 'inline-block'}),

        html.Div([

            html.Div(
                dcc.RadioItems(
                    id = 'dose-entry-method',
                    options=[{'label': 'Use Total Dose', 'value': 'auto'},
                        {'label': 'Manually Enter', 'value': 'manual'}],
                    value='auto'
                ), style={'width':120,'fontSize':14}
            ),

            html.Div([
                dcc.Input(
                    id = 'heart-dose',
                    placeholder='Enter heart dose',
                    type='number',
                    value=''
                ),
                dcc.Input(
                    id = 'lung-dose',
                    placeholder='Enter lung dose',
                    type='number',
                    value=''
                ),
                dcc.Input(
                    id = 'esoph-dose',
                    placeholder='Enter esophagus dose',
                    type='number',
                    value=''
                )
            ], style={'width':80,'display': 'inline-block','fontSize':14,'margin-top':10}),

            html.Button(id='submit-button', n_clicks=0, children='Submit')

        ], style={'width':120,'float':'right','display': 'inline-block','margin-right':80, 'margin-top':20})

    ],style={'width':850})

],style={'margin-top':20})

@app.callback(
    Output('graph','figure'),
    [Input('rsi-slider','value'),
    Input('dose-slider','value'),
    Input('dose-entry-method','value'),
    Input('submit-button', 'n_clicks')],
    [State('heart-dose','value'),
    State('lung-dose','value'),
    State('esoph-dose','value')]
)

def update_figure(selected_rsi,selected_dose,selected_entry_method,n_clicks, hdose,ldose,edose):
    rval = selected_rsi
    alpha_val = (np.log(rval)+beta*n*(d**2))/(-n*d)
    rxdose_val = 33/(alpha_val+beta*d)
    dose_val = selected_dose
    gard_val = dose_val*(alpha_val+beta*d)

    traces = [hist_rsi,hist_rxdose,wb1,wb2]
    set_vis = False # attribute for plc-rsi curve


    ticker_color = 'rgb(.92,.5,.1)'

    traces.append(go.Scatter(
        xaxis='x3',yaxis='y3',
        line = {'color':ticker_color},
        y=np.linspace(0,6,50),
        x=np.full((50),rval))
    )
    traces.append(go.Scatter(
        name = 'rxdose',
        xaxis='x1', yaxis='y1',
        line = {'color':ticker_color},
        y=np.linspace(0,0.03,50),
        x=np.full((50),rxdose_val))
    )
    traces.append(go.Scatter(
        name = 'selected dose',
        xaxis='x1', yaxis='y1',
        line = {'color':'rgb(.9,.4,.45)'},
        y=np.linspace(0,0.03,50),
        x=np.full((50),dose_val))
    )
    traces.append(go.Scatter(
        name='plc_rsi',
        xaxis='x2',yaxis='y2',
        line = {'color':ticker_color},
        x=t,y=exp_rsi(t,rval),
        visible=set_vis)
    )

    traces.append(go.Scatter(
        name='plc_gard',
        xaxis='x2',
        yaxis='y2',
        line = dict(color='rgb(.8,.1,.1)'),
        x=t,
        y=exp_gard(t,gard_val))
    )

    if selected_entry_method == 'auto':
        penalized_version = pfs_gard(t,gard_val,dose_val)
    elif selected_entry_method == 'manual':
        penalized_version= pfs_gard_man(t,gard_val,hdose,ldose,edose)

    traces.append(go.Scatter(
        name='penalized-GARD',
        xaxis='x2',yaxis='y2',
        line = {'color':'rgb(.4,.1,.5)'},
        x=t,
        y=penalized_version)
    )

    return {
        'data':traces,
        'layout':layout
    }

@app.callback([
    Output('display-selected-rsi','children'),
    Output('display-selected-dose','children')],
    [Input('rsi-slider','value'),
    Input('dose-slider','value')])

def update_slider_text(selected_rsi,selected_dose):
    return 'RSI: {}'.format(selected_rsi), 'Total Dose: {}'.format(selected_dose)

# @app.callback(
#     Output('heart-dose','disabled'),
#     Input('dose-entry-method','value'))
#
# def update_input_boxes(selected_entry_method):
#     if selected_entry_method == 'auto':
#         access = 'False'
#     if selected_entry_method == 'manual':
#         access = 'True'
#     return {'disabled':access}

if __name__ == '__main__':
    app.run_server(debug=True)
