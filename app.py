
import os

import numpy as np

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

from scipy.special import erf
from scipy import stats

import pandas as pd
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

    total_dens=np.sum(kdens)
    cdf_array=np.zeros(shape=len(x))
    delta=x[1]-x[0]
    for i in range(len(x)):
        cdf_array[i] = np.sum(kdens[:i])*delta

    return x,kdens, cdf_array

def cumulative_hist(hist_obj):
    counts = hist_obj[0]
    cdf_hist_obj = np.zeros(len(counts))
    total_sum = np.sum(counts)
    for i in range(len(counts)):
        cdf_hist_obj[i]  = np.sum(counts[:i])/total_sum
    return cdf_hist_obj,hist_obj[1]


def calc_cdf(array,var,bandwidth=None):
    if bandwidth == None:
        bw = 1.2*array.std()*np.power(array.size,-1/5)
    else:
        bw = bandwidth
    kde=stats.gaussian_kde(dataset=array,bw_method=bw)
    return kde.integrate_box_1d(low=0,high=var)

#%%

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
                    marker = {'color':'rgb(140,140,140)'}, # gray
                    # marker = {'color':'rgb(.1,.6,.85)'}, # blue
                    opacity=.4,
                    hoverinfo='none',
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
                           name='Rx-RSI (Gy)',
                           hoverinfo='none')

rxdose_kde = construct_kde(rxdose_tcc)
dist_rxdose = go.Scatter(x=rxdose_kde[0],y=rxdose_kde[1],
                xaxis='x1',yaxis='y1',
                line=dict(
                    # color='rgb(.1,.6,.85)', # blue
                    color='rgb(80,80,80)',
                    width=2.5
                ),

                hoverinfo='none',
                showlegend=False)

rsi_kde = construct_kde(r)
dist_rsi = go.Scatter(x=rsi_kde[0],y=rsi_kde[1],
                xaxis='x2',yaxis='y2',
                line=dict(color='rgb(.1,.75,.3)'),
                # line=dict(color='rgb(110,30,160)'), # purple
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
        b=70,
        t=50,
        r=130,
        pad=0.1),
    legend=dict(x=1.05,y=.2,
        font={'size':11},
        bgcolor= 'rgb(255,255,255)',
        bordercolor=axes_color,
        borderwidth=1),
    paper_bgcolor='rgb(.97,.97,.97)'
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
        b=90,
        t=70,
        pad=.1),
    legend=dict(x=.1,y=.1,
                bgcolor= 'rgb(255,255,255)',
                bordercolor=axes_color,
                borderwidth=1),
    paper_bgcolor='rgb(.97,.97,.97)'
)

#%%

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


#%%

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

#%%

marks={.01:'0'}
for k in rsi_interval[4::5]:
    marks[str(k)]=str(k)

th_style={'font-size':12,'paddingLeft':10,'paddingTop':5,'paddingBottom':5,'backgroundColor':'rgb(245,245,245)'}
td_style={'font-size':12,'textAlign':'center','padding':5}

input_style={
    'width':'60px','height':'30px','display':'inline-block',
    'paddingRight':5,
    'paddingLeft':8,
    'marginTop':1,
    'marginBottom':1
}
radio_args=dict(
    options=[{'label': ' ', 'value': 'auto'},
    {'label': ' ', 'value': 'manual'}],
    labelStyle={'display': 'inline-block','paddingRight':'60px'},
    value='')

input_args=dict(
    placeholder='0.0',
    type='number', value='',
    debounce=True,
    min=0.0, max=100.0, step=1,
    n_blur=0, n_submit=0)

""" --------------- app layout --------------- """
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
                                html.Tr(html.Th('Parameters',colSpan=2,style={'font-size':13,
                                                                           'background-color':'rgb(220,220,220)',
                                                                            'textAlign':'center',
                                                                            'padding':5}))
                            ], style={'height':'20px'}),
                            html.Colgroup([
                                html.Col(style={'backgroundColor':'rgb(220,220,220)','width':100}),
                                html.Col(style={'width':100})
                            ]),
                            html.Tbody([
                                html.Tr([html.Th('RSI',style=th_style), html.Td(id='rsi-output',style=td_style)]),
                                html.Tr([html.Th('Dose',style=th_style), html.Td(id='dose-output',style=td_style)]),
                                html.Tr([html.Th('RxRSI',style=th_style),html.Td(id='rxdose-output',style=td_style)]),
                                html.Tr([html.Th('GARD (Tx)',style=th_style),html.Td(id='gard-output',style=td_style)]),

                                html.Tr(html.Th('Normal Tissue Doses',style={'textAlign':'center','fontSize':13,'padding':5},colSpan=2)),

                                html.Tr([html.Th('Heart',style=th_style), html.Td(id='heart-dose-output',style=td_style)]),
                                html.Tr([html.Th('Lung',style=th_style), html.Td(id='lung-dose-output',style=td_style)]),
                                html.Tr([html.Th('Esophagus',style=th_style), html.Td(id='esoph-dose-output',style=td_style)]),

                                html.Tr(html.Th('Outcomes',style={'textAlign':'center','fontSize':13,'padding':5},colSpan=2)),

                                html.Tr([html.Th('1-yr pLC',style=th_style), html.Td(id='pefs-output-1',style=td_style)]),
                                html.Tr([html.Th('2-yr pLC',style=th_style), html.Td(id='pefs-output-2',style=td_style)]),
                                html.Tr([html.Th('5-yr pLC',style=th_style), html.Td(id='pefs-output-5',style=td_style)])

                            ])
                        ], style={'margin-top':10}
                    )
                ], style={'float':'right','height':500,'margin-right':50,
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


            ], style={'width':450,'display':'inline-block'}),

            # bottom right
            html.Div(children=[
                    html.H6('Normal Tissue Dose',style={'fontSize':20}),

                    html.Table([
                        html.Thead([
                            html.Tr(children=[html.Th(children=['Site',html.Br(),'(mean dose)'],rowSpan=2,style={'paddingLeft':'10px','paddingBottom':'5px','paddingTop':'5px'}),
                                              html.Th('Calculation Method:',colSpan=2,style={'padding':'5px'}),
                                              html.Th(children=['Percent of',html.Br(),'Total Dose'],rowSpan=2,style={'padding':'5px'})],
                                    style={'fontSize':13,'height':20,
                                           'color':'rgb(245,245,245)','backgroundColor':'rgb(120,120,120)'}),

                            html.Tr(children=[
                                              html.Td('Use Defaults', style={'padding':'5px'}),
                                              html.Td('Manually Enter',style={'padding':'5px'})
                                              ],
                                    style={'fontSize':12,'height':20,
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
                                html.Td('Apply to All',
                                        style={'font-size':11,'font-weight':'bold',
                                               'paddingLeft':'10px','paddingTop':'3px','paddingBottom':'3px',
                                               'color':'rgb(40,40,40)',
                                               'backgroundColor':'#ffffff'}),
                                html.Td([
                                    dcc.RadioItems(id = 'apply-all',
                                                  options=[{'label': '', 'value': 'auto'},
                                                  {'label': '', 'value': 'manual'}],
                                                  labelStyle={'display': 'inline-block','paddingRight':'60px'},
                                                  value='initial',
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
                                    ),
                                    html.Div(
                                        children=['Clear'],
                                        style={'font-weight':'bold','font-size':12,'display':'inline-block','marginLeft':5}
                                    )
                                ], style={'paddingBottom':5,'paddingTop':5})
                            ], style={'height':'30px'}
                            ),

                            html.Tr([
                                html.Th(['Heart'],style={'paddingLeft':'10px'}),

                                html.Td([
                                    html.Div(id='heart_radio-table-cell',children=[
                                        dcc.RadioItems(id = 'entry-method-heart',
                                                       **radio_args,
                                                      style={'display':'inline-block'}),
                                    ]),
                                ],colSpan=2),

                                html.Td([
                                    html.Div(id='heart_input-container',children=[
                                        dcc.Input(id = 'heart-dose-input',
                                                  **input_args,
                                                  style=input_style),
                                        html.Div('%',style={'fontSize':13,'display':'inline','marginLeft':5})
                                    ],hidden=True),
                                    html.Div(id='heart-default-percent',hidden=True)
                                ], style={'paddingRight':15,'paddingTop':1,'paddingBottom':1})
                            ]),

                            html.Tr([
                                html.Th(['Lung'],style={'paddingLeft':'10px'}),
                                html.Td([
                                    html.Div(id='lung_radio-container',children=[
                                        dcc.RadioItems(id = 'entry-method-lung',
                                                       **radio_args,
                                                      style={'display':'inline-block'})
                                    ]),
                                ], colSpan=2),

                                html.Td([
                                    html.Div(id='lung_input-container',children=[
                                        dcc.Input(id = 'lung-dose-input',
                                                  **input_args,
                                                  style=input_style),
                                        html.Div('%',style={'fontSize':13,'display':'inline','marginLeft':5})
                                    ],hidden=True),
                                    html.Div(id='lung-default-percent',hidden=True)
                                ], style={'paddingRight':15,'paddingTop':1,'paddingBottom':1})
                            ]),

                            html.Tr([
                                html.Th(['Esophagus'],style={'paddingLeft':'10px'}),
                                html.Td([
                                    html.Div(id='esoph_radio-container',children=[
                                        dcc.RadioItems(id = 'entry-method-esoph',**radio_args,style={'display':'inline-block'})
                                    ]),
                                ], colSpan=2),
                                html.Td([
                                    html.Div(id='esoph_input-container',children=[
                                        dcc.Input(id = 'esoph-dose-input',**input_args,style=input_style),
                                        html.Div('%',style={'fontSize':13,'display':'inline','marginLeft':5})
                                    ],hidden=True),
                                    html.Div(id='esoph-default-percent',hidden=True)
                                ], style={'paddingRight':15,'paddingTop':1,'paddingBottom':1})
                            ])
                        ], style={'fontSize':13})
                    ]),
                    # end table

                    html.Div(id='display-store-data',style={'display':'inline-block'})

            ], style={'width':500,'margin-left':10,'margin-right':50,'float':'right'}
             ),
            dcc.Store(id='dose-output-store'),
            dcc.Store(id='memory-store'),

        ], style={'width':1050, 'margin-left':50,'margin-top':5, 'display':'block'})

    ], style={'margin-top':40}

)


@app.callback([
    Output('display-selected-rsi','children'),
    Output('display-selected-dose','children')],
    [Input('rsi-slider','value'),
    Input('dose-slider','value')])

def update_slider_text(selected_rsi,selected_dose):
    return 'RSI: %.2f' %selected_rsi, 'Treatment Dose: {} (Gy)'.format(selected_dose)


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
    traces = [bar_rsi, dist_rsi, bar_rxdose, dist_rxdose]

    ticker_color = 'rgb(.95,.6,.2)'
    traces.append(go.Scatter(
        name='Selected RSI',
        xaxis='x2',yaxis='y2',
        line = {'color':ticker_color,'width':1.5,'dash':'dash'},
        y=np.linspace(0,6,50),
        x=np.full((50),rval))
    )
    traces.append(go.Scatter(
        name = 'RxRSI',
        xaxis='x1', yaxis='y1',
        line = {'color':ticker_color,'width':2.75},
        y=np.linspace(0,gard_range_y[1],50),
        x=np.full((50),rxdose_val),
        hoverinfo='none'
        )
    )
    traces.append(go.Scatter(
        name = 'Treatment Dose',
        xaxis='x1', yaxis='y1',
        line = {'color':'rgb(.95,.2,.1)','width':2.75},
        y=np.linspace(0,gard_range_y[1],50),
        x=np.full((50),dose_val),
        hoverinfo='none')
    )

    # rxrsi hoverlabel and kde shading #
    idx = np.argwhere(rxdose_kde[0]>=rxdose_val)[0][0]
    kde_yval_rxdose=rxdose_kde[1][idx]
    percentile = np.round(calc_cdf(rxdose_tcc,rxdose_val,bandwidth=.1)*100,1)
    display_rxrsi=np.int(np.round(rxdose_val,0))
    traces.append(go.Scatter(
        name='RxRSI',
        x=[rxdose_val],
        y=[kde_yval_rxdose],
        mode='markers',
        marker=dict(color=ticker_color,size=8),
        text='<b>{} Gy</b><br>{}%'.format(display_rxrsi,percentile),
        hoverinfo='text+name',
        hoverlabel=dict(font={'size':12,'color':'white'}),
        showlegend=False
    ))

    # dose hoverlabel #
    j = np.argwhere(rxdose_kde[0]>=dose_val)[0][0]
    kde_yval_dose=rxdose_kde[1][j]
    percentile_td = np.round(calc_cdf(rxdose_tcc,dose_val,bandwidth=.1)*100,1)
    traces.append(go.Scatter(
        name='Dose',
        x=[dose_val],
        y=[kde_yval_dose],
        xaxis='x1', yaxis='y1',
        mode='markers',
        marker=dict(color='rgb(.95,.2,.1)',size=8),
        text='<b>{} Gy</b><br>{}%'.format(dose_val,percentile_td),
        hoverinfo='text+name',
        hoverlabel=dict(font={'size':12}),
        showlegend=False
        )
    )

    xmin=np.min([rxdose_val,dose_val])
    xmax=np.max([rxdose_val,dose_val])
    k = np.argwhere((rxdose_kde[0]>=xmin) & (rxdose_kde[0]<=xmax)).flatten()
    xvals_fill = rxdose_kde[0][k]
    yvals_fill=np.full(shape=(len(k)),fill_value=gard_range_y[1])
    fill_color = 'rgba(200,230,255,.2)' if rxdose_val<dose_val else 'rgba(255,180,180,.2)'
    traces.append(
        go.Scatter(
            x=xvals_fill,
            y=yvals_fill,
            xaxis='x1', yaxis='y1',
            line=dict(color='rgb(250,250,250)',width=0),
            fill='tozeroy',
            fillcolor=fill_color,
            mode='lines',
            hoverinfo='none',
            showlegend=False
        )
    )

    return {
        'data':traces,
        'layout':layout_hists
    }
"""---------------------------------------------------------------------"""


""" *************** add rsi, gard, dose to output table *************** """
"""---------------------------------------------------------------------"""
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

    return '%.2f' %rval, '{} Gy'.format(dose_val), '%.0f Gy' % rxdose_val, gard_val
"""------------------------------------------------------------------"""


""" ********************* plot outcome curve *********************
------------------------------------------------------------------"""
@app.callback([Output('outcome-curves','figure'),
               Output('pefs-output-1','children'),
               Output('pefs-output-2','children'),
              Output('pefs-output-5','children')],
              [Input('rsi-slider','value'),
               Input('dose-slider','value'),
               Input('dose-output-store','data'),
               # Input('heart-dose-output','children'),
               # Input('lung-dose-output','children'),
               # Input('esoph-dose-output','children')
               ]
)
def update_outcome_figure(selected_rsi,selected_dose,data):   #hdose,ldose,edose):
    rval = selected_rsi
    alpha_val = (np.log(rval)+beta*n*(d**2))/(-n*d)
    rxdose_val = 33/(alpha_val+beta*d)
    dose_val = selected_dose
    gard_val = dose_val*(alpha_val+beta*d)
    G33 = True if (dose_val>=rxdose_val) else False

    traces = []
    tb_output = []
    normal_tissue_doses = np.array(data)

    if (np.any(normal_tissue_doses=='')) or (np.any(normal_tissue_doses==None)):
        all_doses_entered = False
    else:
        all_doses_entered = True
        hdose=data[0]
        ldose=data[1]
        edose=data[2]

    if all_doses_entered:
        pfs_array = pfs_gard33_man(t,G33,hdose,ldose,edose)
        traces.append(go.Scatter(
            name='Penalized-LC',
            line = {'color':'rgb(.8,.2,.1)'},
            x=t,
            y=pfs_array,
            visible=True,
            showlegend=True))
        tb_output.append(np.round(pfs_array[9],2))
        tb_output.append(np.round(pfs_array[19],2))
        tb_output.append(np.round(pfs_array[49],2))
        vis=True
    else:
        tb_output=['']*3
        vis=False

    traces.append(go.Scatter(
        name='Local Control',
        line = dict(color='rgb(.4,.1,.5)'),
        x=t,
        y=plc_gard33(t,G33),
        visible=False,
        showlegend=True))

    layout = layout_outcome_curves
    if all_doses_entered:
        layout['annotations']=None
    else:
        layout['annotations']=[dict(
            x=2.4,
            y=.6,
            xref='x',
            yref='y',
            text='<i>Enter Normal<br>  Tissue Doses</i>',
            showarrow=False,
            font = {'size':24,'color':'rgb(190,190,190)'},
            align='left'
        )]

    return [{'data':traces,'layout':layout}] + tb_output
"""------------------------------------------------------------------"""


""" ********************* add doses output table *********************"""
@app.callback([Output('heart-dose-output','children'),
               Output('lung-dose-output','children'),
               Output('esoph-dose-output','children'),
               Output('dose-output-store','data')],
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
    percent_total = [7.0,11.8,25.2]
    manual_percent= [hdose, ldose, edose]

    display_site_doses = []
    store_site_doses = []

    for i in range(len(entry_methods)):
        if (entry_methods[i]=='auto'):
            site_dose_i = selected_dose*percent_total[i]/100
            store_site_doses.append(site_dose_i)
            display_site_doses.append('%.1f Gy' % site_dose_i)
        elif (entry_methods[i] == 'manual') and (manual_percent[i]!=''):
            site_dose_i= selected_dose*manual_percent[i]/100
            store_site_doses.append(site_dose_i)
            display_site_doses.append('%.1f Gy' % site_dose_i)
        else:
            store_site_doses.append('')
            display_site_doses.append('')

    return display_site_doses + [store_site_doses]
"""--------------------------------------------------------"""

"""***************************** reset dose inputs ***************************"""
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

@app.callback([
               Output('heart_input-container','hidden'),
               Output('lung_input-container','hidden'),
               Output('esoph_input-container','hidden'),
               Output('heart-default-percent','hidden'),
               Output('lung-default-percent','hidden'),
               Output('esoph-default-percent','hidden'),
               Output('heart-default-percent','children'),
               Output('lung-default-percent','children'),
               Output('esoph-default-percent','children')],
              [Input('entry-method-heart','value'),
               Input('entry-method-lung','value'),
               Input('entry-method-esoph','value')]
)

def update_dose_input_editable(h_entry_method,l_entry_method,e_entry_method):
    entry_methods = [h_entry_method,l_entry_method,e_entry_method]
    inputs_disabled = []
    inputs_hidden = []
    auto_pp_hidden = []
    auto_pp_value = []
    defaults = ['7.0','11.8','25.2']
    for i in range(len(entry_methods)):
        if (entry_methods[i] == 'auto'):
            inputs_hidden.append(True)
            auto_pp_hidden.append(False)
            auto_pp_value.append('{}%'.format(defaults[i]))
        elif entry_methods[i] == 'manual':
            inputs_hidden.append(False)
            auto_pp_hidden.append(True)
            auto_pp_value.append('')
        elif entry_methods[i] == '':
            inputs_hidden.append(True)
            auto_pp_hidden.append(True)
            auto_pp_value.append('')

    return  inputs_hidden + auto_pp_hidden + auto_pp_value
"""----------------------------------------------------------------------"""


""" ********************* entry method-apply to all sites ********************* """

@app.callback(
    [Output('entry-method-heart','value'),
     Output('entry-method-lung','value'),
     Output('entry-method-esoph','value')],
    [Input('apply-all','value')],
    [State('entry-method-heart','value'),
     State('entry-method-lung','value'),
     State('entry-method-esoph','value')]
)

def update_all_dose_radios(entry_method_all,hval,lval,eval):

    if (entry_method_all == 'auto'):
        val = 'auto'
    elif (entry_method_all == 'manual'):
        val = 'manual'
    elif (entry_method_all=='cleared'):
        val = ''
    elif (entry_method_all=='initial'):
        raise PreventUpdate

    return [val]*3
"""----------------------------------------------------------------------"""


@app.callback([Output('apply-all','value'),
               Output('memory-store','data')],
              [
              Input('clear-button','n_clicks'),
              Input('entry-method-heart','value'),
              Input('entry-method-lung','value'),
              Input('entry-method-esoph','value')
              ],
              [State('apply-all','value'),
               State('memory-store','data')]
)
def  update_apply_all(nclicks,hval,lval,eval,all_val,data):

    if data==None:
        data={'clicks':0}
    else:
        data=data

    button_trigger=False
    if (nclicks!=None) and (data!=None):
        if nclicks>data['clicks']:
            button_trigger=True
            data['clicks']=data['clicks']+1
            new_all_val='cleared'

            return new_all_val, data

    all_on = True if ((all_val=='auto') or (all_val=='manual')) else False

    current_vals=np.array([hval,lval,eval])
    if (button_trigger==False) and (all_on==True):
        if np.all(current_vals!='') and np.all(current_vals!=None):
            if np.any(current_vals!=all_val):
                new_all_val='initial'
                return new_all_val, data

    if button_trigger==False & all_on == False:
        raise PreventUpdate


"""----------------------------------------------------------------"""

if __name__ == '__main__':
    app.run_server(debug=True)
