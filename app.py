
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

# bins_bins = list(np.arange(21,141,6))+[240]
# bins_bins = list(np.arange(22,131,4))+[240]
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
    height=550,
    margin=go.layout.Margin(
        l=40,
        b=80,
        t=80,
        r=130,
        pad=0.1),
    legend=dict(x=1.05,y=.2,
        font={'size':11},
        bgcolor= 'rgb(255,255,255)',
        bordercolor=axes_color,
        borderwidth=1),
    paper_bgcolor='rgb(.97,.97,.97)',
    title=go.layout.Title(text='[Title Here]',x=0.1,yref='container',y=.91)
)

layout_outcome_curves = go.Layout(
    xaxis=dict(
        title='Time (yrs)',
        range=[0, 5.2],
        showgrid = False,
        tickmode='array',
        tickvals = list(range(6)),
        ticktext= list('012345'),
        ticklen=4),
    yaxis=dict(
        title = 'Probability',
        range=[0, 1],
        showgrid=False),
    width=400,
    height=550,
    margin=go.layout.Margin(
        l=60,
        r=40,
        b=80,
        t=80,
        pad=.1),
    legend=dict(x=.1,y=.1),
    paper_bgcolor='rgb(.9,.9,.9)',
    # plot_bgcolor = 'rgb(250,250,250)',
    # paper_bgcolor='rgb(20,60,110)',
    # font=dict(color='rgb(230,230,230)'),
    title=go.layout.Title(text='[Title Here...]',x=0.1,yref='container',y=.91)
)

#%%

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table

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
                style={'display':'inline-block'}),
            html.Div(
                dcc.Graph(id='outcome-curves'),
                style={'display':'inline-block'}),
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
                                html.Tr([html.Th('RxDose',style={'paddingLeft':10}),html.Td(id='rxdose-output',style={'textAlign':'center'})]),
                                html.Tr([html.Th('GARD (Tx)',style={'paddingLeft':10}),html.Td(id='gard-output',style={'textAlign':'center'})]),

                                html.Tr([html.Th('Heart',style={'paddingLeft':10}), html.Td(id='heart-dose-output',style={'textAlign':'center'})]),
                                html.Tr([html.Th('Lung',style={'paddingLeft':10}), html.Td(id='lung-dose-output',style={'textAlign':'center'})]),
                                html.Tr([html.Th('Esophagus',style={'paddingLeft':10}), html.Td(id='esoph-dose-output',style={'textAlign':'center'})])
                            ],style={'fontSize':13})
                        ], style={'margin-top':100}
                    )
                ], style={'float':'right','height':550,'margin-right':30}
            )
        ], style = {'margin-left':40, 'width':1200,'display':'block'}),

        # bottom row
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
                ], style = {'fontSize':14})
            ], style={'width':400,'display':'inline-block'}),

            # bottom right
            html.Div(children=[
                    html.H6('Normal Tissue Dose',style={'fontSize':20}),
                    html.Div(id='radio-container', children=[
                        dcc.RadioItems(
                            id = 'dose-entry-method',
                            options=[{'label': 'Use Total Dose', 'value': 'auto'},
                                {'label': 'Manually Enter', 'value': 'manual'}],
                            labelStyle={'display': 'inline-block','fontSize':14}
                        )
                    ]),
                    html.Div(id='inputs-container', children=[
                        dcc.Input(
                            id = 'heart-dose',
                            placeholder='Heart dose (Gy)',
                            type='number',
                            value=''),
                        dcc.Input(
                            id = 'lung-dose',
                            placeholder='Lung dose (Gy)',
                            type='number',
                            value=''),
                        dcc.Input(
                            id = 'esoph-dose',
                            placeholder='Esophagus dose (Gy)',
                            type='number',
                            value='')
                    ], style={'display':'inline-block','fontSize':14,'margin-top':10}),
                    html.Div(
                        html.Button(id='submit-button',
                            n_clicks=0,
                            children='Submit'),
                        style={'margin-top':10}
                    ),
                    dcc.ConfirmDialog(
                        id='confirm',
                        message='Values not entered for all tissue sites'
                    )
            ], style={'width':250,'margin-left':20,'float':'right'})

        ], style={'width':700, 'margin-left':60,'margin-top':10, 'display':'block'})

    ], style={'margin-top':40}

)

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


@app.callback(
    [Output('outcome-curves','figure'),
    Output('confirm', 'displayed')],
    [Input('submit-button', 'n_clicks')],
    [State('rsi-slider','value'),
    State('dose-slider','value'),
    State('heart-dose','value'),
    State('lung-dose','value'),
    State('esoph-dose','value'),
    State('dose-entry-method','value')]
)

def update_outcome_figure(n_clicks, selected_rsi,selected_dose, hdose,ldose,edose,entry_method):
    rval = selected_rsi
    alpha_val = (np.log(rval)+beta*n*(d**2))/(-n*d)
    rxdose_val = 33/(alpha_val+beta*d)
    dose_val = selected_dose
    gard_val = dose_val*(alpha_val+beta*d)
    G33 = True if (dose_val>=rxdose_val) else False

    traces = []

    if (n_clicks > 0) & (entry_method=='manual'):
        normal_tissue_doses = np.array([hdose,ldose,edose])
        display_warning = np.any(normal_tissue_doses=='')
    else:
        display_warning = False

    if entry_method == 'auto':
        penalized_version = pfs_gard33(t,G33,dose_val)
        traces.append(go.Scatter(
            name='penalized-GARD33',
            line = {'color':'rgb(.4,.1,.5)'},
            x=t,
            y=penalized_version,
            visible=True))
    elif (entry_method == 'manual') & (display_warning==False):
            penalized_version= pfs_gard33_man(t,G33,hdose,ldose,edose)
            traces.append(go.Scatter(
                name='penalized-GARD33',
                line = {'color':'rgb(.4,.1,.5)'},
                x=t,
                y=penalized_version,
                visible=True))

    traces.append(go.Scatter(
        name='local-control-gard33',
        line = dict(color='rgb(.8,.1,.1)'),
        x=t,
        y=plc_gard33(t,G33),
        visible=True))

    return {'data':traces,'layout':layout_outcome_curves}, display_warning

#%% rsi and gard continuous outcome curves
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
#%%


@app.callback([Output('rsi-output','children'),
               Output('dose-output','children'),
               Output('rxdose-output','children'),
               Output('gard-output','children'),
               Output('heart-dose-output','children'),
               Output('lung-dose-output','children'),
               Output('esoph-dose-output','children')],
               [Input('rsi-slider','value'),
                Input('dose-slider','value'),
                Input('dose-entry-method','value'),
                Input('submit-button', 'n_clicks')],
                [State('heart-dose','value'),
                State('lung-dose','value'),
                State('esoph-dose','value')]
)

def update_output_table(selected_rsi,selected_dose, entry_method, n_clicks, hdose, ldose, edose):
    rval = np.round(selected_rsi,2)
    alpha_val = (np.log(rval)+beta*n*(d**2))/(-n*d)
    rxdose_val = np.round(33/(alpha_val+beta*d),0)
    dose_val = selected_dose
    gard_val = np.round(dose_val*(alpha_val+beta*d),0)
    G33 = True if (dose_val>=rxdose_val) else False

    if entry_method == 'auto':
        h = np.round(selected_dose/14,1)
        l = np.round(selected_dose/8.5,1)
        e = np.round(selected_dose/4,1)
    elif entry_method == 'manual':
        h = np.round(hdose,1)
        l = np.round(ldose,1)
        e = np.round(edose,1)
    else:
        h,l,e = ['_']*3

    return rval, dose_val, rxdose_val, gard_val, '{} Gy'.format(h),'{} Gy'.format(l),'{} Gy'.format(e)




@app.callback([
    Output('display-selected-rsi','children'),
    Output('display-selected-dose','children')],
    [Input('rsi-slider','value'),
    Input('dose-slider','value')])

def update_slider_text(selected_rsi,selected_dose):
    return 'RSI: {}'.format(selected_rsi), 'Total Dose: {}'.format(selected_dose)


@app.callback(
    [Output('heart-dose','disabled'),
    Output('lung-dose','disabled'),
    Output('esoph-dose','disabled')],
    [Input('dose-entry-method','value')]
)

def update_dose_entry_editable(entry_method):
    if entry_method == 'auto':
        disabled = True
    elif entry_method == 'manual':
        disabled = False
    else:
        disabled = False
    return [disabled]*3





    # table_trace = go.Table(
    #     header=dict(
    #         values=['','Outcomes']
    #     ),
    #     cells=dict(values=[['RSI','Dose','GARD'],
    #                        [rval,dose_val,gard_val]]
    #     )
    # )
    # layout_table = dict(width=250)
    #
    # traces=[table_trace]
    #
    # return {'data':traces,'layout':layout_table}



# @app.callback(
#     [Output('heart-dose-output','children'),
#     Output('lung-dose-output','children'),
#     Output('esoph-dose-output','children')],
#     [Input('dose-slider','value'),
#     Input('dose-entry-method','value'),
#     Input('submit-button', 'n_clicks')],
#     [State('heart-dose','value'),
#     State('lung-dose','value'),
#     State('esoph-dose','value')])
#
# def update_site_dose_text(total_dose, entry_method, n_clicks, hdose, ldose, edose):
#     if entry_method == 'auto':
#         h = np.round(total_dose/14,1)
#         l = np.round(total_dose/8.5,1)
#         e = np.round(total_dose/4,1)
#         return ('{} Gy'.format(h),'{} Gy'.format(l),'{} Gy'.format(e))
#     elif entry_method == 'manual':
#         return ('{} Gy'.format(hdose),'{} Gy'.format(ldose),'{} Gy'.format(edose))

if __name__ == '__main__':
    app.run_server(debug=True)
