#!/usr/bin/env python
# coding: utf-8

# ## importing pkgs

# In[1]:


import numpy as np
exp = np.exp
arange = np.arange
ln = np.log
from datetime import *

import matplotlib.pyplot as plt
from matplotlib import patches

import plotly.plotly as py
import plotly.graph_objs as go

from scipy.stats import norm
from scipy import interpolate as interp
pdf = norm.pdf
cdf = norm.cdf
ppf = norm.ppf

from scipy import stats
from scipy import special
erf = special.erf

import pandas as pd
import palettable
import seaborn as sns
cp = sns.color_palette()

from lifelines import KaplanMeierFitter
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn import mixture
from sklearn import preprocessing


# In[3]:


lc_file = '/Users/geoffreysedor/Documents/lungcohortdata.xlsx'
lc_df = pd.read_excel('/Users/geoffreysedor/Documents/lungcohortdata.xlsx')

lc_df.rename(columns = {'SF2':'rsi','GAD':'gardtx','Deathyn':'OS_status','OS':'OS_time'},inplace=True)

lc_df['td'] = lc_df['recurr_lasteval_date'] - lc_df['SurgDate']
lc_df['td_yrs'] = lc_df['td']/np.timedelta64(365,'D')
lc_df['gard33'] = lc_df['gardtx']>33

beta = 0.05
lc_df['new_dose'] = 33/(lc_df.alpha+beta*lc_df.fxsize)
lc_df['new_dose_5070'] = np.maximum(np.minimum(lc_df.new_dose,70),50)
lc_df['rxrsi_to_70'] = np.minimum(lc_df.new_dose,70)

rsi_file = '/Users/geoffreysedor/Documents/GARD_data.xlsx'
rsi_df = pd.read_excel(rsi_file)

nsclc = pd.DataFrame()
nsclc['rsi'] = rsi_df.RSI[(rsi_df.SOO=='Lung Adeno') | (rsi_df.SOO=='Lung Squamous')]


# ## kde / cdf functions

# In[4]:


def create_kde(array, bandwidth=None):
    """ calculating KDE and CDF using scipy """
    if bandwidth == None:
        bw = 'scott'
    else:
        bw = bandwidth
    kde = stats.gaussian_kde(dataset=array,bw_method=bw)
    
    num_test_points=200
    x = np.linspace(0,np.max(array)*1.2,num_test_points)
    kdens=kde.pdf(x)
    
    cdf=np.zeros(shape=num_test_points)
    for i in range(num_test_points):
        cdf[i] = kde.integrate_box_1d(low=0,high=x[i])
        
    return x,kdens,cdf


def calc_cdf(array,var,bandwidth=None):
    if bandwidth == None:
        bw = 1.2*array.std()*np.power(array.size,-1/5)
    else:
        bw = bandwidth
    kde=stats.gaussian_kde(dataset=array,bw_method=bw)
    return kde.integrate_box_1d(low=0,high=var)



# ## NTCP adjustments:

# In[5]:


def prob_pneumonitis(dose_h, dose_l = 0):
    fx_to_lung = 8.5
    MLDh = dose_h/fx_to_lung
    MLDl = dose_l/fx_to_lung
    b0 = -3.87
    b1 = 0.126
    prob_h = np.exp(b0+b1*MLDh)/(1+np.exp(b0+b1*MLDh))
    if np.all(dose_l==0):
        prob_l = np.exp(b0+b1*MLDl)/(1+np.exp(b0+b1*MLDl))
    else:
        prob_l = np.zeros(dose_l.size)
    return prob_h, prob_l

def pneumonitis_RR(dose_h, dose_l = 0):
    prob_h, prob_l = prob_pneumonitis(dose_h,dose_l)
    rr = (1+prob_h)/(1+prob_l)
#     rr = (1-prob_l)/(1-prob_h)
    return np.round(rr,3)

def prob_esoph(dose_h, dose_l = 0):
    EUD1 = dose_h/4
    EUD2 = dose_l/4
    TD50 = 47
    m = 0.36
    t1 = (EUD1-TD50)/(m*TD50)
    t2 = (EUD2-TD50)/(m*TD50)
    y = (erf(t1)-erf(t2))/2
    return np.round(y,3)

def esophagitis_RR(dose_h, dose_l = 0):
    
    prob_h = prob_esoph(dose_h)
    prob_l = prob_esoph(dose_l)
    rr = (1+prob_h)/(1+prob_l)
#     rr = (1-prob_l)/(1-prob_h)
#     rr = prob_h/prob_l
    return np.round(rr,4)

def cardiac_event_RR(dose_h, dose_l = 0):
    
    dose_diff = dose_h-dose_l
    delta_dose_heart = np.mean((dose_h-dose_l)/14)
    risk_per_gy = np.zeros(len(t))
    risk_per_gy = np.where(t<5,16.3,0)
    risk_per_gy = np.where(5<=t,15.5,risk_per_gy)
    risk_per_gy = np.where(t>=10,1.2,risk_per_gy)
    #risk_per_gy = 7.4    #  percent increased risk
    cardiac_event_rr = 1+risk_per_gy*delta_dose_heart/100
#     cardiac_event_rr = 1+.074*delta_dose_heart
    
    return cardiac_event_rr


# In[7]:


def H_esoph(dose_h, dose_l = 0, CI = None):
    EUD1 = dose_h/4
    EUD2 = dose_l/4
    TD50 = 47
    TD50l = 60
    TD50u = 41
    
    m = 0.36
    mu = 0.55
    ml = 0.25
    
    if CI == 'upper':
        TD50 = TD50u
        m = mu
    elif CI == 'lower':
        TD50 = TD50l
        m = ml
        
    t1 = (EUD1-TD50)/(m*TD50)
    t2 = (EUD2-TD50)/(m*TD50)
    y = (erf(t1)-erf(t2))/2
    return np.round(y,3)

def H_lung(dose_h, dose_l = 0, CI = None):
    fx_to_lung = 1/8.5
    MLDh = dose_h*fx_to_lung
    MLDl = dose_l*fx_to_lung
    b0 = -3.87
    b0u = -3.33
    b0l =  -4.49
    
    b1 = 0.126
    b1u = .153
    b1l = .100
    
    # TD50 = 30.75 [28.7–33.9] Gy
    if CI == 'upper':
        b0 = b0u
        b1 = b1u
    elif CI == 'lower':
        b0 = b0l
        b1 = b1l
    
    prob_h = np.exp(b0+b1*MLDh)/(1+np.exp(b0+b1*MLDh))
    if np.all(dose_l==0):
        prob_l = np.exp(b0+b1*MLDl)/(1+np.exp(b0+b1*MLDl))
    else:
        prob_l = np.zeros(dose_l.size)
    return prob_h - prob_l

def RH_cardiac(d,CI=None):
    d=d/14
    if CI =='upper':
        RR = 1 + .643*d
    elif CI == 'lower':
        RR = 1 + .03*d
    else:
        RR = 1 + .163*d
    return RR

def RH_cardiac_20yr(d,CI=None):
    d=d/14
    if CI =='upper':
        RR = 1 + .074*d
    elif CI == 'lower':
        RR = 1 + .029*d
    else:
        RR = 1 + .145*d
    return RR


# In[11]:


d = np.arange(10,80,.1)
p = 100*(prob_pneumonitis(d)[0]-prob_pneumonitis(d)[1])
p2 = 100*H_lung(d,CI='upper')
p3 = 100*H_lung(d,CI='lower')
e = 100*prob_esoph(d)
e2 = 100*H_esoph(d,CI='lower')
e3 = 100*H_esoph(d,CI='upper')
c = .074*d/14

fig,ax = plt.subplots()

ax.plot(d,p2,d,p3)
ax.plot(d,e2,d,e3)

ax.set_ylim(-.5,10)
ax.grid(True)
ax.axhline(y=0,c='k',lw=1)


# In[12]:


d = np.arange(10,80,.1)
p = 100*(prob_pneumonitis(d)[0]-prob_pneumonitis(d)[1])
e = 100*prob_esoph(d)
c = .074*d/14

fig,ax = plt.subplots()
ax.plot(d,p)
ax.plot(d,e)

ax.set_ylim(-.5,5)
ax.grid(True)
ax.axhline(y=0,c='k',lw=1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
d=d.reshape(-1,1)
lr.fit(d,p)
print(lr.score(d,p))

d2 = np.arange(40,80,.1)
d2=d2.reshape(-1,1)
e2 = 100*prob_esoph(d2)
lr2 = LinearRegression()
lr2.fit(d2,e2)
print(lr2.score(d2,e2))

print(lr.intercept_,lr.coef_)
print(lr2.intercept_,lr2.coef_)

def risk_p(dose):
    d = np.maximum(dose-18,0)
    r = d*0.068
    return r
def risk_e(dose):
    d = np.maximum(dose-40,0)
    r = 0.026*d
    return r

ax.plot(d,risk_p(d))
ax.plot(d,risk_e(d))


# In[58]:



from scipy.stats import norm

def H_esoph(dose):
    y=norm.cdf(dose/4,51,14)
    return y


def H_lung(dose_h, dose_l = 0, CI = None):
    fx_to_lung = 1/8.5
    MLDh = dose_h*fx_to_lung
    MLDl = dose_l*fx_to_lung
    b0 = -3.87
    b0u = -3.33
    b0l =  -4.49
    
    b1 = 0.126
    b1u = .153
    b1l = .100
    
    # TD50 = 30.75 [28.7–33.9] Gy
    if CI == 'upper':
        b0 = b0u
        b1 = b1u
    elif CI == 'lower':
        b0 = b0l
        b1 = b1l
    
    prob_h = np.exp(b0+b1*MLDh)/(1+np.exp(b0+b1*MLDh))
    if np.all(dose_l==0):
        prob_l = np.exp(b0+b1*MLDl)/(1+np.exp(b0+b1*MLDl))
    else:
        prob_l = np.zeros(dose_l.size)
    return prob_h - prob_l

def risk_p(dose):
    d = np.maximum(dose-18,0)
    r = d*0.068
    return r
def risk_e(dose):
    d = np.maximum(dose-40,0)
    r = 0.026*d
    return r


# ## TCC analysis

# In[15]:


r = nsclc.rsi

d = 2
beta = 0.05

# for SF2 alpha
n = 1
alpha_tcc = (np.log(r)+beta*n*(d**2))/(-n*d)
rxdose_tcc = 33/(alpha_tcc+beta*d)
rxdose_tcc=rxdose_tcc.values


print('TCC RSI min = {}\n'.format(r.min()))
print('TCC RSI max = {}\n'.format(r.max()))


# In[16]:


cohort_rsi = lc_df.rsi.values
tcc_rsi = nsclc.rsi.values

from scipy.stats import ks_2samp

print(ks_2samp(cohort_rsi, tcc_rsi))

from scipy.stats import anderson_ksamp

print(anderson_ksamp([cohort_rsi,tcc_rsi]))


# In[17]:


n1=tcc_rsi.size
n2=cohort_rsi.size
c_alpha=1.36
1.95*np.sqrt((n1+n2)/(n1*n2))


# In[18]:


ks_2samp(cohort_rsi.round(2), tcc_rsi.round(2))


# In[19]:


anderson_ksamp([cohort_rsi.round(2), tcc_rsi.round(2)])


# In[29]:


def create_kde2(array, bandwidth=None):
    """ calculating KDE and CDF using scipy """
    if bandwidth == None:
        bw = 'scott'
    else:
        bw = bandwidth
    kde = stats.gaussian_kde(dataset=array,bw_method=bw)
    
    num_test_points=200
    x = np.linspace(0,np.max(array)*1.2,num_test_points)
    kdens=kde.pdf(x)
    
    cdf=np.zeros(shape=num_test_points)
    for i in range(num_test_points):
        cdf[i] = kde.integrate_box_1d(low=0,high=x[i])
        
    return x,kdens,cdf,kde

tcc_kde = create_kde(tcc_rsi,bandwidth=.3)
cohort_kde = create_kde(cohort_rsi[1:],bandwidth=.4)


# In[27]:


# ks_2samp(tcc_kde[1],cohort_kde[1] )

ks_2samp(tcc_kde[1],cohort_rsi)


# In[31]:


import palettable
from matplotlib.path import Path
from matplotlib.patches import PathPatch

tcc_kde = create_kde(tcc_rsi,bandwidth=.3)
cohort_kde = create_kde(cohort_rsi[1:],bandwidth=.4)

col_map2 = palettable.cartocolors.sequential.agGrnYl_5.mpl_colormap
col_map2 = palettable.matplotlib.Viridis_20.mpl_colormap

fig, axes = plt.subplots(nrows=2,ncols=1,figsize=(6,6), sharex=False)

e1 = 1
e2 = .5

ax = axes[0]

x = tcc_kde[0]
kdens = tcc_kde[1]
a = np.concatenate((x,np.flip(x)))
b = np.concatenate((kdens,np.flip(-1*kdens)))
path = Path(np.array([a,b]).transpose())
p = PathPatch(path,fill=False,lw=.1)
ax.add_patch(p)
ax.hlines(y=0,xmin=0,xmax=1,lw=.5,color=(.2,.2,.2))

ax.plot(x,kdens,lw=.5,c='k')
ax.plot(x,-kdens,lw=.5,c='k')

X,Y=np.meshgrid(kdens,np.full(kdens.size,1))
X = np.power(X,e1)
ax.imshow(X,origin='lower',extent = [x.min(),x.max(),-kdens.max(),kdens.max()],
          aspect='auto',clip_path=p,clip_on=True,cmap=col_map2,alpha=.9,interpolation='bicubic')

ax.set_ybound(-5,5)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlim(0,.8)
ax.set_ylabel('TCC Pts')

"""lower part"""
ax = axes[1]

x = cohort_kde[0]
kdens = cohort_kde[1]

a = np.concatenate((x,np.flip(x)))
b = np.concatenate((kdens,np.flip(-1*kdens)))
path = Path(np.array([a,b]).transpose())
p = PathPatch(path,fill=False,lw=.1)
ax.add_patch(p)
ax.hlines(y=0,xmin=0,xmax=1,lw=.5,color=(.2,.2,.2))
ax.plot(x,kdens,lw=.5,c='k')
ax.plot(x,-kdens,lw=.5,c='k')

X,Y=np.meshgrid(kdens,np.full(kdens.size,1))
X = np.power(X,e2)
ax.imshow(X,origin='lower',extent = [x.min(),x.max(),-kdens.max(),kdens.max()],
          aspect='auto',clip_path=p,clip_on=True,cmap=col_map2,alpha=.9,interpolation='bicubic')

ax.set_ybound(-5.5,5.5)
ax.set_yticks([])
ax.set_xlim(0,.8)
ax.set_ylabel('60 pt')
ax.set_xlabel('RSI')

# plt.savefig('/Users/geoffreysedor/Downloads/'+'viol_rsi5', dpi =300)


# In[33]:


##### r = nsclc.rsi

d = 2
beta = 0.05

# for SF2 alpha
n = 1
alpha_tcc = (np.log(r)+beta*n*(d**2))/(-n*d)
rxdose_tcc = 33/(alpha_tcc+beta*d)
rxdose_tcc=rxdose_tcc.values

""" plotting histograms """
fig, ax = plt.subplots(figsize=(8,6))
binlist=list(np.arange(0,150,2))+[300]

""" <60 range """
xdata = rxdose_tcc[np.where(rxdose_tcc<60)]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.6,#ec = 'k',
        color=cp[0],  
        weights = wts)

""" 60-74 range """
xdata = rxdose_tcc[np.where((rxdose_tcc>60)&(rxdose_tcc<74))]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.8,#ec = 'k',
        color=(.4,.4,.4),  
        weights = wts,zorder=5)

""" >74 range """
xdata = rxdose_tcc[np.where((rxdose_tcc>74))]  #&(rxdose_tcc<80))]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.7,#ec = 'k',
        color=cp[3],  
        weights = wts)

rxdose_kde = create_kde(rxdose_tcc,bandwidth=.3)

ppf_60 = np.where(rxdose_kde[1]<60)

ax.plot(rxdose_kde[0], rxdose_kde[1] , c=(.2,.2,.3),lw=1,ls='--',label = 'KDE')

ax.set_xlim(-2,130)
ax.set_yticks([])
ax.set_xlabel('RxRSI for TCC Lung')

#plt.savefig('/Users/geoffreysedor/Downloads/'+'TCC_lung_dist2', dpi =300)


# ## fig 1

# In[34]:


from matplotlib import patches
from matplotlib import path
Path=path.Path

def bracket(xi, y, dy=.1, dx = .04,tail=.1):

    yi = y - dy/2
    xf = xi+dx
    yf = yi+dy
    vertices = [(xi,yi),(xf,yi),(xf,yf),(xi,yf)]+[(xf,y),(xf+tail,y)]
    codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.MOVETO] + [Path.LINETO]
    return Path(vertices,codes)

def hbracket(x, yi, dx=.1, dy = .04,tail=.1):

    xi = x - dx/2
    xf = xi+dx
    yf = yi-dy
    vertices = [(xi,yi),(xi,yf),(xf,yf),(xf,yi)]+[(x,yf),(x,yf-tail)]
    codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.MOVETO] + [Path.LINETO]
    return Path(vertices,codes)

def double_arrow(x,y,length,orient,endlength=.04,r=10):
    l=length
    if orient == 'horz':
        x1= x - l/2
        x2 = x + l/2
        el = endlength/2
        vertices = [(x1,y),(x2,y)]+[(x1+l/r,y+el),(x1,y),(x1+l/r,y-el)]+[(x2-l/r,y+el),(x2,y),(x2-l/r,y-el)]
    else:
        y1= y - l/2
        y2 = y + l/2
        el = endlength/2
        vertices = [(x,y1),(x,y2)]+[(x-el,y1+l/r),(x,y1),(x+el,y1+l/r)]+[(x+el,y2-l/r),(x,y2),(x-el,y2-l/r)]
    codes = [Path.MOVETO,Path.LINETO]+[Path.MOVETO]+[Path.LINETO]*2+[Path.MOVETO]+[Path.LINETO]*2
    return Path(vertices,codes)


# In[35]:


div_cmap = sns.light_palette((0,.5,.8),n_colors=20)#as_cmap=True)
#sns.palplot(div_cmap, size = .8)

colors = [(0,.5,.8),(.98,.98,.98),(.7,.1,.1)]
# sns.palplot(sns.blend_palette(colors,n_colors=20))

colmap=sns.blend_palette(colors,as_cmap=True)


# In[37]:



fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(18,6))

axes[0].set_title('(A)', loc='left')
axes[1].set_title('(B)', loc='left')
axes[2].set_title('(C)', loc='left')

ax=axes[0]

x, k, c = create_kde(r)
ax.plot(x,k)

bins=np.arange(0,1,.04)
hist = np.histogram(r,bins=bins,density=True)

bar_width = (hist[1][1]-hist[1][0])*.7
ax.bar(hist[1][:-1],hist[0],width=bar_width,alpha=.6,color=(.6,.6,.6))
ax.set_yticks([])

"""-----------------------------------------------------------------------------------------------"""

ax = axes[1]

x = lc_df.new_dose_5070.values
x.sort()
range60 = range(1,61)
x2 = lc_df.new_dose.values
x2.sort()
dose_5070 = lc_df.new_dose_5070.sort_values()
full70 = np.full(len(x),70)

ax.scatter(range60,x2, s = 80, c=x2,cmap=colmap,edgecolors='k',zorder=10) #label = 'RxRSI > 70')
ax.scatter(range60,x,edgecolor = 'k',facecolor='white', marker = 'o', s = 60, zorder = 5, label = 'RxRSI scaled\nto 50-70')

ax.hlines(y = [50,70],xmin = [-2,-2],xmax=[62,62], color = 'k',lw=1.5,zorder=0)
ax.fill_between([-2,62],70,50, color = (.95,.95,.95),alpha=.2)

j = np.where(x2<50)[0][-1]
k = np.where(x2>70)[0][0]
ax.vlines(range60[k:],ymin = full70[k:], ymax = x2[k:], lw = .5, linestyle = '--')
ax.vlines(x = range60[:j], ymin = x2[:j], ymax = np.full(j,50), lw = .5, linestyle = '--')

ax.set_xticklabels('')
ax.set_ylim((10,100))
ax.set_xlim(-1,61)
ax.set_ylabel('RxRSI (Gy)')
ax.set_xlabel('Patient IDs')
ax.set_xticks([])

"""-------------------------------------------------------------------------------"""

ax=axes[2]

r = nsclc.rsi
d = 2
beta = 0.05

# for SF2 alpha
n = 1
alpha_tcc = (np.log(r)+beta*n*(d**2))/(-n*d)
rxdose_tcc = 33/(alpha_tcc+beta*d)
rxdose_tcc=rxdose_tcc.values

""" plotting histograms """

binlist=list(np.arange(0,150,2))+[300]

""" <60 range """
xdata = rxdose_tcc[np.where(rxdose_tcc<60)]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.6,#ec = 'k',
        color=cp[0],  
        weights = wts)
""" 60-74 range """
xdata = rxdose_tcc[np.where((rxdose_tcc>60)&(rxdose_tcc<74))]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.8,#ec = 'k',
        color=(.4,.4,.4),  
        weights = wts,zorder=5)
""" >74 range """
xdata = rxdose_tcc[np.where((rxdose_tcc>74))]  #&(rxdose_tcc<80))]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.7,#ec = 'k',
        color=cp[3],  
        weights = wts)

rxdose_kde = create_kde(rxdose_tcc,bandwidth=.28)
ax.plot(rxdose_kde[0], rxdose_kde[1] , c=(.2,.2,.3),lw=1,ls='--',label = 'KDE')

ax.set_xlim(-2,130)
ax.set_yticks([])
ax.set_xlabel('RxRSI for TCC Lung')

fig.subplots_adjust(left=.06, right=.95, wspace=.25)

#plt.savefig('/Users/geoffreysedor/Documents/'+'fig1_dec09', dpi =300)


# ## fig 2

# In[38]:


d = np.arange(0,80,.1)
mean = 45
std = 15
tcp = cdf(d, mean, std)
c1 = (.1,.55,.7)
c2 = (.9,.4,.1)
tcp_pdf = pdf(d, mean, std)

r = nsclc.rsi
d = 2
beta = 0.05
# for SF2 alpha
n = 1
alpha_tcc = (np.log(r)+beta*n*(d**2))/(-n*d)
rxdose_tcc = 33/(alpha_tcc+beta*d)
rxdose_tcc=rxdose_tcc.values

x,rxdose_pdf,rxdose_cdf = create_kde(rxdose_tcc,bandwidth=.28)

e = H_esoph(x)
p = H_lung(x)
MCE_base = 0.01
c = MCE_base*(1+.16*x/13)-.01

fig, axes = plt.subplots(figsize=(20,6),nrows=1,ncols=3,sharex=False)

"""---------------------------------- first part ----------------------------------"""

ax=axes[0]

ax.plot(x, rxdose_cdf,color=c1,label='TCP')

y=rxdose_pdf*35*.92
ax.fill_between(x,y,color=c1,alpha=.1)
ax.plot(x,y,color=c1,lw=1,alpha=.4)

ax.axvline(60,c=(.2,.2,.2),lw=2,ls='--')
ax.axvline(74,c=(.2,.2,.2),lw=2,ls='--')

ax.set_xlim(0,130)
ax.set_ylim(0,1.02)
ax.set_ylabel('TCP')
ax.set_xticks([0,20,40,60,80,100,120])
ax.set_xlabel('Dose (Gy)')

ax.set_title('(A)',loc='left',pad=12)
ax.legend()

"""-------------------------- second  plot ----------------------------------"""

ax=axes[1]
ax.plot(x,e*100,color=(.7,.1,.1),label='Esophagitis')
ax.plot(x,p*100,color=cp[1],label='Pneumonitis')
ax.plot(x,c*100,color=cp[4],label='Major Cardiac Event')

ax.axvline(60,c=(.2,.2,.2),lw=2,ls='--')
ax.axvline(74,c=(.2,.2,.2),lw=2,ls='--')

eq_string = r'$HR_{cum} = \Sigma H_{i}(d)$'
ax.text(.2,.5, eq_string, transform=ax.transAxes)

yticks=[0,2,4,6,8,10]
ax.set_yticks(yticks)
ax.set_yticklabels(['0%','2%','4%','6%','8%','10%'])
ax.set_ylim(0,8.5)
ax.set_xticks([0,20,40,60,74,80])
ax.set_xlim(0,80)
ax.set_xlabel('Dose (Gy)')
ax.set_ylabel('NTCP')

ax.set_title('(B)',loc='left',pad=12)
ax.legend()
    
######################################    
"""-------------------------- third  plot ----------------------------------"""
ax=axes[2]


ax.plot(x, rxdose_cdf,color=c1,label='TCP')

y=rxdose_pdf*35*.92
ax.fill_between(x,y,color=c1,alpha=.1)
ax.plot(x,y,color=c1,lw=1,alpha=.4)

y2=np.power(rxdose_cdf*(1-p)*(1-e),1.16)
ax.plot(x[:85],y2[:85],color=c2,label='Adjusted TCP')

y3=np.gradient(y2)*35*.82
ax.fill_between(x,y3,color=c2,alpha=.2)
ax.plot(x,y3,color=c2,lw=1,alpha=.4)

ax.axvline(60,c=(.2,.2,.2),lw=2,ls='--')
ax.axvline(74,c=(.2,.2,.2),lw=2,ls='--')

ax.set_xlim(0,115)
ax.set_ylim(0,1.02)
ax.set_ylabel('TCP')
ax.set_xticks([0,20,40,60,80,100])
ax.set_xlabel('Dose (Gy)')

ax.set_title('(C)',loc='left',pad=12)
ax.legend()

ax.set_xlabel('RxRSI for TCC Lung')

    
fig.subplots_adjust(left=.06,right=.97,bottom=.15,top=.9,wspace=.25)
#plt.savefig('/Users/geoffreysedor/Documents/' + 'fig2_dec09',dpi=300)


# ## fig 4

# ### tcc hist with percentages (first panel)

# In[105]:


r = nsclc.rsi

d = 2
beta = 0.05

# for SF2 alpha
n = 1
alpha_tcc = (np.log(r)+beta*n*(d**2))/(-n*d)
rxdose_tcc = 33/(alpha_tcc+beta*d)
rxdose_tcc=rxdose_tcc.values

""" plotting histograms """
fig, ax = plt.subplots(figsize=(8,6))
binlist=list(np.arange(0,150,2))+[300]

""" <60 range """
xdata = rxdose_tcc[np.where(rxdose_tcc<60)]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.6,#ec = 'k',
        color=cp[0],  
        weights = wts)

""" 60-74 range """
xdata = rxdose_tcc[np.where((rxdose_tcc>60)&(rxdose_tcc<74))]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.8,#ec = 'k',
        color=(.4,.4,.4),  
        weights = wts,zorder=5)

""" >74 range """
xdata = rxdose_tcc[np.where((rxdose_tcc>74))]  #&(rxdose_tcc<80))]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.7,#ec = 'k',
        color=cp[3],  
        weights = wts)

rxdose_kde = create_kde(rxdose_tcc,bandwidth=.15)

pp_60 = np.where(rxdose_kde[0]<=60,rxdose_kde[2],0).max()
pp_74 = np.where(rxdose_kde[0]<=74,rxdose_kde[2],0).max()
pp_60_74 = pp_74 - pp_60
pp_over_74 = 1 - pp_74

rxdose_kde = create_kde(rxdose_tcc,bandwidth=.26)
ax.plot(rxdose_kde[0], rxdose_kde[1] , c=(.4,.4,.4),lw=1,ls='-',label = 'KDE')

ax.text(.25,.65,'{:0.1f}%'.format(pp_60*100), transform=ax.transAxes, fontsize=15)
ax.text(.485,.8,'{:0.1f}%'.format(pp_60_74*100), transform=ax.transAxes,fontsize=15)
ax.text(.65,.65,'{:0.1f}%'.format(pp_over_74*100), transform=ax.transAxes,fontsize=15)

ax.axvline(x = 60,c='k',lw=1,ls='--')
ax.axvline(x = 74,c='k',lw=1,ls='--')


ax.set_xlim(0,125)
ax.set_yticks([])
ax.set_xlabel('RxRSI for TCC Lung')

#plt.savefig('/Users/geoffreysedor/Downloads/'+'TCC_lung_dist2', dpi =300)


# ## 60 pt group outcome analysis

# In[43]:


from lifelines import KaplanMeierFitter
from lifelines import WeibullFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test


# ### km fits

# In[39]:


T = lc_df['td_yrs'].values
E = lc_df['LocalFailure'].values

gard33=lc_df['gard33']
idx = (gard33 == True)
T1 = T[idx]
T2 = T[~idx]
E1 = E[idx]
E2 = E[~idx]

km1 = KaplanMeierFitter()
km1.fit(T1, event_observed = E1, label = 'KM GARD>33')
km2 = KaplanMeierFitter()
km2.fit(T2, event_observed = E2, label = 'KM GARD<33')
km_all = KaplanMeierFitter()
km_all.fit(T, event_observed = E, label = 'KM All')


# In[42]:


survtable1 = km1.survival_function_
S_km1 = survtable1.values
t_1 = survtable1.index.values

survtable2 = km2.survival_function_
S_km2 = survtable2.values
t_2 = survtable2.index.values

t1_cens = T1[np.where(E1!=1)]
C1 = survtable1.loc[t1_cens]
t2_cens = T2[np.where(E2!=1)]
C2 = survtable2.loc[t2_cens]

fig,ax = plt.subplots(figsize=(6,5))

col = (.1,.4,.7)
ax.step(t_1,S_km1,where = 'post',color=col,lw=1.75)
ax.scatter(t1_cens,C1,marker='|',color=col,s=50,lw=.75)

col  = cp[1]
ax.step(t_2,S_km2,where = 'post',color=col,lw=1.75)
ax.scatter(t2_cens,C2,marker='|',color=col,s=50,lw=.75)


# In[44]:



T = lc_df['td_yrs']
E = lc_df['LocalFailure']
gard=lc_df['gardtx']
idx = gard>33

T1 = T[idx]
T2 = T[~idx]
E1 = E[idx]
E2 = E[~idx]

wf0 = WeibullFitter()
wf0.fit(T,E)
wf1 = WeibullFitter()
wf1.fit(T1, E1)
wf2 = WeibullFitter()
wf2.fit(T2, E2)

def S1(t):
    return wf1.predict(t)
def S2(t):
    return wf2.predict(t)

pd.concat([wf1.summary, wf2.summary, wf0.summary],axis=0)


# ### wbl fit for gard 33 cutpoint

# In[45]:


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


# ### trials of escalation (panels 2 & 3)

# In[46]:



def calc_rxdose(rsi_array):
    d = 2
    beta = 0.05
    n = 1
    alpha_array = (ln(rsi_array)+beta*n*(d**2))/(-n*d)
    rxdose_array = 33/(alpha_array+beta*d)
    return rxdose_array

def percentile(array,val):
    pp = np.argwhere(array<=val).size/array.size
    return pp

""""------------------------------------------------"""

trials = 100
num_pts = 200
t=np.arange(0,10,.1)

rsi_cohort = np.zeros(num_pts)
rxdose = np.zeros(num_pts)

results_1 = np.zeros(shape=(trials,len(t)))
results_2 = np.zeros(shape=(trials,len(t)))
results_3 = np.zeros(shape=(trials,len(t)))
results_4 = np.zeros(shape=(trials,len(t)))

for i in range(trials):
    rsi_cohort = np.random.choice(r,num_pts)
    
    cohort_1 = calc_rxdose(np.random.choice(r,num_pts))    # 60 Gy arm
    cohort_2 = calc_rxdose(np.random.choice(r,num_pts))    # selective escalation

    """case 1: 60 vs. selected escalation to 74"""

    C1 = percentile(cohort_1,60)
    C2 = 1 - C1
    lc_cohort_1 = C1*S1(t) + C2*S2(t)

    C1 = percentile(cohort_2,74)
    C2 = 1 - C1
    lc_cohort_2 = C1*S1(t) + C2*S2(t)

    dose_cohort_2 = np.where((cohort_2>60) & (cohort_2<=74), 74, 60)
    average_dose_cohort2 = np.mean(dose_cohort_2)
    
    risk60 = (risk_p(60)+risk_e(60))/100
    plc_cohort_1 = np.power(lc_cohort_1,np.exp(risk60))#*(1+60/14*0.074))

    risk6074 = (risk_p(average_dose_cohort2)+risk_e(average_dose_cohort2))/100
    #plc_cohort_2 = np.power(lc_cohort_2,np.exp(risk6074)*(1+average_dose_cohort2/14*0.074))
    plc_cohort_2 = np.power(lc_cohort_2,np.exp(risk6074)*(1+(average_dose_cohort2-60)/14*0.074))
    
    results_1[i,:] = plc_cohort_1
    results_2[i,:] = plc_cohort_2
    
    """case 1.5 -> 60 vs 60-74"""
    
    dose_cohort_4 = np.where((cohort_2>60) & (cohort_2<=74), cohort_2, 60)
    average_dose_cohort4 = np.mean(dose_cohort_4)
    risk60_thru_74 = (risk_p(average_dose_cohort4)+risk_e(average_dose_cohort4))/100
    #plc_cohort_4 = np.power(lc_cohort_4,np.exp(risk6074)*(1+average_dose_cohort2/14*0.074))
    plc_cohort_4 = np.power(lc_cohort_2,np.exp(risk60_thru_74)*(1+(average_dose_cohort4-60)/14*0.074))
    results_4[i,:] = plc_cohort_4

    """case 2: 60 vs. 45-80"""
    cohort_3 = calc_rxdose(np.random.choice(r,num_pts))    # precision dosing

    C1 = percentile(cohort_1,80)
    C2 = 1 - C1
    lc_cohort_3 = C1*S1(t) + C2*S2(t)

    dose_cohort_3 = np.where((cohort_2>=45) & (cohort_2<=80), cohort_2, 60)
    average_dose_cohort3 = np.mean(dose_cohort_3)

    risk4580 = (risk_p(average_dose_cohort3)+risk_e(average_dose_cohort3))/100
    #plc_cohort_3 = np.power(lc_cohort_3,np.exp(risk4580)*(1+average_dose_cohort3/14*0.074))
    plc_cohort_3 = np.power(lc_cohort_3,np.exp(risk4580)*(1+(average_dose_cohort3-60)/14*0.074))
    results_3[i,:] = plc_cohort_3

plc_cohort_1_ave = np.average(results_1, axis=0)
plc_cohort_2_ave = np.average(results_2, axis=0)
plc_cohort_3_ave = np.average(results_3, axis=0)
plc_cohort_4_ave = np.average(results_4, axis=0)


# calculating percentages at 2 and 5 years

# In[47]:


for i in [20,50]:
    z=plc_cohort_1_ave[i]*100
    y=plc_cohort_2_ave[i]*100
    print('{}\n'.format(y-z))


# In[48]:


for i in [20,50]:
    z=plc_cohort_1_ave[i]*100
    y=plc_cohort_4_ave[i]*100
    print('{}\n'.format(y-z))


# In[49]:


for i in [20,50]:
    z=plc_cohort_4_ave[i]*100
    y=plc_cohort_3_ave[i]*100
    print('{}\n'.format(y-z))


# In[50]:


fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (12,6),sharex=False)
    
"""------- first panel ---------"""
ax = axes[0]

ax.plot(t,plc_cohort_1_ave, label='PLC 60 Gy All')
ax.plot(t,plc_cohort_2_ave, label='PLC 60 Gy \nw. Selective 74 Gy Escalation')

ax.plot(t,plc_cohort_4_ave, color='k')
# ax.plot(t,plc_cohort_1)
# ax.plot(t,plc_cohort_2)


"""-------- second panel -----------"""
ax = axes[1]
        
ax.plot(t,plc_cohort_1_ave,label='PLC 60 Gy All')
ax.plot(t,plc_cohort_3_ave, color = cp[3],label='PLC 60 Gy \nw. 45-80 Gy Individualized Dose')


for i in range(axes.size):
    ax=axes[i]    
    ax.set_yticks(np.arange(0,1.2,.2),minor=False)

    ax.set_xticks(range(6))
    ax.set_xticklabels(list('012345'))
    ax.set_xlabel('Time (yr)')
    if i == 0: ax.set_ylabel('Probability of Event')
    ax.set_yticks(np.arange(0,1,.05),minor=True)
    ax.set_xbound(0,5)
    ax.set_ylim(0,1.02)
    ax.legend(fontsize=14)


fig.subplots_adjust(wspace=.3)

#plt.savefig('/Users/geoffreysedor/Downloads/'+'0617_sim6', dpi =300)


# In[ ]:





# ### combined 3 panel for fig #4

# In[53]:


from matplotlib.gridspec import GridSpec

r = nsclc.rsi
d = 2
beta = 0.05
n = 1
alpha_tcc = (np.log(r)+beta*n*(d**2))/(-n*d)
rxdose_tcc = 33/(alpha_tcc+beta*d)
rxdose_tcc=rxdose_tcc.values


#fig,axes = plt.subplots(nrows=1,ncols=3,figsize = (20,6),sharex=False)

fig = plt.figure(figsize=(18,6))
gs1 = GridSpec(1, 1, figure=fig)
gs2 = GridSpec(1, 2, figure=fig)

""" ------------------plotting histogram--------------- """
# ax=axes[0]
ax = fig.add_subplot(gs1[0,0])

binlist=list(np.arange(0,150,2))+[300]

""" <60 range """
xdata = rxdose_tcc[np.where(rxdose_tcc<60)]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.6,#ec = 'k',
        color=cp[0],  
        weights = wts)

""" 60-74 range """
xdata = rxdose_tcc[np.where((rxdose_tcc>60)&(rxdose_tcc<74))]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.8,#ec = 'k',
        color=(.4,.4,.4),  
        weights = wts,zorder=5)

""" >74 range """
xdata = rxdose_tcc[np.where((rxdose_tcc>74))]  #&(rxdose_tcc<80))]
wts =  np.full(len(xdata),.0002)
ax.hist(xdata,bins = binlist,
        alpha=.7,#ec = 'k',
        color=cp[3],  
        weights = wts)

rxdose_kde = create_kde(rxdose_tcc,bandwidth=.15)

pp_60 = np.where(rxdose_kde[0]<=60,rxdose_kde[2],0).max()
pp_74 = np.where(rxdose_kde[0]<=74,rxdose_kde[2],0).max()
pp_60_74 = pp_74 - pp_60
pp_over_74 = 1 - pp_74

rxdose_kde = create_kde(rxdose_tcc,bandwidth=.26)
ax.plot(rxdose_kde[0], rxdose_kde[1] , c=(.4,.4,.4),lw=1,ls='-',label = 'KDE')

ax.text(.25,.65,'{:0.1f}%'.format(pp_60*100), transform=ax.transAxes, fontsize=15)
ax.text(.485,.8,'{:0.1f}%'.format(pp_60_74*100), transform=ax.transAxes,fontsize=15)
ax.text(.65,.65,'{:0.1f}%'.format(pp_over_74*100), transform=ax.transAxes,fontsize=15)

ax.axvline(x = 60,c='k',lw=1,ls='--')
ax.axvline(x = 74,c='k',lw=1,ls='--')

ax.set_xlim(0,125)
ax.set_yticks([])
ax.set_xlabel('RxRSI for TCC Lung')
ax.set_title('(A)',loc='left',pad=8)

    
"""------- second panel ---------"""
# ax = axes[1]
ax = fig.add_subplot(gs2[0,0])

ax.plot(t,plc_cohort_1_ave, label='PLC 60 Gy All')
ax.plot(t,plc_cohort_2_ave, label='PLC 60 Gy w. Selective\n 74 Gy Escalation')

ax.set_yticks(np.arange(0,1.2,.2),minor=False)
ax.set_xticks(range(6))
ax.set_xticklabels(list('012345'))
ax.set_xlabel('Time (yr)')
ax.set_ylabel('Probability of Event')
#ax.set_yticks(np.arange(0,1,.05),minor=True)
ax.set_xbound(0,5)
ax.set_ylim(0,1.02)
ax.legend(fontsize=14)
ax.set_title('(B)',loc='left',pad=8)

"""-------- third panel -----------"""
# ax = axes[2]
ax = fig.add_subplot(gs2[0,1])
        
ax.plot(t,plc_cohort_1_ave,label='PLC 60 Gy All')
ax.plot(t,plc_cohort_3_ave, color = cp[3],label='PLC 60 Gy w. 45-80 Gy\nIndividualized Dose')

   
ax.set_yticks(np.arange(0,1.2,.2),minor=False)
ax.set_xticks(range(6))
ax.set_xticklabels(list('012345'))
ax.set_xlabel('Time (yr)')
ax.set_ylabel('Probability of Event')
#ax.set_yticks(np.arange(0,1,.05),minor=True)
ax.set_xbound(0,5)
ax.set_ylim(0,1.02)
ax.legend(fontsize=14)
ax.set_title('(C)',loc='left',pad=8)


# fig.subplots_adjust(left=.06,right=.97,bottom=.15,top=.9,wspace=.2)

midpoint = 0.4
gs1.tight_layout(fig, rect=[0, 0, midpoint, .98])
gs2.tight_layout(fig, rect=[midpoint,0,1,.98])
# gs2.update(wspace=.35)

# plt.savefig('/Users/geoffreysedor/Documents/'+'fig4_dec09', dpi =300)


# In[76]:


t = np.arange(0,10,.1)
fig,axes = plt.subplots(nrows=1,ncols=5,figsize = (20,5),sharex=False)
tab20=sns.color_palette("tab20c",n_colors=20)
col_pal = palettable.cartocolors.sequential.RedOr_6_r.mpl_colors
col_pal2 = palettable.cartocolors.sequential.Sunset_6_r.mpl_colors

C_all = [pp_below,pp_above,pp_localcontrol,pp_underdosed]
doses_given = [dose_46_74,dose_given2,dose_given3]

C_rand = [0.5097, 0.5927, 0.4267] # percent local control, CI top, CI bottom
lc_rand_5070 = C_rand[0]*S1(t)+(1-C_rand[0])*S2(t)

p_mce_1gy = .16

t5 = np.array([5])
P_5yr = dict()

for i in range(axes.size):
    ax = axes[i]
    
    """------- first panel ---------"""
    if  i  ==0:
        """wb fits for each KM:"""
        ax.plot(S1(t),label='GARD > 33', c=tab20[16])
        ax.plot(S2(t),label = 'GARD < 33', c=tab20[16], linestyle = '--')
        P_5yr['LC gard>33']=S1(t5)
        P_5yr['LC gard<33']=S2(t5)
        
        ax.plot(lc_rand_5070, c = (.1,.1,.4), label = 'Random Dose 50-70 Gy')
        P_5yr['LC random 50-70']=C_rand[0]*S1(t5)+(1-C_rand[0])*S2(t5)
        
        """ plotting KM lines """
        kmf_gt33.plot(ax=ax,ci_show = False,show_censors = False, label = 'KM for >GARD33', c= 'k', linewidth = 1,linestyle=':')
        kmf_lt33.plot(ax=ax,ci_show = False,show_censors = False, label = 'KM for >GARD33', c = 'k', linewidth = 1,linestyle=':')
        
        h = ax.get_legend_handles_labels()[0]
        h = h[0:3]
        ax.legend(handles = h,loc = 3, fontsize = 10)  
    
    """-------- second panel -----------"""
    if  i ==1:
        """ plotting KM lines """
        kmf_gt33.plot(ax=ax,ci_show = False,show_censors = False, label = 'KM for >GARD33', c= 'k', linewidth = 1,linestyle=':')
        kmf_lt33.plot(ax=ax,ci_show = False,show_censors = False, label = 'KM for >GARD33', c = 'k', linewidth = 1,linestyle=':') 
        
        #______________ 60Gy all ___________________
        LC60 = 0.41*S1(t)+0.59*S2(t)
        risk60 = (risk_p(60)+risk_e(60))/100
        EFS60 = np.power(LC60,np.exp(risk60))
        ax.plot(t,EFS60,color = cp[0],label='60 Gy (all patients)')
        
        P_5yr['EFS 60 Gy All']=EFS60[5]
        #__ envelope: _____
        LC60u = 0.47*S1(t)+.53*S2(t)
        EFS60u = np.power(LC60u,np.exp(risk60))
        LC60l = 0.35*S1(t)+.65*S2(t)
        EFS60l = np.power(LC60l,np.exp(risk60))
        ax.plot(t,EFS60u,t,EFS60l,c=cp[0],lw=0,linestyle='-')
        ax.fill_between(x = t,y1 = EFS60u, y2 =EFS60l, color=cp[0],alpha=.2)
        
        #_______________74Gy all_____________________
        LC74 = 0.58*S1(t)+0.42*S2(t)
        risk74 = (risk_p(74)+risk_e(74))/100
        MCE_7460 = 1+p_mce_1gy
        EFS74 = np.power(LC74,MCE_7460*np.exp(risk74))
        ax.plot(t,EFS74,label='74 Gy (all patients)',c=cp[4])
        
        P_5yr['EFS 74 Gy All']=EFS74[5]
        
        #___ envelope: ___
        LC74u = 0.63*S1(t)+.37*S2(t)
        EFS74u = np.power(LC74u,MCE_7460*np.exp(risk74))
        LC74l = 0.52*S1(t)+.48*S2(t)
        EFS74l = np.power(LC74l,MCE_7460*np.exp(risk74))
        ax.plot(t,EFS74u,t,EFS74l,c=cp[4],lw=0,linestyle='-')
        ax.fill_between(x = t,y1 = EFS74u, y2 =EFS74l, color=cp[4],alpha=.5)
        
        h = ax.get_legend_handles_labels()[0]
        h = h[0:3]+[h[5]]
        ax.legend(handles = h,loc = 3, fontsize = 9)
        
        
    """_________________ 3rd panel _________________"""
    if  i ==2:
        
        """60Gy all w/ envelope"""
        LC60 = 0.41*S1(t)+0.59*S2(t)
        risk60 = (risk_p(60)+risk_e(60))/100
        EFS60 = np.power(LC60,np.exp(risk60))
        ax.plot(t,EFS60,label='60 Gy (all patients)')
        
        #envelope for 60
        LC60u = 0.47*S1(t)+.53*S2(t)
        EFS60u = np.power(LC60u,np.exp(risk60))
        LC60l = 0.35*S1(t)+.65*S2(t)
        EFS60l = np.power(LC60l,np.exp(risk60))
        ax.plot(t,EFS60u,t,EFS60l,c=cp[0],lw=0,linestyle='-')
        ax.fill_between(x = t,y1 = EFS60u, y2 =EFS60l, color=cp[0],alpha=.2)
        
        """60 with  selected  escalation to 74"""
        pp6074=pp_localcontrol
        LC6074=pp6074*S1(t)+(1-pp6074)*S2(t)
        d_given = dose_given3
        risk6074 = np.mean((risk_p(d_given)+risk_e(d_given))/100)
        #excess_dose = np.maximum(d_given-rxdose,0)
        excess_dose = np.maximum(d_given-60,0)
        risk_c=1+np.mean(excess_dose)/14*p_mce_1gy
        
        EFS6074 = np.power(LC6074,risk_c*np.exp(risk6074))
        
        ax.plot(t,EFS6074, label = '60 Gy, 74 Gy selected escalation', c = col_pal[2])
        
        P_5yr['EFS all 60, selected 74']=EFS6074[5]
        
        # envelope for 60_74
        c_u = .63
        c_l = .54
        d_given = dose_given3
        risk6074 = np.mean((risk_p(d_given)+risk_e(d_given))/100)
        #excess_dose = np.maximum(d_given-rxdose,0)
        excess_dose = np.maximum(d_given-60,0)
        risk_c=1+np.mean(excess_dose)/14*p_mce_1gy
        
        LC6074u = c_u*S1(t)+(1-c_u)*S2(t)
        EFS6074u = np.power(LC6074u,risk_c*np.exp(risk6074))
        LC6074l = c_l*S1(t)+(1-c_l)*S2(t)
        EFS6074l = np.power(LC6074l,risk_c*np.exp(risk6074))
        ax.plot(t,EFS6074u,t,EFS6074l,color = col_pal[2],lw=0,linestyle='-')
        ax.fill_between(x = t,y1 = EFS6074u, y2 =EFS6074l, color = col_pal[2],alpha=.5)
        
        """ labels & legend """
        ax.set_xlabel('Time (years)',fontsize=14)
        h = ax.get_legend_handles_labels()[0]
        h = [h[0],h[3]]
        ax.legend(handles = h,loc = 3, fontsize = 9)
        
    """ --------  4th panel ----------"""
    if  i ==3:
        
        """rx = 62-74 subgroup"""
        rxdose_ref = d_6074
        LC60_ = S2(t)
        risk60 = (risk_p(60)+risk_e(60))/100
        EFS60_ = np.power(LC60_,np.exp(risk60))
        ax.plot(t,EFS60_, label = '60 Gy (RxRSI 62-74)', c = col_pal[4])
        
        P_5yr['EFS at 60Gy, RxRSI=62-74']=EFS60_[5]
        
        LC74_ = S1(t)
        risk74 = (risk_p(74)+risk_e(74))/100
        risk_c_7460 = 1+(74-60)/14*p_mce_1gy
        EFS74_ = np.power(LC74_,risk_c_7460*np.exp(risk74))
        ax.plot(t,EFS74_,label='74 Gy (RxRSI 62-74)',c=col_pal[0])
        
        P_5yr['EFS at 74Gy, RxRSI=62-74']=EFS74_[5]
        
        """ labels & legend """
        ax.set_xlabel('Time (years)',fontsize=14)
        h = ax.get_legend_handles_labels()[0]
        #h = [h[7],h[6],h[3],h[0],h[-1]]
        ax.legend(handles = h,loc = 3, fontsize = 9)
    
    """--------- check for 80gy escalation --------"""
    if i == 4:
        """ plotting KM lines """
        kmf_gt33.plot(ax=ax,ci_show = False,show_censors = False, label = 'KM for >GARD33', c= 'k', linewidth = 1,linestyle=':')
        kmf_lt33.plot(ax=ax,ci_show = False,show_censors = False, label = 'KM for >GARD33', c = 'k', linewidth = 1,linestyle=':')
        
        
        LC60 = 0.41*S1(t)+0.59*S2(t)
        risk60 = (risk_p(60)+risk_e(60))/100
        EFS60 = np.power(LC60,np.exp(risk60))
        ax.plot(t,EFS60,color = cp[0],label='60 Gy (all patients)',lw=1.25)
        
        LC60u = 0.47*S1(t)+.53*S2(t)
        EFS60u = np.power(LC60u,np.exp(risk60))
        LC60l = 0.35*S1(t)+.65*S2(t)
        EFS60l = np.power(LC60l,np.exp(risk60))
        ax.plot(t,EFS60u,t,EFS60l,c=cp[0],lw=1,linestyle='-',zorder=0)
        ax.fill_between(x = t,y1 = EFS60u, y2 =EFS60l, color=cp[0],alpha=.2)
        
        #k80_ave,k80_05, k80_95
        LC80 = k80_ave*S1(t)+(1-k80_ave)*S2(t)
        risk80 = (risk_p(80)+risk_e(80))/100
        MCE_8060 = 1+(80-60)/14*p_mce_1gy
        EFS80 = np.power(LC80,MCE_8060*np.exp(risk80))
        ax.plot(t,EFS80,label='80 Gy (all patients)',c=cp[3],lw=1.25)
        
        LC80u = k80_95*S1(t)+(1-k80_95)*S2(t)
        EFS80u = np.power(LC80u,MCE_8060*np.exp(risk80))
        LC80l = k80_05*S1(t)+(1-k80_05)*S2(t)
        EFS80l = np.power(LC80l,MCE_8060*np.exp(risk80))
        ax.plot(t,EFS80u,t,EFS80l,c=cp[3],lw=0,linestyle='-')
        ax.fill_between(x = t,y1 = EFS80u, y2 =EFS80l, color=cp[3],alpha=.4,zorder=5)
        
        h = ax.get_legend_handles_labels()[0]
        h = h[0:3]+[h[5]]
        ax.legend(handles = h,loc = 3, fontsize = 9)
        
    ax.set_yticks(np.arange(0,1,.05),minor=True)
    ax.set_yticks(np.arange(0,1,.2),minor=False)

    ax.set_xticks(range(6))
    ax.set_xticklabels(list('012345'),fontsize=10)
    ax.set_xlabel('Time (yr)',fontsize=12)
    if i == 0: ax.set_ylabel('Probability of Event',fontsize=14)
    ax.set_xbound(0,5)
    ax.set_ybound(0,1.0)


fig.subplots_adjust(wspace=.2)

#plt.savefig('/Users/geoffreysedor/Downloads/'+'0617_sim6', dpi =300)


# ## fig 3

# ### 0617 simulation (60 vs. 74)

# In[54]:



def calc_rxdose(rsi_array):
    d = 2
    beta = 0.05
    n = 1
    alpha_array = (ln(rsi_array)+beta*n*(d**2))/(-n*d)
    rxdose_array = 33/(alpha_array+beta*d)
    return rxdose_array

def percentile(array,val):
    pp = np.argwhere(array<=val).size/array.size
    return pp

""""------------------------------------------------"""

trials = 100
num_pts = 200
t=np.arange(0,10,.1)

rsi_cohort = np.zeros(num_pts)
rxdose = np.zeros(num_pts)

results_1 = np.zeros(shape=(trials,len(t)))
results_2 = np.zeros(shape=(trials,len(t)))

lc_results_1 = np.zeros(shape=(trials,len(t)))
lc_results_2 = np.zeros(shape=(trials,len(t)))

for i in range(trials):
    rsi_cohort = np.random.choice(r,num_pts)
    
    cohort_1 = calc_rxdose(np.random.choice(r,num_pts))    # 60 Gy arm
    cohort_2 = calc_rxdose(np.random.choice(r,num_pts))    # selective escalation

    """case 1: 60 vs. selected escalation to 74"""

    C1 = percentile(cohort_1,60)
    C2 = 1 - C1
    lc_cohort_1 = C1*S1(t) + C2*S2(t)

    C1 = percentile(cohort_2,74)
    C2 = 1 - C1
    lc_cohort_2 = C1*S1(t) + C2*S2(t)
    
    risk60 = (risk_p(60)+risk_e(60))/100
    plc_cohort_1 = np.power(lc_cohort_1,np.exp(risk60))
#     plc_cohort_1 = np.power(lc_cohort_1,np.exp(risk60)*(1+60/14*0.074))

    risk74 = (risk_p(74)+risk_e(74))/100
    plc_cohort_2 = np.power(lc_cohort_2,np.exp(risk74)*(1+14/14*0.074))
#     plc_cohort_2 = np.power(lc_cohort_2,np.exp(risk74)*(1+74/14*0.074))
    
    # append results to array
    results_1[i,:] = plc_cohort_1
    results_2[i,:] = plc_cohort_2
    
    lc_results_1[i,:] = lc_cohort_1
    lc_results_2[i,:] = lc_cohort_2

# average all plc results from trials
plc_cohort_1_ave = np.average(results_1, axis=0)
plc_cohort_2_ave = np.average(results_2, axis=0)
plc_cohort_1_std = np.std(results_1,axis=0)
plc_cohort_2_std = np.std(results_2,axis=0)

# average and std deviation for lc results all trials
lc_cohort_1_ave = np.average(lc_results_1, axis=0)
lc_cohort_2_ave = np.average(lc_results_2, axis=0)
lc_cohort_1_std = np.std(lc_results_1,axis=0)
lc_cohort_2_std = np.std(lc_results_2,axis=0)


# In[55]:


time = 10
std = plc_cohort_1_std[time]
pred_1yr_efs60 = [plc_cohort_1_ave[time],plc_cohort_1_ave[time]-3*std,plc_cohort_1_ave[time]+3*std]
std = plc_cohort_2_std[time]
pred_1yr_efs74 = [plc_cohort_2_ave[time],plc_cohort_2_ave[time]-3*std, plc_cohort_2_ave[time]+3*std]

time = 20
std = plc_cohort_1_std[time]
pred_2yr_efs60 = [plc_cohort_1_ave[time],plc_cohort_1_ave[time]-3*std,plc_cohort_1_ave[time]+3*std]
std = plc_cohort_2_std[time]
pred_2yr_efs74 = [plc_cohort_2_ave[time],plc_cohort_2_ave[time]-3*std, plc_cohort_2_ave[time]+3*std]


# In[56]:


fig,ax = plt.subplots(figsize = (6,7.5),sharex=False)
    
"""------- first panel ---------"""

ax.plot(t,plc_cohort_1_ave, label='PLC 60 Gy All')
std = plc_cohort_1_std
ax.fill_between(t,plc_cohort_1_ave+2*std,plc_cohort_1_ave-2*std,color=cp[0],alpha=.4)

ax.plot(t,plc_cohort_2_ave, label='PLC 74 Gy All', color=cp[4])
std = plc_cohort_2_std
ax.fill_between(t,plc_cohort_2_ave+2*std,plc_cohort_2_ave-2*std,color=cp[4],alpha=.4)
  
ax.set_yticks(np.arange(0,1.2,.2),minor=False)
ax.set_xticks(range(6))
ax.set_xticklabels(list('012345'))
ax.set_xlabel('Time (yr)')
if i == 0: ax.set_ylabel('Probability of Event')
ax.set_yticks(np.arange(0,1,.05),minor=True)
ax.set_xbound(0,5)
ax.set_ylim(0,1.02)
ax.legend(fontsize=14)



#plt.savefig('/Users/geoffreysedor/Downloads/'+'0617_sim6', dpi =300)


# In[415]:


def boxplt(height,mean,ci_lower,ci_upper,color='k',label='',alpha=1,ls='-'):
#     ax = plt.gca()
    width = 4
    h = height
    center_xcoord = mean
    lower_xcoord = ci_lower
    upper_xcoord = ci_upper
    mid_h = .04
    end_h = .03
    kwargs = {'alpha':alpha,'color':color,'transform':ax.transData}
    ax.vlines(x=center_xcoord,ymin=h-mid_h,ymax=h+mid_h,linewidth=width,**kwargs)
    ax.vlines(x = lower_xcoord,ymin=h-end_h,ymax=h+end_h,**kwargs)
    ax.vlines(x = upper_xcoord,ymin=h-end_h,ymax=h+end_h,**kwargs)
    ax.hlines(y = h, xmin = ci_lower,xmax=ci_upper,**kwargs,linestyle=ls,label=label)

fig,ax = plt.subplots(figsize=(12,7.5))

time = 10
std = plc_cohort_1_std[time]
pred_1yr_efs60 = [plc_cohort_1_ave[time],plc_cohort_1_ave[time]-3*std,plc_cohort_1_ave[time]+3*std]

std = plc_cohort_2_std[time]
pred_1yr_efs74 = [plc_cohort_2_ave[time],plc_cohort_2_ave[time]-3*std, plc_cohort_2_ave[time]+3*std]


time = 20
std = plc_cohort_1_std[time]
pred_2yr_efs60 = [plc_cohort_1_ave[time],plc_cohort_1_ave[time]-3*std,plc_cohort_1_ave[time]+3*std]

std = plc_cohort_2_std[time]
pred_2yr_efs74 = [plc_cohort_2_ave[time],plc_cohort_2_ave[time]-3*std, plc_cohort_2_ave[time]+3*std]

reported_lc1yr_60 = [.837,.787,.886]
reported_lc1yr_74 = [.752,.693,.811]
reported_lc2yr_60 = [.693,.631,.755]
reported_lc2yr_74 = [.614,.547,.681]

h1 = [.96,.9,.84,.74,.66]
boxplt(.96,*reported_lc1yr_60,label='Reported 60 Gy')
boxplt(.72,*reported_lc1yr_74,color='k',ls='--',label='Reported 74 Gy')

boxplt(.84,*pred_1yr_efs60,color=cp[0],label = 'PLC 60 Gy')
boxplt(.6,*pred_1yr_efs74,color=cp[4],label='PLC 74 Gy')

h2=[.44,.38,.3,.22,.14,.07]


boxplt(.34,*pred_2yr_efs60,color=cp[0])
boxplt(.09,*pred_2yr_efs74,color=cp[4])

boxplt(.44,*reported_lc2yr_60)
boxplt(.22,*reported_lc2yr_74,ls='--')

ax.axhline(y=.5,color='k',lw=1)
ax.set_yticks([.25,.75])
ax.set_yticklabels(['2-yr','1-yr'])

ax.set_ylim(0,1.03)

ax.legend(loc=2)
ax.set_xlim(.5,1)

ax.set_xlabel('Cohort Fraction',labelpad=8,fontsize=12)





# plt.savefig('/Users/geoffreysedor/Downloads/'+'forest_plot3', dpi =200)


# ## fig 3 all panels

# In[57]:


from matplotlib.gridspec import GridSpec

def boxplt(height,mean,ci_lower,ci_upper,color='k',label='',alpha=1,ls='-'):
#     ax = plt.gca()
    width = 4
    h = height
    center_xcoord = mean
    lower_xcoord = ci_lower
    upper_xcoord = ci_upper
    mid_h = .04
    end_h = .03
    kwargs = {'alpha':alpha,'color':color,'transform':ax.transData}
    ax.vlines(x=center_xcoord,ymin=h-mid_h,ymax=h+mid_h,linewidth=width,**kwargs)
    ax.vlines(x = lower_xcoord,ymin=h-end_h,ymax=h+end_h,**kwargs)
    ax.vlines(x = upper_xcoord,ymin=h-end_h,ymax=h+end_h,**kwargs)
    ax.hlines(y = h, xmin = ci_lower,xmax=ci_upper,**kwargs,linestyle=ls,label=label)

    
fig = plt.figure(figsize=(18,14))


# gs = GridSpec(1, 1, figure=fig, width_ratios=[3, 2], height_ratios=[1, 1])
gs1 = GridSpec(1, 1, figure=fig)
gs2 = GridSpec(2, 1, figure=fig)

"""----------------- left panel--------------"""
ax = fig.add_subplot(gs1[0,0])
image = plt.imread('/Users/geoffreysedor/Documents/trial_schema.png',format='png')
ax.imshow(image)
# ax.axis('off')
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('(A)',loc='left',pad=8)

"""-----------------bottom right panel--------------"""
ax = fig.add_subplot(gs2[1,0])

# time = 10
# std = plc_cohort_1_std[time]
# pred_1yr_efs60 = [plc_cohort_1_ave[time],plc_cohort_1_ave[time]-3*std,plc_cohort_1_ave[time]+3*std]
# std = plc_cohort_2_std[time]
# pred_1yr_efs74 = [plc_cohort_2_ave[time],plc_cohort_2_ave[time]-3*std, plc_cohort_2_ave[time]+3*std]

# time = 20
# std = plc_cohort_1_std[time]
# pred_2yr_efs60 = [plc_cohort_1_ave[time],plc_cohort_1_ave[time]-3*std,plc_cohort_1_ave[time]+3*std]
# std = plc_cohort_2_std[time]
# pred_2yr_efs74 = [plc_cohort_2_ave[time],plc_cohort_2_ave[time]-3*std, plc_cohort_2_ave[time]+3*std]

reported_lc1yr_60 = [.837,.787,.886]
reported_lc1yr_74 = [.752,.693,.811]
reported_lc2yr_60 = [.693,.631,.755]
reported_lc2yr_74 = [.614,.547,.681]

h1 = [.96,.9,.84,.74,.66]
boxplt(.96,*reported_lc1yr_60,label='Reported 60 Gy')
boxplt(.72,*reported_lc1yr_74,color='k',ls='--',label='Reported 74 Gy')
boxplt(.84,*pred_1yr_efs60,color=cp[0],label = 'PLC 60 Gy')
boxplt(.6,*pred_1yr_efs74,color=cp[4],label='PLC 74 Gy')

h2=[.44,.38,.3,.22,.14,.07]
boxplt(.34,*pred_2yr_efs60,color=cp[0])
boxplt(.09,*pred_2yr_efs74,color=cp[4])
boxplt(.44,*reported_lc2yr_60)
boxplt(.22,*reported_lc2yr_74,ls='--')

ax.axhline(y=.5,color='k',lw=1)
ax.set_yticks([.25,.75])
ax.set_yticklabels(['2-yr','1-yr'])

ax.set_ylim(0,1.03)
ax.legend(loc=4)
ax.set_xlim(.5,1)
ax.set_xlabel('Cohort Fraction',labelpad=8)
ax.set_title('(C)',loc='left',pad=8)

                        
                        
"""-----------------top right panel--------------"""
ax = fig.add_subplot(gs2[0,0])


ax.plot(t,plc_cohort_1_ave, label='PLC 60 Gy All')
std = plc_cohort_1_std
ax.fill_between(t,plc_cohort_1_ave+2*std,plc_cohort_1_ave-2*std,color=cp[0],alpha=.4)

ax.plot(t,plc_cohort_2_ave, label='PLC 74 Gy All', color=cp[4])
std = plc_cohort_2_std
ax.fill_between(t,plc_cohort_2_ave+2*std,plc_cohort_2_ave-2*std,color=cp[4],alpha=.4)
  
ax.set_yticks(np.arange(0,1.2,.2),minor=False)
ax.set_xticks(range(6))
ax.set_xticklabels(list('012345'))
ax.set_xlabel('Time (yr)')
ax.set_ylabel('Probability of Event')
ax.set_yticks(np.arange(0,1,.05),minor=True)
ax.set_xbound(0,5)
ax.set_ylim(0,1.02)
ax.legend(fontsize=14)
ax.set_title('(B)',loc='left',pad=8)

midpoint = 0.57
gs1.tight_layout(fig, rect=[0, 0, midpoint, .98])
gs2.tight_layout(fig, rect=[midpoint,0,.97,.98])


# plt.savefig('/Users/geoffreysedor/Documents/'+'fig3_dec16', dpi =300, edgecolor='black', linewidth=4)


# In[424]:


iterables = [['60gy','74gy'],['lc','lc_std', 'plc', 'plc_std']]

columns = pd.MultiIndex.from_product(iterables)

predicted_outcomes = pd.DataFrame(columns=columns)

times = [10, 20, 50]

for t in times:
    
    predicted_outcomes.loc[t,'60gy'] = [lc_cohort_1_ave[t],lc_cohort_1_std[t],plc_cohort_1_ave[t],plc_cohort_1_std[t]]

    predicted_outcomes.loc[t,'74gy'] = [lc_cohort_2_ave[t],lc_cohort_2_std[t],plc_cohort_2_ave[t],plc_cohort_2_std[t]]

predicted_outcomes*100


# In[ ]:




