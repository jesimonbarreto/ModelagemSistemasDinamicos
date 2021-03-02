import numpy as np

# Runge-Kutta:

def run_rk(fun,t:np.ndarray, x0, u, h):
# function x=run_rk(fun,t,x0,ux,uy,h)
# 
# This function implements a numerical integration algorithm known as 4th-order Runge-Kutta  
# fun is the callable Right-hand side of the system. The calling signature is fun(t, y).
# x0 is the initial condition
# ux and uy (if different from zero) are external forces ("control actions") added to the first and second
# ux=uy=0, the autonomous case is simulated.
# h integration interval.
# t the time BEFORE calling the function.
# LAA 15/8/18
# Python version by Felipe F. Rocha, on Jan, 26, 2021
# Revised by Fernando Souza, on Feb, 06, 2021


    z_x = np.zeros((len(x0), len(t) - 1))
    x = np.append(x0, z_x, axis=1)

    for k in range(1, len(t)):
        result = rk_model(fun,x[:, k - 1].copy(), u[k], u[k], h, t[k])
        x[:, k] = result
    return x



def rk_model(fun,x0, ux, uy, h, t):
    # 1st evaluation
    xd = fun(x0, ux, uy, t)
    savex0 = x0.copy()
    phi = xd.copy()
    for i in range(len(x0)):
        x0[i] = savex0[i] + 0.5 * h * xd[i]

    # 2nd evaluation
    xd = fun(x0.T, ux, uy, t + 0.5 * h)
    phi = (phi + 2 * xd)
    for i in range(len(x0)):
        x0[i] = savex0[i] + 0.5 * h * xd[i]

    # 3rd evaluation
    xd = fun(x0, ux, uy, t + 0.5 * h)
    phi = phi + 2 * xd
    for i in range(len(x0)):
        x0[i] = savex0[i] + h * xd[i]

    # 4th evaluation
    xd = fun(x0, ux, uy, t + h)

    result_x = x0.copy()
    for i in range(len(x0)):
        result_x[i] = savex0[i] + (phi[i] + xd[i]) * h / 6

    # Restrição de nivel

    return result_x

#def dvmodel(x, ux, uy, t):
# example of callable Right-hand side of the system for run_rk function
#
#    k = 1 #cm3
#    b = 2 #cm2
#    H = 1 #cm2
#    I = 5
#    
#    dx0 = x[1]
#    dx1 = -k/I*x[0] - b/I*x[1] + H/I*ux*math.cos(x[0])
#
#   return np.array([[dx0], [dx1]])


# PRBS:

import math

def prbs(N, b, m):
# u=prbs(N,b,m)
# generates a PRBS signal with length N and with b  
# bits each value is held during m sampling times.
# For b=8 the PRBS will not be an m-sequence.
# Luis A. Aguirre - BH 18/10/95
# - revision 01/02/1999
# Python version by Fernando Souza, on Feb, 06, 2021


    
    u = np.zeros(N);
    x = np.random.rand(b) 
    x = np.round(x)

    j=1; # for most cases the XOR of the last bit is with the one before the last. The exceptions are
    if b==5:
       j=2;
    elif b==7:
       j=3;
    elif b==9:
      j=4;
    elif b==10:
       j=3;
    elif b==11:
       j=2;

    for i in range(1,math.floor(N/m)):
        u[m*(i-1):m*i] = x[b-1]*np.ones(m)
        xor = int(u[m*(i-1)]) ^ int(x[b-1-j])
        x = np.append([xor],x[0:b-1])

    return u

# Autocorrelation or cross-correlation
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

def myccf(y,u,lag,flag1=0,flag2=1,cor_stem ='green',cor_conf='blue',legend = "empty", fig_size = (12, 4)):
# t, r, l, B = myccf(y,u,lag,0,1,cor_stem,cor_conf,legend) returns the auto-correlation or cross-correlation
#	  function depending on the signals in the vectors y and u.
#
# On entry
#	y     - signal vector.
#	u     - signal vector.
#	lag   - scalar (positive number).
#	flag1 - the ccf are calculated from -lag/2 to lag/2 if flag1 = 1 (flag1=0 is the default).
# 		the ccf are calculated from 0 to lag if flag1 = 0
# 	flag2 - plots the ccf between c1 and c2 if flag2 = 1 
#                if flag2=0 the ccf is returned in r (with respective lags in t), but not plotted
# 		 (the 95# confidence interval is the default)
# 	cor_stem - for stemlines color, e.g: 'blue', 'red', 'green', 'black'
#       (cor_stem='green' is the default)
# 	cor_conf - for confidence value lines, e.g: 'blue', 'red', 'green', 'black'
#       (cor_conf='blue' is the default)
# 	legend- string for the plot legend, e.g: "autocorrelation", "cross-correlation",
#       (no legend by default)
#   fig_size - figure size (default fig_size = (12,4))
#
# On return
#	t     - vector with lags
#	r     - auto (or cross) correlation values
#	l     - 95# confidence value.
#	B     - maximum value of the auto(cross)-correlation
# 		r*B is the unnormalized value of r.
#
# Luis Aguirre - Sheffield - may 91 
#	       - Belo Horizonte - Jan 99, update
# Modified by E. Mendes on April, 25 2004.
# Python version by Fernando Souza, on Feb, 06, 2021

# Tests

    if np.size(lag) != 1:
        raise AttributeError(f'lag is a scalar')

    lag=abs(math.floor(lag))

    if np.size(flag1) != 1:
        raise AttributeError(f'flag1 is a scalar')

    flag1=math.floor(flag1)

    if np.size(flag2) != 1:
        raise AttributeError(f'flag2 is a scalar')

    flag2=math.floor(flag2)

# Calculations

    if flag1==1:
        lag = math.floor(lag/2)

    c1 = np.squeeze(y)
    c1 = c1-np.mean(c1)
    c2 = np.squeeze(u)
    c2 = c2-np.mean(c2) 
    cc1 = np.var(c1,ddof =1)
    cc2 = np.var(c2,ddof =1)
    m=math.floor(0.1*len(c1)) 
    r12=covf(np.column_stack([c1, c2]),lag+1) 

    t=np.arange(0,lag,1)
    l=np.ones(lag)*1.96/np.sqrt(len(c1)) 

# ccf 
    raux=r12[2,lag+1::-1]
 
    B=math.sqrt(cc1*cc2) 
    r=np.append(raux[0:len(raux)-1], r12[1,:])/B 

# if -lag to lag but no plots
    if flag1 == 1:
        t=np.arange(-(lag),lag+1)
    else:
        t=np.arange(0,lag)
        r=r12[1,0:lag]/B
        
# if plot 
    if flag2 == 1:
        fig = plt.figure(figsize=fig_size)  
# if -lag to lag 
        if flag1 == 1: 
            l=np.ones(2*lag+1)*1.96/np.sqrt(len(c1)) 
        else: 
            l=np.ones(lag)*1.96/np.sqrt(len(c1)) 
 
        markerline, stemlines, baseline = plt.stem(t, r,'-')
        plt.setp(markerline, 'color', cor_stem, 'linewidth', 2)
        plt.setp(stemlines, 'color', cor_stem, 'linewidth', 2)
        plt.setp(baseline, 'color', cor_stem, 'linewidth', 2)
        if legend != "empty":
        #if  not legend is "empty":
            plt.legend([f'{legend}'],loc='best', fontsize = 'small')
            
        plt.plot(t,  l,':',color = cor_conf)
        plt.plot(t, -l,':',color = cor_conf)
        plt.xlabel('lag')
        plt.show()
        
    return t, r, l[0], B

def covf(z,M,maxsize=[]):
#COVF  Computes the covariance function estimate for a data matrix.
#   R = COVF(Z,M)
#
#   Z : An   N x nz data matrix, typically Z=[y u]
#
#   M: The maximum delay - 1, for which the covariance function is estimated
#   R: The covariance function of Z, returned so that the entry
#   R((i+(j-1)*nz,k+1) is  the estimate of E Zi(t) * Zj(t+k)
#   The size of R is thus nz^2 x M.
#   For complex data z, RESHAPE(R(:,k+1),nz,nz) = E z(t)*z'(t+k)
#   (z' is complex conjugate, transpose)
#
#   Only for nz=2, an FFT algorithm is used.
#
#   The memory trade-off is affected by
#   R = COVF(Z,M,maxsize)
#   L. Ljung 10-1-86,11-11-94
#   Copyright 1986-2001 The MathWorks, Inc.
#   $Revision: 1.6 $  $Date: 2001/04/06 14:21:37 $
#   Python version and revision by Fernando Souza, on Feb, 06, 2021

    Ncap = len(z)
    nz = len(z[0])
    maxsdef = 250000
    if not maxsize:
        maxsize=maxsdef
    if nz > Ncap:
        raise AttributeError(f'The data should be arranged in columns')
   
    nfft = int(2.**math.ceil(math.log(2*Ncap)/math.log(2)))
    Ycap = fft_lag(z[:,0],nfft)
    Ucap = fft_lag(z[:,1],nfft)
    YUcap=Ycap*np.conjugate(Ucap)
    UAcap=abs(Ucap)**2
    del Ucap
    YAcap=abs(Ycap)**2
    del Ycap
    #
    RYcap = fft_lag(YAcap,nfft)
    n = len(RYcap)
    R=RYcap[0:M].real/Ncap/n
    del RYcap                              
    #
    RUcap=fft_lag(UAcap,nfft)
    ru=RUcap[0:M].real/Ncap/n
    del RUcap
    RYUcap=fft_lag(YUcap,nfft)
    ryu=RYUcap[0:M].real/Ncap/n
    ruy=RYUcap[n-1:n-M:-1].real/Ncap/n
    del RYUcap
    R = np.row_stack([R, np.append(ryu[0], ruy), ryu, ru])
    return R
    
def fft_lag(x,nfft):
    Ncap = len(x)
    if Ncap >= nfft:
        x = x[0:nfft]
    else:
        zn = np.zeros(nfft-Ncap)
        x  = np.append(x,zn)

    X = fft(x)
    return X 

# Bode simulation

def bode_sim(fun,t:np.ndarray, x0, W:np.ndarray, h, cut=0, mod=False):
# Bode diagram by simulations
# Fernando Souza - Feb, 06, 2021

    
    cut = math.floor(len(t) - len(t)/3)
    B = np.array([[], []])
    for p in range(0, len(W)):
        u_sin = np.sin(np.array( t * W[p]))  
        u_cos = np.cos(np.array( t * W[p]))  
        y_sin = run_rk(fun,t.copy(), x0, u_sin, h)
        
        u_sin = u_sin[cut:]
        u_cos = u_cos[cut:]
        y_sin = y_sin[0,cut:]
        y_sin = np.matrix(y_sin)
        
        M = np.matrix([u_sin, u_cos]).T 
        Msqr = np.dot(M.T,M)
        Mpiv = np.dot(Msqr.I,M.T)
        X = np.inner(Mpiv,y_sin)
        
        phi = math.atan2(X[1],X[0])
        mod = abs(X[0,0])/abs(math.cos(phi))
        mod = 20*math.log10(mod)
        phi = phi*(180/math.pi)
                
        B = np.column_stack([B, [[mod],[phi]]])

    return B

def plot_yu(t:np.ndarray, x:np.ndarray, u, cut=0, t_scale = 1, u_label = 'x1', y_label = 'x2', t_label = 'sec', fig_size = (12, 4), loc_legend = 'upper right'):
# plot_yu(t, x, u, cut=0, t_scale = 1, u_label = 'x1', y_label = 'x2', t_label = 'sec', fig_size = (12, 4), loc_legend = 'upper right')
# Plot input u and output x at the same figure
# On entry
#	t     - time vector.
#	u     - input vector.
#	x     - output vector.
#	cut   - cut the vectors.
#   t_scale  - multiply the time vector.
#   u_label  - input label
#   y_label  - ouput label
#   t_label  - time label
#   fig_size - figure size (default fig_size = (12,4))
#   loc_legend - legend location: 
#       'best'
#       'upper right'
#       'upper left'
#       'lower left'
#       'lower right'
#       'right'
#       'center left'
#       'center right'
#       'lower center'
#       'upper center'
#       'center' 
#
# Fernando Souza - Feb, 08, 2021

    fig, ax = plt.subplots(constrained_layout=True, figsize = fig_size)
    ax0 = ax.twinx()

    ax.set(xlabel=t_label, xticks=(np.arange(0, len(t)*t_scale, 1)))
    top =  max( u[cut:]) * 1.1
    bottom = min(u[cut:]) - top/20
    ax.set(ylabel= u_label,ylim=(bottom, top),yticks=(np.arange(bottom, top, (top-bottom)/10)))
    p1, = ax.plot(t[cut:]*t_scale, u[cut:]+0, 'g-', label=u_label)
    
    top =  max(x[cut:]) * 1.1
    bottom = min(x[cut:]) - top/20
    ax0.set(ylabel=y_label,ylim=(bottom,top),yticks=(np.arange(bottom, top, (top-bottom)/10)))
    p2, = ax0.plot(t[cut:]*t_scale, x[cut:], 'r-' ,label=y_label)
    
    lines = [p1,p2]
    ax.legend(lines, [l.get_label() for l in lines],loc=loc_legend, fontsize = 'small')
    ax.yaxis.label.set_color('green')
    ax0.yaxis.label.set_color('red')
    plt.grid(which='both', axis='both', linestyle='--')
    plt.show()