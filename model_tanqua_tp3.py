import numpy as np
import random
import matplotlib.pyplot as plt
from math import *
import scipy.signal as signal
import random
from random import choice
###bib a parte
####!pip install DeCida
from decida.Pattern import Pattern
#para reprodução do mesmo experimento setando a semente que garante gerar a mesma sequancia para execuçoes diferentes
random.seed(1234)

#função que implementa as equações diferenciais referente ao modelo do tanque quádruplo
def dvTanQua(x, uv1, uv2, t):
    A1,A2,A3,A4 = 32,28,32,28 #cm²
    alpha1,alpha2,alpha3,alpha4 = 0.071,0.071,0.071,0.071
    g = 981#cm²s⁻²
    k1, k2, kc = 3.33,3.33,0.5
    y1,y2 = 0.6, 0.7 #verificar esses valores
    xd = []
    #equação 1
    xd_0 = (-alpha1/A1) * np.sqrt(2*g*x[0]) + (alpha3/A1)*np.sqrt(2*g*x[2]) + ((y1*k1)/A1)*uv1
    xd.append(xd_0*kc)
    #equação 2
    xd_1 = (-alpha2/A2) * np.sqrt(2*g*x[1]) + (alpha4/A2)*np.sqrt(2*g*x[3]) + ((y2*k2)/A2)*uv2
    xd.append(xd_1*kc)
    #Equação 3
    xd_2 = (-alpha3/A3) * np.sqrt(2*g*x[2]) + (((1 - y2)*k2)/A3)*uv2
    xd.append(xd_2)
    #Equação 4
    xd_3 = (-alpha4/A4) * np.sqrt(2*g*x[3]) + (((1 - y1)*k1)/A4)*uv1
    xd.append(xd_3)

    return xd


#Função que implementa o Runge–Kutta de quarta ordem 
def rkTanQua(x0, uv1, uv2, h, t):
    #1st evaluation
    xd = dvTanQua(x0, uv1, uv2, t)
    savex0 = x0.copy()
    phi = xd.copy()
    for i in range(len(x0)):
        x0[i] = savex0[i] + 0.5 * h * xd[i]

    #2nd evaluation
    xd = dvTanQua(x0, uv1, uv2, t + 0.5 * h)
    phi = (phi + 2 * xd)
    for i in range(len(x0)):
        x0[i] = savex0[i] + 0.5 * h * xd[i]

    #3rd evaluation
    xd = dvTanQua(x0, uv1, uv2, t + 0.5 * h)
    phi = phi + 2 * xd
    for i in range(len(x0)):
        x0[i] = savex0[i] + h * xd[i]

    #4th evaluation
    xd = dvTanQua(x0, uv1, uv2, t + h)

    result_x = x0.copy()
    for i in range(len(x0)):
        result_x[i] = savex0[i] + (phi[i] + xd[i]) * h / 6

    return result_x

def prbs():
    while True:
    yield choice([False, True])

def acf_pacf(x):
    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(221)
    fig = sm.graphics.tsa.plot_acf(x, lags=40, ax=ax1)
    ax2 = fig.add_subplot(222)
    fig = sm.graphics.tsa.plot_pacf(x, lags=40, ax=ax2)
    acf_pacf(rs_ar1)


#função para plotar os resultados
def plotsignals(signals,ref,t, name_input,init_):
    l, = plt.plot(t,ref, label=name_input+' (entrada)')
    for idx,sig in enumerate(signals):
        for id,out in enumerate(sig):
            l, = plt.plot(t, out, label='h'+str(id)+'_estadoInicial_'+str(int(init_[idx])))
        
    #plt.legend([name_input,'h1_sign_'+str(idx),'h2_sign_'+str(idx),'h3_sign_'+str(idx),'h4_sign_'+str(idx)], loc='lower right')
    plt.legend(loc='lower right')
    plt.xlabel('Tempo de atuação')
    plt.ylabel('Altura(cm) ou entrada(tensão)')
    plt.show()

## Funções que implementam os sinais de entrada

def degrau(t, gain):
    sign = gain*np.ones((int(len(t)), 1))
    return sign

def seno(time, gain, freq=0.5, point_medium = 0):
    w = 2. * np.pi * freq
    sign = gain*np.sin(time*w) + point_medium
    return sign

def impulso(t, gain):
    imp = gain*signal.unit_impulse(int(len(t)), [0])
    return imp

def aleatorio(t, low=3,high=4):
    sign = np.array([random.uniform(low,high) for _ in range(len(t))])
    return sign

#Começa com o degrau com um valor de altura e depois é criado mais um degrau a entrada
def signal_super(t , gains):
    n_parts = len(t)/len(gains)
    signs=[]
    for gain in gains:
        sig = gain*np.ones((int(n_parts), 1))
        signs.append(sig)
    sign = np.concatenate((signs))
    return sign


def main_TanQua():
    #### gerar configurações da entrada e do tempo de avaliação
    
    #input_ pode ser - degrau, impulso, senoide, aleatoria, superposi, ponto_dif
    #cada entrada refere-se a um experimento apresentado na documentação
    input_ = 'superposi'
    gain = 3
    #referente ao primeiro segundo degrau da superposição
    gains = [1,2]
    #para senoide
    freq = 0.01
    ponto_medio = 3
    
    #variaveis de tempo de execução
    t0 = 0
    tf = 1000
    h = 0.01
    t = np.arange(t0, tf, h)
    
    #setando valores de pontos iniciais são necessários para execução em questão
    init_ = [10.1]
    
    #executando simulação 
    x_f = []
    for point_stt in init_:
        x0 = np.array([[point_stt], [point_stt], [point_stt], [point_stt]])

        dc = np.zeros((len(x0), len(t) - 1))
        x = x0.copy()
        x = np.append(x, dc, axis=1)
        #Seleção de entrada
        if input_ == 'degrau':
            u = degrau(t,gain)
        elif input_ == 'senoide':
            u = seno(t, gain, freq, ponto_medio)
        elif input_ == 'impulso':
            u = impulso(t, 15)
        elif input_ == 'aleatoria':
            u = aleatorio(t,0,6)
        elif input_ == 'superposi':
            u = signal_super(t,gains)
        else:
            u = np.zeros(len(t))

        for k in range(1, len(t)):
            result = rkTanQua(x[:, k - 1], u[k], u[k], h, t[k])
            x[:, k] = result
        x_f.append(x)
            
    plotsignals(x_f,u,t,input_,init_)

if __name__ == "__main__":
    main_TanQua()