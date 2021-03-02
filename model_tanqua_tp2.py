import numpy as np
import random
import matplotlib.pyplot as plt
from math import *
import scipy.signal as signal
import random
import matplotlib as mplt
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

#função para plotar os resultados multiplicados pelo fator para normalizar sinal
def plotsignals_p(signals,ref,t, name_input,init_):
    #l, = plt.plot(t[cont_time:],ref[cont_time:], label=name_input+' (entrada)')
    for idx,sig in enumerate(signals):
        for id_,out in enumerate(sig):
            out_n = []
            nout = out[cont_time:]
            fac = np.max(nout)
            for num in nout:
                out_n.append(num/fac)

            l, = plt.plot(t[cont_time:], np.array(out_n), label='h'+str(id_)+'_estadoInicial_'+str(int(init_[idx])))
        
    #plt.legend([name_input,'h1_sign_'+str(idx),'h2_sign_'+str(idx),'h3_sign_'+str(idx),'h4_sign_'+str(idx)], loc='lower right')
    #plt.legend(loc='upper right')
    #plt.tick_params(axis='both')
    #plt.gcf().autofmt_xdate()
    plt.yticks(np.arange(0, 2, 0.2))
    plt.xlabel('Tempo de atuação')
    plt.ylabel('Altura(cm) ou entrada(tensão)')
    plt.show()

#função para plotar os resultados
def plotsignals(signals,ref,t, name_input,init_,cont_time=100000):
    #l, = plt.plot(t[cont_time:],ref[cont_time:], label=name_input+' (entrada)')
    for idx,sig in enumerate(signals):
        for id_,out in enumerate(sig):
            nout = out[cont_time:]
            l, = plt.plot(t[cont_time:]-1000, nout - np.min(nout), label='h'+str(id_)+'_estadoInicial_'+str(int(init_[idx])))
        
    #plt.legend([name_input,'h1_sign_'+str(idx),'h2_sign_'+str(idx),'h3_sign_'+str(idx),'h4_sign_'+str(idx)], loc='lower right')
    #plt.legend(loc='upper right')
    #plt.tick_params(axis='both')
    #plt.gcf().autofmt_xdate()
    #plt.yticks(np.arange(0, 6, 0.4))
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
def signal_super(t , gains, prop = [0,1,1,1]):
    n_parts = len(t)/len(prop)
    signs=[]
    gains_l = []
    for n in prop:
        gains_l.append(gains[n])

    for gain in gains_l:
        sig = gain*np.ones((int(n_parts), 1))
        signs.append(sig)
    sign = np.concatenate((signs))
    return sign

def model_i(u, ki, t = 200, td = 1):
    Si = (ki * np.exp(-td*u))/(t*u)
    return Si



def main_TanQua():
    #### gerar configurações da entrada e do tempo de avaliação
    
    #input_ pode ser - degrau, impulso, senoide, aleatoria, superposi, ponto_dif
    #cada entrada refere-se a um experimento apresentado na documentação
    input_ = 'superposi'
    gain = 3
    #referente ao primeiro segundo degrau da superposição
    gains = [2,4]
    #para senoide
    freq = 0.01
    ponto_medio = 3
    cont_time=100000
    #variaveis de tempo de execução
    t0 = 0
    tf = 4000
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
            
    
    plot_mode = 'segOrd'
    mode_ = 'exc'
    
    #valores encontrados modelando com base no video do prof Aguirre
    #https://www.youtube.com/watch?v=J-ZQ29dcmJU&feature=youtu.be
    #https://www.youtube.com/watch?v=dm5cdpsuxlM&feature=youtu.be
    
    
    if plot_mode == 'primOrd':
        #criando sistemas de primeira ordem com valores calculados como no video do prof. Aguirre
        num = [4]
        den = [200, 1]
        sist1 = signal.TransferFunction(num, den)
        num = [2.7]
        den = [200, 1]
        sist2 = signal.TransferFunction(num, den)
        num = [0.5]
        den = [50, 1]
        sist3 = signal.TransferFunction(num, den)
        num = [0.3]
        den = [50, 1]
        sist4 = signal.TransferFunction(num, den)
        #estabelecendo valores entre 0 e 3000
        t_v = t[:-cont_time]

        #resposta ao degrau de cada sistema
        t1, out1 = signal.step(sist1, X0=None, T=t_v, N=None)
        plt.plot(t1,out1, '--')
        t2, out2 = signal.step(sist2, X0=None, T=t_v, N=None)
        plt.plot(t2,out2, '--')
        t3, out3 = signal.step(sist3, X0=None, T=t_v, N=None)
        plt.plot(t3,out3, '--')
        t4, out4 = signal.step(sist4, X0=None, T=t_v, N=None)
        plt.plot(t4,out4, '--')
        
        plotsignals(x_f,u,t,input_,init_)
    
    elif  plot_mode == 'segOrd':
        #criando sistemas de segunda ordem
        num = [4]
        den = [0.575, 7.656, 1]
        sist1 = signal.TransferFunction(num, den)
        num = [2.7]
        den = [0.575, 7.656, 1]
        sist2 = signal.TransferFunction(num, den)
        num = [0.5]
        den = [8672, 259, 1]
        sist3 = signal.TransferFunction(num, den)
        num = [0.3]
        den = [8672, 259, 1]
        sist4 = signal.TransferFunction(num, den)
        #estabelecendo uma quantidade de 3000 valores
        t_v = t[:-cont_time]
        
        #plots referentes a cada questão específica.
        if mode_ == 'normal':
            #plotando as a resposta ao degrau 
            t1, out1 = signal.step(sist1, X0=None, T=t_v, N=None)
            plt.plot(t1,out1, '--')
            t2, out2 = signal.step(sist2, X0=None, T=t_v, N=None)
            plt.plot(t2,out2, '--')
            t3, out3 = signal.step(sist3, X0=None, T=t_v, N=None)
            plt.plot(t3,out3, '--')
            t4, out4 = signal.step(sist4, X0=None, T=t_v, N=None)
            plt.plot(t4,out4, '--')
            plotsignals(x_f,u,t,input_,init_)
        elif mode_ == 'normaliz':
            plotsignals_p(x_f,u,t,input_,init_)
        elif mode_ == 'exc':
            t1, out1 =sist1.output(u, t, X0=None)
            plt.plot(t1,out1[:-cont_time], '--')
            t2, out2 =sist2.output(u, t, X0=None)
            plt.plot(t1,out1[:-cont_time], '--')
            t3, out3 =sist3.output(u, t, X0=None)
            plt.plot(t1,out1[:-cont_time], '--')
            t4, out4 =sist4.output(u, t, X0=None)
            plt.plot(t1,out1[:-cont_time], '--')
            plotsignals(x_f,u,t,input_,init_)

        else:

            w, mag, phase = sist1.bode()
            plt.figure()
            plt.legend('Saida 1 Magnitude')
            plt.semilogx(w, mag)    # Bode magnitude plot
            plt.figure()
            plt.legend('Saida 1 Fase')
            plt.semilogx(w, phase)  # Bode phase plot
            plt.show()
            w, mag, phase = sist2.bode()
            plt.figure()
            plt.legend('Saida 2 Magnitude')
            plt.semilogx(w, mag)    # Bode magnitude plot
            plt.figure()
            plt.legend('Saida 2 Fase')
            plt.semilogx(w, phase)  # Bode phase plot
            plt.show()
            w, mag, phase = sist3.bode()
            plt.figure()
            plt.legend('Saida 3 Magnitude')
            plt.semilogx(w, mag)    # Bode magnitude plot
            plt.figure()
            plt.legend('Saida 3 Fase')
            plt.semilogx(w, phase)  # Bode phase plot
            plt.show()
            w, mag, phase = sist4.bode()
            plt.figure()
            plt.legend('Saida 4 Magnitude')
            plt.semilogx(w, mag)    # Bode magnitude plot
            plt.figure()
            plt.legend('Saida 4 Fase')
            plt.semilogx(w, phase)  # Bode phase plot
            plt.show()
    


if __name__ == "__main__":
    main_TanQua()