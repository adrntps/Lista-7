import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

I = np.ones(shape=(1,16))
# Definindo um vetor linha com todas as componentes iguais a 1. Útil para 
# impor a normalização do comportamento P(a0,a1,b0,b1) de Fine. 

G = np.zeros(shape=(16,16))
G[0,0]=G[0,1]=G[0,4]=G[0,5]=1
G[1,2]=G[1,3]=G[1,6]=G[1,7]=1
G[2,8]=G[2,9]=G[2,12]=G[2,13]=1
G[3,10]=G[3,11]=G[3,14]=G[3,15]=1
G[4,0]=G[4,2]=G[4,4]=G[4,6]=1
G[5,1]=G[5,3]=G[5,5]=G[5,7]=1
G[6,8]=G[6,10]=G[6,12]=G[6,14]=1
G[7,9]=G[7,11]=G[7,13]=G[7,15]=1
G[8,0]=G[8,1]=G[8,8]=G[8,9]=1
G[9,2]=G[9,3]=G[9,10]=G[9,11]=1
G[10,4]=G[10,5]=G[10,12]=G[10,13]=1
G[11,6]=G[11,7]=G[11,14]=G[11,15]=1
G[12,0]=G[12,2]=G[12,8]=G[12,10]=1
G[13,1]=G[13,3]=G[13,9]=G[13,11]=1
G[14,4]=G[14,6]=G[14,12]=G[14,14]=1
G[15,5]=G[15,7]=G[15,13]=G[15,15]=1
# Definindo a matriz G que codifica restrições lineares de igualdade. Em
# particular, as marginalizações de P(a0,a1,b0,b1) corresponderem às componentes
# de p_{\alpha,\beta}(a,b|x,y). 

Prbox = np.zeros(shape=(16,1))
Prbox[0] = 1/2
Prbox[3] = 1/2
Prbox[4] = 1/2
Prbox[7] = 1/2
Prbox[8] = 1/2
Prbox[11] = 1/2
Prbox[13] = 1/2
Prbox[14] = 1/2
# Definindo o comportamento referente à caixa PR.

PL = np.zeros(shape=(16,1))
PL[0] = 1
PL[4] = 1
PL[8] = 1
PL[12] = 1
# Definindo o comportamento local.

PI = np.full((16,1),1/4)
# Definindo o comportamento isotrópico.

alpha = cp.Variable(nonneg=True)
# Definindo o parâmetro \alpha como uma das variáveis do programa linear.

beta = cp.Parameter(nonneg=True)
# Definindo o parâmetro \beta como um parâmetro a ser variado em um loop.

beta_val = np.linspace(0, 1, num=50)
# Retorna números uniformemente espaçados em um determinado intervalo. Em
# particular, escolheu-se o intervalo de validade do parâmetro \beta; e 
# são criados 50 números, i.e., o loop seguinte correrá por 50 valores
# de \beta no intervalo de 0 a 1.

x=[]
y=[]
# Criando duas listas para receber os valores de \beta e \alpha, respectivamente,
# de forma a se plotar um gráfico do \alpha máximo para o qual o comportamento
# ainda é local em função de \beta.

for val in beta_val:
# O loop corre para cada um dos 50 números gerados anteriormente.
    beta.value = val
    # \beta recebe o valor de "val". Esse passo corresponde ao "dado um valor de
    # \beta, no programa linear.
    x.append(beta.value)
    # Salvando o valor de \beta na lista.
    P_abxy = alpha*Prbox + (1-alpha)*(beta*PL + (1-beta)*PI)
    # Definindo a família de comportamentos para um dado \beta, e em função de
    # \alpha.
    P = cp.Variable(shape=(16,1), nonneg=True)
    # Definindo a segunda variável de otimização; o comportamento P(a0,a1,b0,b1)
    # que deve existir caso "P_abxy" seja local dado um \beta, em função do \alpha.
    objective = cp.Maximize(alpha)
    # Função objetivo: dado um \beta, deve-se encontrar o máximo valor de \alpha
    # de forma que as restrições seguintes sejam satisfeitas.
    constraints = [cp.matmul(G,P) == P_abxy, cp.matmul(I,P) == 1, alpha <= 1]
    # Restrições do programa linear: \alpha <= 1, do enunciado; "cp.matmul(I,P)
    # == 1 impõe normalização ao comportamento P(a0,a1,b0,b1);
    # cp.matmul(G,P) == P_abxy impõe que a marginalização de P(a0,a1,b0,b1) seja
    # igual, componente a componente, a "P_abxy". Em outras palavras, dado um 
    # valor de \beta, o programa linear buscará o máximo valor de \alpha para o
    # qual existe uma probabilidade conjunta P(a0,a1,b0,b1) que recupera 
    # P(\alpha,\beta). Se isso ocorre, pelo teorema de Fine, o comportamento 
    # é local.
    prob = cp.Problem(objective, constraints)
    prob.solve()
    y.append(alpha.value)
    print(alpha.value)
    
plt.plot(x,y)