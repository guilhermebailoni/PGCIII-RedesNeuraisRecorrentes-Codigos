import numpy as np
import pandas as pd
from tensorflow.python.keras.backend import sqrt

def normalizaTrain(dados):
  data = pd.DataFrame(dados)
  print(data.describe())
  hO = max(data['Open'])
  lO = min(data['Open'])
  hH = max(data['High'])
  lH = min(data['High'])
  hL = max(data['Low'])
  lL = min(data['Low'])
  hC = max(data['Close'])
  lC = min(data['Close'])
  hA = max(data['Adj Close'])
  lA = min(data['Adj Close'])
  hV = max(data['Volume'])
  lV = min(data['Volume'])
  minValues = {"Open":[lO],
              "High":[lH],
              "Low":[lL],
              "Close":[lC],
              "Adj Close":[lA],
              "Volume":[lV]}
  minMaxValues = {"Open":[hO-lO],
              "High":[hH-lH],
              "Low":[hL-lL],
              "Close":[hC-lC],
              "Adj Close":[hA-lA],
              "Volume":[hV-lV]}
  minData = pd.DataFrame(minValues)
  minMaxData = pd.DataFrame(minMaxValues)
  data = data.sub(minData.iloc[0])
  data = data.div(minMaxData.iloc[0])
  return data,minData,minMaxData

def normalizaTest(dados, minData, minMaxData):
  data = dados.copy()
  data = data.sub(minData.iloc[0])
  data = data.div(minMaxData.iloc[0])
  return data

def desnormalizaData(dados, minData, minMaxData):
  data = dados.copy()
  data = data.mul(minMaxData.iloc[0])
  data = data.add(minData.iloc[0])
  return data

def desnormalizaSaida(dados, minData, minMaxData):
  data = dados.copy()
  data = np.multiply(data,minMaxData)
  data = np.add(data,minData)
  return data

#Usado para criar um conjunto com o formato adequado para o aprendizado nas RNNs e LSTMs
def prepareData(dataX, dataY, memory):
  x,y = [],[]
  for i in range(len(dataX)-memory):
    temp = dataX[i:(i+memory)]
    x.append(temp)
    y.append(dataY[(i+memory-1)])
  return np.array(x),np.array(y)

def mae(predicted, expected):
  #erro medio absoluto
  erro = 0
  for i in range(len(predicted)):
    erro = erro + abs(predicted[i]-expected[i])
  return (erro/len(predicted))

def rmse(predicted, expected):
  #root mean squared error
  erro = 0
  for i in range(len(predicted)):
    erro = erro + (predicted[i]-expected[i])**2
  return (erro/len(predicted))**0.5

#TECHNICAL INDICATORS

#Highest and Lowest functions
#returns array with the highest and lowest for each element acording to the memory chosen.

def highest(high, memory):
  maxV = lambda high, index, memory: max(high[index : (index+memory)])
  reversedCP = np.flip(np.array(high))
  return np.flip(
    np.array(
      [maxV(reversedCP, i, memory) for i,x in enumerate(reversedCP)]))

def lowest(low, memory):
  minV = lambda low, index, memory: min(low[index : (index+memory)])
  reversedCP = np.flip(np.array(low))
  return np.flip(
    np.array(
      [minV(reversedCP, i, memory) for i,x in enumerate(reversedCP)]))

#Moving average functions

#Simple moving average
def sma(closingPrice, memory):
  sma_value = lambda closingPrice, index, memory: (
    sum(closingPrice[index:(index+memory)])/memory 
    if (index+memory)<=len(closingPrice)
    else sum(closingPrice[index:(index+memory)])/len(closingPrice[index:(index+memory)]))
  reversedCP = np.flip(np.array(closingPrice))
  return np.flip(
    np.array(
      [sma_value(reversedCP, i, memory) for i,x in enumerate(reversedCP)]))

def wma_step(closingPrice, index, memory):
  p = (index+memory)
  available_memory = memory if p < len(closingPrice) else (len(closingPrice) - index)
  soma = 0
  for i in range(index,(available_memory+index)):
    soma = soma + (closingPrice[i]*((index+available_memory)-i))
  return (soma/((available_memory*(available_memory+1))/2))

#Weighted moving average: gives more weight to more recent observations
def wma(closingPrice, memory):
  reversedCP = np.flip(np.array(closingPrice))
  return np.flip(
    np.array(
      [wma_step(reversedCP, i, memory) for i,x in enumerate(reversedCP)]))

#Exponential moving average: gives more weight to more recent observations, resulting in faster response
def ema_step(current_price, previous_ema, memory):
  alpha = 2/(memory + 1)
  return (current_price*alpha) + (1-alpha)*previous_ema

def ema(closingPrice, memory):
  ema_vector = sma(closingPrice[0:memory],memory)
  current_ema = ema_vector[(memory-1)]
  for i in range((memory),len(closingPrice)):
    current_ema = ema_step(closingPrice[i],current_ema,memory)
    ema_vector = np.append(ema_vector,current_ema)
  return ema_vector

#hull moving average: faster response and better suavization
def hma(closingPrice,memory):
  wma1 = 2 * wma(closingPrice,int(memory/2))
  wma2 = wma(closingPrice, memory)
  
  return wma((wma1-wma2), int(memory**0.5))


def macd(closingPrice,closerMemory,longerMemory):
  return (ema(closingPrice,closerMemory) - ema(closingPrice,longerMemory))

def acd(closingPrice,high,low,volume):
  current_acd = 0
  acd_vector = []
  for i in range(len(closingPrice)):
    current_acd = current_acd + (volume[i]*(((closingPrice[i]-low[i]) - (high[i]-closingPrice[i]))/(high[i] - low[i])))
    acd_vector = np.append(acd_vector,current_acd)
  return acd_vector

#considering high and low version
# def so_k(closingPrice, high, low):
#   highest_vector = highest(high,5)
#   lowest_vector = lowest(low,5)
  
#   current_so_k = 0
#   so_k_vector = []
#   for i in range(len(closingPrice)):
#     current_so_k = (closingPrice[i] - lowest_vector[i])/(highest_vector[i]-lowest_vector[i])
#     so_k_vector = np.append(so_k_vector,current_so_k)
#   return so_k_vector

# def so_d(closingPrice, high, low):
#   return sma(so_k(closingPrice, high, low),3)

def so_k(closingPrice):
  highest_vector = highest(closingPrice,5)
  lowest_vector = lowest(closingPrice,5)
  
  current_so_k = 0
  so_k_vector = []
  for i in range(len(closingPrice)):
    current_so_k = (closingPrice[i] - lowest_vector[i])/(highest_vector[i]-lowest_vector[i])
    so_k_vector = np.append(so_k_vector,current_so_k)
  return so_k_vector

def so_d(closingPrice):
  return sma(so_k(closingPrice),3)

#on-balance volume indicator
def obv(closingPrice, volume):
  current_obv = 0
  obv_vector = np.array([0])
  for i in range(1,len(closingPrice)):
    if closingPrice[i-1] == closingPrice[i]:
      #symbolic implementation
      current_obv = current_obv + 0
    elif closingPrice[i-1] < closingPrice[i]:
      current_obv = current_obv + volume[i]
    else:
      current_obv = current_obv - volume[i]       
    
    obv_vector = np.append(obv_vector, current_obv)
  return obv_vector

#detrended price oscillator
def dpo(closingPrice, memory):
  sma_vector = sma(closingPrice,memory)
  dpo_vector = np.array([0])
  for i in range(1,len(closingPrice)):
    steps_back = int(memory/2) + 1
    choosen_closingPrice = closingPrice[(i-steps_back)] if (i-steps_back) >= 0 else 0
    current_dpo = choosen_closingPrice - sma_vector[i]
    dpo_vector = np.append(dpo_vector, current_dpo)
  return dpo_vector

def ag_step(price):
  fractional_gains = 0
  for i in range(1,len(price)):
    if price[i] >= price[(i-1)]:
      fractional_gains = fractional_gains + ((price[i]/price[(i-1)])-1)
  return (fractional_gains/len(price))

def average_gain(closingPrice, memory):
  reversedCP = np.flip(np.array(closingPrice))
  
  ag_vector = []
  for i in range(len(closingPrice)):
    ag_vector = np.append(ag_vector, ag_step(reversedCP[i:(memory+i)]))
      
  return np.flip(ag_vector)

def al_step(price):
  fractional_loss = 0
  for i in range(1,len(price)):
    if price[i] <= price[(i-1)]:
      fractional_loss = fractional_loss + abs((price[i]/price[(i-1)])-1)

  return (fractional_loss/len(price))

def average_loss(closingPrice, memory):
  reversedCP = np.flip(np.array(closingPrice))
  
  al_vector = []
  for i in range(len(closingPrice)):
    al_vector = np.append(al_vector, al_step(reversedCP[i:(memory+i)]))

  return np.flip(al_vector)
      
def rsi(closingPrice, memory):
  ag_vector = average_gain(closingPrice,memory)
  al_vector = average_loss(closingPrice,memory)
  
  rs = ag_vector/al_vector #numpy arrays so therefore can use element wise division
  
  rsi_vector = []
  
  for i in range(len(closingPrice)):
    current_rsi = 100 - (100/(1+rs[i]))
    rsi_vector = np.append(rsi_vector, current_rsi)
  return rsi_vector