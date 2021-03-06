
import pandas as pd
import  matplotlib
import matplotlib.pyplot as plt
from dqn_agent import Agent
import torch
import numpy as np
from main import getState


STATE_SIZE = 10

df = pd.read_csv('601211_eval.txt', encoding='utf-8', sep='\t')
print(df['close'])
df['date'] = pd.to_datetime(df['date'])
df.set_index("date", inplace=True)


agent = Agent(state_size=STATE_SIZE, action_size=3)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))


stockData = list(df['close'])
l = len(stockData)-1
window_size = 10
state = getState(stockData, 0, window_size + 1)
total_profit = 0
agent.inventory = []
action_list = []
value_list = []
for t in range(l):
    action = agent.act(state, eps=0)
    next_state = getState(stockData, t + 1, STATE_SIZE + 1)
    if action == 1:# buy
        agent.inventory.append(stockData[t])
                # print("buy" + str(stockData[t]))
    elif action == 2 and len(agent.inventory) > 0: # sell
        bought_price = agent.inventory.pop(0)
        total_profit += stockData[t] - bought_price
    done = 1 if t == l - 1 else 0
    state = next_state
    action_list.append(action)
    value_list.append(stockData[t])
    if done:
        print("------------------------------")
        print("total_profit = " + str(total_profit))
        print("------------------------------")
        #plt.plot(np.arange(len(value_list)), value_list)
        action_list.append(0)
        df['action'] = pd.DataFrame(action_list).values

    #plt(x=df.index[i], y=df['action'][i], c=color)
        df["close"].plot(figsize=(8, 5), grid=True)
    #plt.plot(x=df.index.values, y=df["close"])
    #print(df)
        sell = (df['action'].values == 2)
        plt.scatter(df.index[sell], df["close"].values[sell], c='r')
        buy = (df['action'].values == 1)
        plt.scatter(df.index[buy], df["close"].values[buy], c='b')
        plt.legend(['value', 'sell', 'buy'])
        plt.show()


