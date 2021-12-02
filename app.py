import streamlit as st
import pandas as pd
import importlib
import logging
import numpy as np
from utils import *

# Streamlit App
st.title('Deep Reinforcement Learning for Automated Stock Trader')
st.subheader('Model uses DQN to generate optimal trades for any given data')

stocks = ['FB_2018', 'BABA_2018', '^GSPC_2016', 'NasdaqComposite']
models = ['DQN_best_ep4', 'DQN_ep10']
initial_balance = st.sidebar.slider('Initial Balance', 25000, 100000, 25000)
# balances = [25000, 50000, 100000]
stock_name = st.sidebar.selectbox('Stock Name:', stocks)
model_to_load = st.sidebar.selectbox('Model:', models)
# initial_balance = st.sidebar.selectbox('Initial Balance:', balances)
model_name = model_to_load.split('_')[0]

@st.cache
def load_data_(symbol, window_size):
    prices = pd.read_csv("data/" + symbol + ".csv", nrows=window_size)
    return prices

prices = load_data_(stock_name, 10)
st.write(prices)

submit = st.sidebar.button('Run')

window_size = 5
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

# select evaluation model
model = importlib.import_module(f'agents.{model_name}')

def hold():
    logging.info('Hold')

def buy(t):
    agent.balance -= stock_prices[t]
    agent.inventory.append(stock_prices[t])
    agent.buy_dates.append(t)
    logging.info('Buy:  ${:.2f}'.format(stock_prices[t]))

def sell(t):
    agent.balance += stock_prices[t]
    bought_price = agent.inventory.pop(0)
    profit = stock_prices[t] - bought_price
    global reward
    reward = profit
    agent.sell_dates.append(t)
    logging.info('Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit))

# configure logging
logging.basicConfig(filename=f'logs/{model_name}_evaluation_{stock_name}.log', filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s', 
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logging.info("balance ",initial_balance, " model name ", model_to_load)
portfolio_return = 0
if submit:
    while portfolio_return == 0: # a hack to avoid stationary case
        agent = model.Agent(state_dim=8, balance=initial_balance, is_eval=True, model_name=model_to_load)
        stock_prices = stock_close_prices(stock_name)
        logging.info("Agent inventory ", len(agent.inventory))
        trading_period = len(stock_prices) - 1
        state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))

        for t in range(1, trading_period + 1):
            if model_name == 'DDPG':
                actions = agent.act(state, t)
                action = np.argmax(actions)
            else:
                actions = agent.model.predict(state)[0]
                action = agent.act(state)

            # print('actions:', actions)
            # print('chosen action:', action)

            next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
            previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
            
            # execute position
            logging.info(f'Step: {t}')
            if action != np.argmax(actions): logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
            if action == 0: hold() # hold
            if action == 1 and agent.balance > stock_prices[t]: buy(t) # buy
            if action == 2 and len(agent.inventory) > 0: sell(t) # sell

            current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
            agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
            agent.portfolio_values.append(current_portfolio_value)
            state = next_state

            done = True if t == trading_period else False
            if done:
                portfolio_return = evaluate_portfolio_performance(agent, logging)
    plot_all(stock_name, agent)