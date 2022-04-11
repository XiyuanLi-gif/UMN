import os
import gym
import numpy as np
import matplotlib.pyplot as plt

import paddle.fluid as fluid
import parl
from parl import layers
from parl.utils import logger

import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL


from parl.algorithms import PolicyGradient

LEARNING_RATE = 1e-4