import sys
sys.path.insert(0, 'evoman')
from environment import Environment
import numpy as np


class CustomEnvironment(Environment):
    def fitness_single(self):
        return (100-self.get_enemylife())*1-(100-self.get_playerlife())*2-(np.log(self.get_time()))*2