import sys, os
import numpy as np
sys.path.insert(0, 'evoman')
from controller import Controller


class PlayerController(Controller):
    def control(self, inputs, controller):
        # Normalises the input using min-max scaling (taken from demo_controller)
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
        output = controller.activate(inputs)

        return np.round(output).astype(int)
