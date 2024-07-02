import numpy as np
import torch.nn as nn 

class PolicyNetwork():
    def __init__(self, observation_dim, action_dim):
        self.hiddenspace_1 = 64
        self.hiddenspace_2 = 32

        self.model = nn.Sequential(
            nn.Linear(observation_dim, self.hiddenspace_1),
            nn.ReLU(), 
            nn.Linear(self.hiddenspace_1, self.hiddenspace_2),
            nn.ReLU(), 
            nn.Linear(self.hiddenspace_2, action_dim), 
            nn.ReLU()
        )

        # will give us the action probabilities 

    def forward(self, x): 
        return self.model.forward(x)