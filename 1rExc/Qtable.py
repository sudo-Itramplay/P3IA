import numpy as np
class Qtable:
    # fer estructura de dades clau (estat, acció)
    table = np.array()
    actions = None

    def __init__(self, actions):
        n = len(actions)
        self.actions = actions
        self.table = np.full((n, n), 0)

    def __init__(self, table):
        
        self.table = table

    def lookup(self, state, action):
        reward = 0
        match action:
            case self.actions[0]:
                reward = self.table[state[1], (state[0]+1)]

            case self.actions[1]:
                reward = self.table[state[1], (state[0]-1)]

            case self.actions[2]:
                reward = self.table[(state[1]+1), state[0]]

            case self.actions[3]:
                reward = self.table[(state[1]-1), state[0]]

            case _:
                print("No action asociated")  
