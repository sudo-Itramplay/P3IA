import numpy as np
class Qtable:
    # fer estructura de dades clau (estat, acció)
    table = np.array()

    def __init__(self, actions):
        n = len(actions)
        self.table = np.full((n, n), 1)

    def __init__(self, table):
        
        self.table = table