import numpy as np

class Qtable:
    # Definim les accions per defecte
    actions = (0, 1, 2, 3)
    
    # Inicialització per defecte (3 files, 4 columnes, 4 accions)
    table = np.zeros((3, 4, 4)) 

    def __init__(self, actions=None, table=None):
        # Gestió d'errors bàsica
        if actions is None and table is None:
            # Si no passem res, assumim els valors per defecte de la classe
            pass 

        # 1. Si ens passen una taula existent, la fem servir
        if table is not None:
            self.table = table

        # 2. Si ens passen accions, reconfigurem la taula
        if actions is not None:
            self.actions = actions
            # Obtenim dimensions actuals (files, columnes)
            rows, cols, _ = self.table.shape 
            # Reiniciem la taula amb el nou nombre d'accions
            self.table = np.zeros((rows, cols, len(actions)))

    def get_q_value(self, state, action_index):
        # Retorna el valor d'una cel·la concreta
        row, col = state
        return self.table[row, col, action_index]

    def update_q_value(self, state, action_index, new_value):
        # Escriu el nou valor après
        row, col = state
        self.table[row, col, action_index] = new_value

    def learn(self, state):
        pass

    def lookup(self, state, action):
        """
        Retorna el valor Q per a un estat i una acció donats.
        """
        row, col = state
        
        reward = self.table[row, col, action]
        
        return reward