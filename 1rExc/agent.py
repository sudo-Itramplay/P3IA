import random
import Qtable

# Definir accions
# entenc k ha de tenir estat
class Agent:
    # learning rate (quant ràpid oblidem el passat)
    alpha = 0.1
    # Què tant important és el futur
    gamma = 0.90
    # Cada quant explirem (fem coses rares, bogeries, coses que no feiem abans)
    epsilon = 0.9
    # Cada quant variem epsilon
    decrease_rate=0.1

    actions = ('up', 'down', 'right', 'left')

    qtable = Qtable(actions)


    def __init__(self, actions=None, learning_rate=None, future_weight=None, exploration_rate=None, decrease_rate=None):
        
        # 1. Canviar el valor de alpha (learning_rate) si se'n proporciona un de nou
        if learning_rate is not None:
            self.alpha = learning_rate

        # 2. Canviar el valor de gamma (future_weight) si se'n proporciona un de nou
        if future_weight is not None:
            self.gamma = future_weight

        # 3. Canviar el valor de epsilon (exploration_rate) si se'n proporciona un de nou
        if exploration_rate is not None:
            self.epsilon = exploration_rate

        # 5. Canviar el valor del decrease_rate de epsilon si se'n proporciona un de nou    
        if decrease_rate is not None:
            self.decrease_rate = decrease_rate

        # 5. Canviar el valor de actions si se'n proporciona un de nou
        if actions is not None:
            self.actions = actions
            
        # 6. Inicialitzar la Q-table amb el conjunt d'accions final
        self.qtable = Qtable(self.actions)


    def policy(self, state, env):
        """
        Mètode que executarà l'acció. Epsilon greedy?
        """
        if random.random() < self.epsilon:
            # Exploració
            return self.explore(state, env)
        else:
            # Explotació
            return self.max_Q()

        return "a"

    def learn(self, state, env):
        """
        Mètode que apendra de l'acció feta. Bellman
        """
               
        return "b"
    
    def explore(self, state, env):
        rand=random.random()

        if rand < 0.25:
            return self.actions[0]
        elif rand < 0.5:
            return self.actions[1]
        elif rand < 0.75:
            return self.actions[2]
        else:
            return self.actions[3]
        
    def max_Q(self):
        
        return



    