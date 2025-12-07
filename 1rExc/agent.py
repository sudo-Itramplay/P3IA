import random
import numpy as np

class Agent:
    # Hiperparàmetres per defecte
    alpha = 0.7          # Learning rate - Pes de la nova info sobre la vella
    #gamma = 0.9          # Discount factor - Pes recompenses futures(possibles) respecte a les conegudes i immediates
    gamma = 0.99
    epsilon = 0.9        # Exploration rate - Probabilitat de fer acció aleatoria
    #In de case of the 1.b exercise, we decrease epsilon over time changing the value form 0.1 to 0.5, for a quicker conveergence
    decrease_rate = 0.1  # Epsilon decay - Quant disminuim exploration rate

    actions = (0, 1, 2, 3) # Up, Down, Right, Left
    q_table= np.zeros((3, 4, len(actions)))

    def __init__(self, rows=3, cols=4, actions=None, learning_rate=None, future_weight=None, exploration_rate=None, decrease_rate=None):
        
        if learning_rate is not None:
            self.alpha = learning_rate

        if future_weight is not None:
            self.gamma = future_weight

        if exploration_rate is not None:
            self.epsilon = exploration_rate
 
        if decrease_rate is not None:
            self.decrease_rate = decrease_rate

        if actions is not None:
            self.actions = actions
            
        if rows is not None and cols is not None:
            self.q_table = np.zeros((rows, cols, len(self.actions)))

    # For practice
    def getQtable(self):
        return self.q_table

    def reduce_learning_rate_by_10_percent(self):
        if self.alpha > 0.1:
            self.alpha-=0.1   
                                                                                                    
    def reduce_learning_rate_by_decrease_rate(self):
        # TODO es podria fer aleatori pq no fos tant agressiu, igual que amb el policy alhora de fer explore o maxQ
        if self.alpha > 0.1:
            self.alpha-=self.decrease_rate

    def reduce_exploration_rate_by_10_percent(self):
        if self.epsilon > 0.1:
            self.epsilon-=0.1   
                                                                                                    
    def reduce_exploration_rate_by_decrease_rate(self):
        # TODO es podria fer aleatori pq no fos tant agressiu, igual que amb el policy alhora de fer explore o maxQ
        if self.epsilon > 0.1:
            self.epsilon-=self.decrease_rate

    def think(self, state):
        """
        Decideix l'acció basada en l'estat actual.
        """
        action = self.policy(state)
        return self.policy(state)

    def policy(self, state):
        """
        Epsilon-Greedy Policy.
        """
        if random.random() < self.epsilon:
            return self.explore(state)
        else:
            return self.max_Q(state)

    def learn(self, state, action, reward, next_state, done):
        """
        Actualització Q-Learning (Bellman Equation):
        Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
        """
        row, col = state
        next_r, next_c = next_state
        # 1. Obtenim el Q-value actual
        current_q = self.q_table[row, col, action]
        
        # 2. Calculem el max Q per al següent estat (s')
        if done:
            # Si hem acabat, no hi ha futur, només la recompensa final
            target = reward
        else:
            # Busquem el valor màxim possible des del següent estat
            # Max Q per al següent estat
            max_next_q = np.max(self.q_table[next_r, next_c])
            target = reward + (self.gamma * max_next_q)

        # 3. Calculem el nou valor Q amb el learning rate (alpha)
        new_q = current_q + self.alpha * (target - current_q)
        
        # 4. Actualitzem la taula
        self.q_table[row, col, action] = new_q

    def explore(self, state):
        """
        Retorna una acció aleatòria.
        """
        if random.random() < self.decrease_rate and self.decrease_rate > 0.1:
            self.epsilon -= 0.05
        return random.choice(self.actions)
        
    def max_Q(self, state):
        """
        Retorna la millor acció coneguda per a l'estat donat (Argmax).
        """
        # Inicialitzem amb un valor molt baix
        best_reward = float('-inf')
        
        # Per defecte triem una a l'atzar per si totes són iguals
        best_move = random.choice(self.actions)

        for action in self.actions:
            # Cridava a la funció lookup de la classe externa Qtable
            evaluating_reward = self.q_table[state[0],state[1], action]
            
            if evaluating_reward > best_reward:
                best_reward = evaluating_reward
                best_move = action
                
        return best_move
