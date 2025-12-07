import random
import numpy as np

class Agent:
    # Hiperparàmetres per defecte
    alpha = 0.7          
    gamma = 0.99
    epsilon = 0.9        
    decrease_rate = 0.5  

    actions = (0, 1, 2, 3) # Up, Down, Right, Left
    q_table = np.zeros((8, 8, len(actions))) # Inicialització per defecte a 8x8

    # Canviem rows=3, cols=4 a rows=8, cols=8 per defecte
    def __init__(self, rows=8, cols=8, actions=None, learning_rate=None, future_weight=None, exploration_rate=None, decrease_rate=None):
        
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
        if self.alpha > 0.1:
            self.alpha-=self.decrease_rate

    def reduce_exploration_rate_by_10_percent(self):
        if self.epsilon > 0.1:
            self.epsilon-=0.1   
                                                                                                    
    def reduce_exploration_rate_by_decrease_rate(self):
        if self.epsilon > 0.1:
            self.epsilon-=self.decrease_rate

    def think(self, state):
        """
        Decideix l'acció basada en l'estat actual.
        """
        # Eliminem la crida duplicada
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
        Actualització Q-Learning (Bellman Equation)
        """
        row, col = state
        next_r, next_c = next_state
        
        current_q = self.q_table[row, col, action]
        
        if done:
            target = reward
        else:
            max_next_q = np.max(self.q_table[next_r, next_c])
            target = reward + (self.gamma * max_next_q)

        new_q = current_q + self.alpha * (target - current_q)
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
        best_reward = float('-inf')
        best_move = random.choice(self.actions)

        for action in self.actions:
            evaluating_reward = self.q_table[state[0], state[1], action]
            
            if evaluating_reward > best_reward:
                best_reward = evaluating_reward
                best_move = action
                
        return best_move