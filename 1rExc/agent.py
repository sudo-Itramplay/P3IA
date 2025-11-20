import random
from Qtable import Qtable 

class Agent:
    # Hiperparàmetres per defecte
    alpha = 0.1          # Learning rate
    gamma = 0.90         # Discount factor
    epsilon = 0.9        # Exploration rate
    decrease_rate = 0.1  # Epsilon decay (opcional per a futures millores)

    actions = (0, 1, 2, 3) # Up, Down, Right, Left

    def __init__(self, actions=None, learning_rate=None, future_weight=None, exploration_rate=None, decrease_rate=None):
        
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
            
        # Inicialitzem la Q-table (Instància de la classe Qtable)
        self.qtable = Qtable(self.actions)

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
        # 1. Obtenim el Q-value actual
        current_q = self.qtable.lookup(state, action)
        
        # 2. Calculem el max Q per al següent estat (s')
        if done:
            # Si hem acabat, no hi ha futur, només la recompensa final
            target = reward
        else:
            # Busquem el valor màxim possible des del següent estat
            max_next_q = float('-inf')
            for a in self.actions:
                q_val = self.qtable.lookup(next_state, a)
                if q_val > max_next_q:
                    max_next_q = q_val
            
            target = reward + (self.gamma * max_next_q)

        # 3. Calculem el nou valor Q amb el learning rate (alpha)
        new_q = current_q + self.alpha * (target - current_q)
        
        # 4. Actualitzem la taula
        self.qtable.update_q_value(state, action, new_q)

    def explore(self, state):
        """
        Retorna una acció aleatòria.
        """
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
            evaluating_reward = self.qtable.lookup(state, action)
            
            if evaluating_reward > best_reward:
                best_reward = evaluating_reward
                best_move = action
                
        return best_move