import random
import numpy as np

class Agent:
    # Hiperparàmetres per defecte
    alpha = 0.7          # Learning rate - Pes de la nova info sobre la vella
    gamma = 0.99
    epsilon = 0.9        # Exploration rate - Probabilitat de fer acció aleatoria
    decrease_rate = 0.5  # Epsilon decay - Quant disminuim exploration rate

    actions = (0, 1, 2, 3)
    q_table= {} #We had to implement a dictianry for the Qtable, the states of chess are way more complex than indexes of a matrix

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
            self.q_table = {}

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

    def think(self, state, possible_action):
        """
        This function has been completly changed to adapt to the new Q-table structure using a dictionary.
        """
        if not possible_action:
            return None 

        # 1. Initialize the current state in the Q-table if new
        if state not in self.q_table:
            self.q_table[state] = {
                next_s: 0.0 for next_s in possible_action
            }
        
        # 2. Epsilon-Greedy Policy.
        if random.random() < self.epsilon:
            # Explore: choose a random next state (action)
            return random.choice(possible_action)
        else:
            # Exploit: choose the next state (action) with max Q-value
            return self.max_Q(state, possible_action)

    def policy(self, state):
        """
        Epsilon-Greedy Policy.
        """
        if random.random() < self.epsilon:
            return self.explore(state)
        else:
            return self.max_Q(state)

    def learn(self, state, action, reward, next_state, done, possible_actions_next_state):
        """
        Actualització Q-Learning (Bellman Equation):
        Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
        
        For chess, action_string is the resulting state string.
        """
        
        # 1. Ensure current state and action are in the table
        if state not in self.q_table:
             self.q_table[state] = {}
        if action not in self.q_table[state]:
             self.q_table[state][action] = 0.0
             
        current_q = self.q_table[state][action]
        
        # 2. Calculate the max Q for the next state (s')
        if done:
            max_next_q = 0.0 # Terminal state, no future reward
        else:
            if next_state not in self.q_table:
                # If s' is new, initialize it and max Q(s', a') = 0.0
                self.q_table[next_state] = {
                    next_s: 0.0 for next_s in possible_actions_next_state 
                }
                max_next_q = 0.0
            else:
                # Find the max Q-value from the dictionary of next state's actions
                max_next_q = np.max(list(self.q_table[next_state].values()))


        # 3. Calculate the new Q-value using the Bellman equation
        target = reward + (self.gamma * max_next_q)
        new_q = current_q + self.alpha * (target - current_q)
        
        # 4. Update the Q-table
        self.q_table[state][action] = new_q

    def explore(self, state):
        """
        Retorna una acció aleatòria.
        """
        if random.random() < self.decrease_rate and self.decrease_rate > 0.1:
            self.epsilon -= 0.05
        return random.choice(self.actions)
        
    def max_Q(self, state, possible_action):
        """
        Retorna la millor acció (next_state_string) coneguda per a l'estat donat (Argmax).
        """
        best_reward = -float('inf')
        
        # Default move is a random one from the legal actions in case all Q-values are equal
        best_move = random.choice(possible_action) 

        for action_string in possible_action:
            # .get(action_string, 0.0) safely retrieves the Q-value, defaulting to 0.0 for new actions
            evaluating_reward = self.q_table[state].get(action_string, 0.0)
            
            if evaluating_reward > best_reward:
                best_reward = evaluating_reward
                best_move = action_string
            # Tie-breaking for exploration
            elif evaluating_reward == best_reward and random.random() < 0.1:
                best_move = action_string
                
        return best_move
