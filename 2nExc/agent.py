import random
import numpy as np
import json
import ast

class Agent:
    alpha = 0.7      # Learning rate          
    gamma = 0.99     # Future reward weight
    epsilon = 0.9    # Exploration rate       

    #In de case of the 1.b exercise, we decrease epsilon over time changing the value form 0.1 to 0.5, for a quicker conveergence
    decrease_rate = 0.1 
    actions = (0, 1, 2, 3) # Up, Down, Right, Left
    q_table= {}

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

        self.q_table = {}
    
    # For practice
    def getQtable(self):
        return self.q_table

    def reduce_learning_rate_by_10_percent(self):
        if self.alpha > 0.1:
            self.alpha-=0.1   
                                                                                                    
    def reduce_learning_rate_by_decrease_rate(self):
        # Could make it random so it's not that aggressive, same as with the policy when doing explore or maxQ
        if self.alpha > 0.1:
            self.alpha-=self.decrease_rate

    def reduce_exploration_rate_by_30_percent(self):
        if self.epsilon > 0.1:
            self.epsilon-=0.3   
                                                                                                    
    def reduce_exploration_rate_by_decrease_rate(self):
        if self.epsilon > 0.1:
            self.epsilon-=self.decrease_rate

    def think(self, state):
        """
        Dummy function for better readability
        """
        return self.policy(state)
    
    def policy(self, state, possible_actions_n):
        """
        Epsilon-Greedy Policy.
        """
        if random.random() < self.epsilon:
            return self.explore(possible_actions_n)
        else:
            return self.max_Q(state, possible_actions_n)
        
    def policy_last_iteration(self, state, possible_actions_n):
        """
        Epsilon-Greedy Policy.
        To get the best path that the ai can do in the last iteration
        """
        return self.max_Q(state, possible_actions_n)

    def learn(self, state, action, reward, next_state, done, n_actions_in_state):
        """
        Bellman Equation Implementation
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0] * n_actions_in_state
        
        if action >= len(self.q_table[state]):
             self.q_table[state].extend([0.0] * (action + 1 - len(self.q_table[state])))
             
        current_q = self.q_table[state][action]
        

        if done:
            target = reward
        else:
            if next_state not in self.q_table or not self.q_table[next_state]:
                max_next_q = 0.0
            else:
                max_next_q = max(self.q_table[next_state])
                
            target = reward + (self.gamma * max_next_q)

        new_q = current_q + self.alpha * (target - current_q)
        
        self.q_table[state][action] = new_q

        # To monitor the change in Q-value
        return abs(new_q - current_q)

    def explore(self, state):
        return random.choice(self.actions)
        
    def max_Q(self, state, n_actions_in_state):
        if state not in self.q_table:
            return self.explore(n_actions_in_state)
            
        q_values = self.q_table[state]

        if len(q_values) < n_actions_in_state:
             q_values.extend([0.0] * (n_actions_in_state - len(q_values)))

        best_move_index = np.argmax(np.array(q_values[:n_actions_in_state]))
        
        return int(best_move_index)
    
    """
    -----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    ------------------------------------ <Mètodes practica qtable> --------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    -----------------------------------------------------------------------------------------------------------
    """   

    def save_qtable_to_json(self, filename="qtable.json"):
        """
        Save the agent's Q-table to a JSON file.
        Converts the tuple keys to strings for JSON serialization.
        """
        serializable_qtable = {str(k): v for k, v in self.q_table.items()}
        
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_qtable, f, indent=4)
            print(f"Q-table guardada correctament a {filename}")
        except IOError as e:
            print(f"Error guardant la Q-table: {e}")

    def load_qtable_from_json(self, filename="qtable.json"):
        """
        Load the agent's Q-table from a JSON file.
        Converts the string keys back to tuples after loading.
        """
        try:
            with open(filename, 'r') as f:
                loaded_data = json.load(f)
            
            # Reconstruïm el diccionari convertint les claus de string a tuples
            self.q_table = {ast.literal_eval(k): v for k, v in loaded_data.items()}
            print(f"Q-table carregada correctament des de {filename}")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error carregant la Q-table: {e}")
