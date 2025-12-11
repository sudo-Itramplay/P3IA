import random
import numpy as np

class Agent:
    '''
    L'AGETN TREBALLRA AMB KESY OSIGUI TUPLES
    '''

    # Hiperparàmetres per defecte
    alpha = 0.7          # Learning rate - Pes de la nova info sobre la vella
    #gamma = 0.9          # Discount factor - Pes recompenses futures(possibles) respecte a les conegudes i immediates
    gamma = 0.99
    epsilon = 0.9        # Exploration rate - Probabilitat de fer acció aleatoria
    #In de case of the 1.b exercise, we decrease epsilon over time changing the value form 0.1 to 0.5, for a quicker conveergence
    decrease_rate = 0.1  # Epsilon decay - Quant disminuim exploration rate

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
            
        #if rows is not None and cols is not None:
            #self.q_table = {np.zeros((rows, cols, len(self.actions)))}
            #ENTENC QUE NO CAL MANTENIR LA LGOICA DEL GRID, SENZILLAMENT POSOS AIXO I AJ ESTA
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

    def think(self, state):
        """
        Dummy function
        """
        return self.policy(state)

    #A la politica hem d'afegir estats possibles, ja que hem de tenir en compte que hi ha estats no valids en el futur
    #possible_acions_n hauria de ser el nombre daccions possibles en l'estat actual
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
        """
        return self.max_Q(state, possible_actions_n)

    def learn(self, state, action, reward, next_state, done, n_actions_in_state):
        """
        Actualització Q-Learning (Bellman Equation):
        Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
        """
        #mirem que lestat existeixi
        if state not in self.q_table:
            self.q_table[state] = [0.0] * n_actions_in_state
        
        #hauriemde rmodelar la mida si ara tenim mes possibles accions
        if action >= len(self.q_table[state]):
             self.q_table[state].extend([0.0] * (action + 1 - len(self.q_table[state])))
             
        current_q = self.q_table[state][action]
        
        #busquem el max q del seguent estat
        if done:
            target = reward
        else:
            if next_state not in self.q_table or not self.q_table[next_state]:
                max_next_q = 0.0
            else:
                max_next_q = max(self.q_table[next_state])
                
            target = reward + (self.gamma * max_next_q)

        #Calculem el nou valor Q amb el learning rate (alpha)
        new_q = current_q + self.alpha * (target - current_q)
        
        #Actualitzem la taula
        self.q_table[state][action] = new_q

    def explore(self, state):
        """
        Retorna una acció aleatòria.
        """

        """
        if random.random() < self.decrease_rate and self.decrease_rate > 0.1:
            self.epsilon -= 0.05
        """
        return random.choice(self.actions)
        
    def max_Q(self, state, n_actions_in_state):
        '''
        Retorna la millor acció coneguda per a l'estat donat (Argmax).
        
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
        '''
        if state not in self.q_table:
            #estat no vist 
            return self.explore(n_actions_in_state)
            
        q_values = self.q_table[state]

        # Assegurar que el vector de Q-values cobreixi les accions disponibles
        if len(q_values) < n_actions_in_state:
             q_values.extend([0.0] * (n_actions_in_state - len(q_values)))

        #Arecompewnsa maxima sobre les accions disponibles.
        best_move_index = np.argmax(np.array(q_values[:n_actions_in_state]))
        
        return int(best_move_index)
