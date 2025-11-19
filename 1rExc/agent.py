import random

# Definir accions
# entenc k ha de tenir estat
class Agent:
    #learning rate (quant ràpid oblidem el passat)
    alpha=0.1
    # Què tant important és el futur
    gamma=0.90
    # Cada quant explirem (fem coses rares, bogeries, coses que no feiem abans)
    epsilon=0.9

    actions=('up', 'down', 'right', 'left')


    def policy(self):
        """
        Mètode que executarà l'acció. Epsilon greedy?
        """
        if random.random() < self.epsilon:
            # Exploració
            return self.explore()
        else:
            # Explotació
            return self.max_Q()

        return "a"

    def learn(self):
        """
        Mètode que apendra de l'acció feta. Bellman
        """
               
        return "b"
    
    def explore(self):
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
        return 'up'



    