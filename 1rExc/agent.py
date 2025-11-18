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


    def policy():
        """
        Mètode que executarà l'acció. Epsilon greedy?
        """
        if random.random() < self.epsilon:
            # Exploració
            return acció_aleatòria()
        else:
            # Explotació
            return acció_amb_max_Q(state)

        return "a"

    def learn():
        """
        Mètode que apendra de l'acció feta. Bellman
        """
               
        return "b"



    