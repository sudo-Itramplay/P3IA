import agent 
import aichess 
import matplotlib.pyplot as plt
import numpy as np
def plot_convergence(rewards_history, iterations,conv_point, filename='convergence_plot.png'):
    """
    Genera un gràfic de la convergència (recompensa per episodi) i el guarda en un fitxer.
    
    Args:
        rewards_history (list): Llista amb la recompensa total de cada episodi.
        window_size (int): Mida de la finestra per calcular la mitjana mòbil.
        filename (str): Nom del fitxer on es guardarà la imatge.
    """
    plt.figure(figsize=(10, 6))
    
    # Gràfic de les recompenses brutes (en blau clar i transparent per veure la variància)
    plt.plot(rewards_history, color='red', alpha=0.6, label='Convergencia per episodi')
    
    """    # Càlcul i gràfic de la mitjana mòbil per veure la tendència (en blau fosc)
    if len(rewards_history) >= iterations:
        moving_average = np.convolve(rewards_history, np.ones(iterations)/iterations, mode='valid')
        # Ajustem l'eix X perquè la mitjana coincideixi amb el final de la finestra
        plt.plot(range(iterations-1, len(rewards_history)), moving_average, color='blue', linewidth=2, label=f'Mitjana mòbil ({iterations})')
    """
    
    plt.axvline(x=conv_point, color='blue', linestyle='--', linewidth=2, label=f'Marca punt on es troba la convergencia (x={conv_point})')
    plt.title("Convergència de l'Agent: Convergencia vs Episodis")
    plt.xlabel("Episodis")
    plt.ylabel("Convergencia")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar el gràfic
    plt.savefig(filename)
    print(f"Gràfic guardat a {filename}")
    plt.show() # Descomenta això si l'executes en local i vols veure la finestra

def check_convergence(rewards, min_episodes=50):
    """
    Calcula la convergència.
    min_episodes: Nombre mínim d'episodis abans de comprovar si s'ha estancat.
    """
    if len(rewards) == 0:
        return 0.0, -1
    
    iterations = -1
    conv_list=[]
    # Comencem a 'min_episodes' o a 1, el que sigui més gran.
    start_index = 1 if min_episodes < 50 else min_episodes
    avg_best = 0.0
    
    for i in range(len(rewards)-1):
        if i > start_index and avg_best == 0.0:
            if (rewards[i-1]<0 or rewards[i]<0 or rewards[i+1]<0):
                continue  # Ignorem estancaments en recompenses negatives
            

            # Comprovem si hi ha 3 valors idèntics consecutius
            variation1 = rewards[i-1] - rewards[i]
            variation2 = rewards[i] - rewards[i+1]

            if -20 <= variation1 <= 20 and -20 <= variation2 <= 20:
                # Només considerem estancament si el reward és raonable   
                if rewards[i] > 0: 
                    avg_best = sum(rewards[i-2:i+2]) / i
                    conv_point = i
                
        
        if i >= 2:
            conv_list.append(abs(sum(rewards[i-2:i+2]) / i))
            
    # Si no troba convergència, retornem el total d'episodis com a "temps de convergència"
    if iterations == -1:
        iterations = len(rewards)
            
    return avg_best, conv_point, conv_list


def check_convergence_final(loss_history, rewards, min_episodes=50, loss_thresh=1.0, reward_std_thresh=10.0):
    """
    Calcula la convergència basant-se en l'estabilitat del 'loss' (aprenentatge) 
    i l'estabilitat de la recompensa (resultat).
    """
    if len(rewards) < min_episodes:
        return 0.0, -1, []
    
    avg_best = 0.0
    conv_point = None
    conv_list = [] 
    
    # Iterem a partir de min_episodes per tenir sempre una finestra completa
    for i in range(len(rewards)):
        
        # 1. Definir la finestra lliscant (els últims 'min_episodes')
        # Si i < min_episodes, agafem des de l'inici fins a i (encara que sigui petita)
        start_idx = max(1, i - min_episodes)
        current_window_rewards = rewards[start_idx:i+1]
        current_window_loss = loss_history[start_idx:i+1]
        
        # Calculem mètriques de la finestra actual
        std_reward = np.std(current_window_rewards)
        avg_loss = np.mean(current_window_loss)
        current_avg_reward = np.mean(current_window_rewards) if len(current_window_rewards) > 0 else 0.0

        conv_list.append(std_reward)

        # 2. Comprovació de Convergència (només si tenim prou històric)
        if i >= min_episodes and conv_point is None:
            
            # A. El cervell està estable (Loss baix -> l'agent ja no canvia gaire la seva Q-table)
            is_brain_stable = avg_loss
            
            # B. Els resultats són estables (Poca variació -> sempre obté resultat similar)
            is_performance_stable = std_reward < reward_std_thresh
            
            # C. El resultat és bo (Evitem convergir en un comportament de perdre sempre)
            # Assumim que rewards positius indiquen bon comportament o victòria
            is_winning = current_avg_reward > 0 

            if is_performance_stable and is_winning:
                avg_best = current_avg_reward
                conv_point = i

                
    
    # Si no troba convergència, retornem el total d'episodis
    if conv_point is None:
        return 0.0, -1, conv_list
            
    return avg_best, conv_point, conv_list


def Convergence_sim(self, agent, num_episodes, max_steps_per_episode=200, reward_func='simple', stochasticity=0.0):
        #This function is the core of the algorithm that trains the agent using Q-learning

        initial_state = self.chess.getCurrentState() 
        initial_state_key = self.state_to_key(initial_state)
        rewards_history = []
        Qtables_history = []

        current_delta_loses = []

        for episode in range(num_episodes):
            #This part is just to capture the Qtables.
         

            self.chess.reset_environment()
            current_state = self.chess.getCurrentState()
            self.listVisitedStates = []
            done = False
            total_reward = 0
            
            state_key = self.state_to_key(current_state)

            for step in range(max_steps_per_episode):
                legal_next_states = self.chess.get_all_next_states(current_state, True)
                
                if not legal_next_states:
                    reward = -500
                    done = True
                    break
                
                num_actions = len(legal_next_states)
                if episode != (num_episodes-1):
                    action_index = agent.policy(state_key, num_actions) 
                else:
                    action_index = agent.policy_last_iteration(state_key, num_actions) 

                final_next_state = legal_next_states[action_index]
                
                bk_state = self.getPieceState(final_next_state, 12)
                
                reward = 0
                is_checkmate = False
                is_draw = False

                if bk_state is None:
                    #We don't want the agent to kill the bk but to chekcmate it, so we penalize heavily
                    reward = -500
                    done = True
                else:
                    #In the case the king hasn't died, we check for checkmate and draw conditions
                    is_checkmate = self.isBlackInCheckMate(final_next_state)
                    is_draw = self.is_Draw(final_next_state)
                    #Depending on the reward function selected, we calculate the reward
                    if reward_func == 'heuristic':
                        reward = self.heuristica(final_next_state, step)

                    elif reward_func == 'simple':
                        reward = -1

                    if is_checkmate:
                        #In case of checkmate, we give a high reward, and there is the end of the episode
                        reward += 100
                        done = True                   

                    if is_draw:
                        reward = -50
                        done = True 
                   
                
                total_reward += reward
                
                next_state_key = self.state_to_key(final_next_state)
                
                #make the agent learn from the transition
                current_delta_loses.append(agent.learn(state_key, action_index, reward, next_state_key, done, num_actions))
                
                #We actually move the piece in the board
                self.chess.movePiece(current_state, final_next_state)
                
                current_state = final_next_state
                state_key = next_state_key

                if done:
                    break
            
            rewards_history.append(total_reward)
            Qtables_history.append(np.sum(current_delta_loses)/len(current_delta_loses) if current_delta_loses else 0.0)
            #search for the optimal path found so far
            if total_reward > self.best_reward or (total_reward == self.best_reward and step + 1 < self.min_steps_for_best_reward):
                
                self.best_reward = total_reward
                self.min_steps_for_best_reward = step + 1
                self.optimal_path_key = initial_state_key

            if episode % 200 == 0: 
                agent.reduce_exploration_rate_by_decrease_rate() 

            if episode % 1000 == 0: 
                agent.reduce_exploration_rate_by_30_percent() 
        
        return Qtables_history, rewards_history


# --- Main Configuration ---
"""
alphas = [0.1, 0.3, 0.5, 0.7]
gammas = [0.5, 0.7, 0.9]
epsilons = [0.1, 0.3, 0.5, 0.9]
"""

alphas = [0.3, 0.5, 0.7]
gammas = [0.5, 0.7]
epsilons = [0.3, 0.5]

def main():
    convergences = []
    best_conv = float("-inf")
    best_alpha = best_gamma = best_epsilon = None
    best_reward_for_best_conv = 0.0
    best_itr = 3000 # Assumim el màxim inicialment
    best_conv_list = []
    
    # Guardarem el millor agent per demostrar el guardat JSON al final
    best_agent_instance = None 

    print("Iniciant simulació de convergència...")
    
    total_iterations = len(alphas) * len(gammas) * len(epsilons)
    current_iter = 0

    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                current_iter += 1
                print(f"Simulació {current_iter}/{total_iterations}: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
                
                # Inicialitzem entorn i agent
                # Utilitzem Aichess ja que conté el Board i lògica auxiliar
                game = aichess.Aichess() 
                
                agent_instance = agent.Agent(
                    learning_rate=alpha,
                    future_weight=gamma,
                    exploration_rate=epsilon
                )
                
                NUM_EPISODES =3000
                #rewards = training_simulation(game, agent_instance, num_episodes=NUM_EPISODES, reward_func='heuristic')
                Qtables, rewards = Convergence_sim(game, agent_instance, num_episodes=NUM_EPISODES, reward_func='heuristic')
                # Comprovem convergència
                conv, conv_point, conv_list = check_convergence_final(
                                                    Qtables,            # Això és el loss_history
                                                    rewards, 
                                                    min_episodes=50, 
                                                    loss_thresh=0.5,    # Ajusta segons els valors que vegis de Loss (printa'ls si cal)
                                                    reward_std_thresh=20.0 # Ajusta segons la variància desitjada
                                                )
                
                # Si iterations és -1, significa que no ha trobat plateau, posem el màxim
                conv_iteration = conv_point if conv_point != -1 else NUM_EPISODES
                
                current_best_reward = max(rewards) if len(rewards) > 0 else 0.0

                convergences.append((alpha, gamma, epsilon, conv, current_best_reward, conv_iteration))
                
                

                # Criteri: Major convergència (reward mig final) i menor nombre d'iteracions per estabilitzar-se
                if conv >= best_conv:
                    # Si la convergència és millor, o és igual però amb menys iteracions (més ràpid)
                    if conv > best_conv or (conv == best_conv and conv_iteration < best_itr):
                        best_conv = conv
                        best_alpha, best_gamma, best_epsilon = alpha, gamma, epsilon
                        best_reward_for_best_conv = current_best_reward
                        best_itr = conv_iteration
                        best_agent_instance = agent_instance
                        best_conv_list = conv_list

    print("\n" + "="*50)
    print("Simulació completada.")
    print("Totes les convergències:")
    for alpha, gamma, epsilon, conv, rew, itr in convergences:
        print(f"alpha={alpha}, gamma={gamma}, epsilon={epsilon} ->max_reward={rew}, conv={conv:.3f}, itr={itr}")

    print("\nMillor convergència trobada:")
    print(
        f"alpha={best_alpha}, gamma={best_gamma}, epsilon={best_epsilon} "
        f"-> convergència={best_conv:.3f}, max_reward={best_reward_for_best_conv}, iteracions={best_itr}"
    )

    # Generem gràfic de la millor convergència
    print("\nGenerant gràfic de la millor convergència...")
    print(best_itr)
    print(len(best_conv_list))
    plot_convergence(best_conv_list, iterations=len(best_conv_list), conv_point=best_itr,filename='best_convergence_plot.png')
    
    # Demostració de guardar/carregar
    if best_agent_instance:
        print("\nGuardant la Q-table del millor agent...")
        agent_instance.save_qtable_to_json("best_agent_qtable.json")
        
        # Test de càrrega
        print("Verificant càrrega...")
        dummy_agent = agent.Agent()
        dummy_agent.load_qtable_from_json("best_agent_qtable.json")
        print(f"Mida de la Q-table carregada: {len(dummy_agent.q_table)} estats.")

if __name__ == "__main__":
    main()