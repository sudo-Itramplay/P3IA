import agent 
import aichess 


def check_convergence(rewards, window=20, min_episodes=50):
    """
    Calcula la convergència.
    min_episodes: Nombre mínim d'episodis abans de comprovar si s'ha estancat.
    """
    if len(rewards) == 0:
        return 0.0, -1
    
    # 1. Càlcul de la mitjana final (igual que tenies)
    curr_window = min(window, len(rewards))
    last_rewards = rewards[-curr_window:]
    avg_last = sum(last_rewards) / curr_window
    
    iterations = -1
    
    # 2. Detectar estancament (plateau), però ignorant el principi
    # Comencem a 'min_episodes' o a 1, el que sigui més gran.
    start_index = 1 if min_episodes < 50 else min_episodes
    
    for i in range(start_index, len(rewards)-1):
        # Comprovem si hi ha 3 valors idèntics consecutius
        variation1 = rewards[i-1] - rewards[i] - rewards[i+1]
        variation2 = rewards[i] - rewards[i+1]

        if -20 <= variation1 <= 20 and -20 <= variation2 <= 20:
            # Només considerem estancament si el reward és raonable   
            if rewards[i] > 0: 
                iterations = i
            break
            
    # Si no troba convergència, retornem el total d'episodis com a "temps de convergència"
    if iterations == -1:
        iterations = len(rewards)
            
    return avg_last, iterations

def training_simulation(env_ai, agent_instance, num_episodes=200, max_steps=200, reward_func='heuristic'):
    """
    Executa un bucle d'entrenament similar a aichess.qLearningChess però retorna
    l'històric de recompenses per poder calcular la convergència.
    """
    rewards_history = []
    
    for episode in range(num_episodes):
        env_ai.chess.reset_environment()
        current_state = env_ai.chess.getCurrentState()
        state_key = env_ai.state_to_key(current_state)
        
        total_reward = 0
        done = False
        
        for step in range(max_steps):
            # Obtenir estats següents legals
            legal_next_states = env_ai.chess.get_all_next_states(current_state, True)
            
            if not legal_next_states:
                # Stalemate o sense moviments, penalització gran
                reward = -500
                done = True
                # Aprenem el final (sense next_state vàlid realment útil aquí, però mantenim estructura)
                agent_instance.learn(state_key, -1, reward, state_key, done, 0)
                total_reward += reward
                break

            num_actions = len(legal_next_states)
            
            # Decisió de l'acció
            if episode != (num_episodes - 1):
                action_index = agent_instance.policy(state_key, num_actions)
            else:
                action_index = agent_instance.policy_last_iteration(state_key, num_actions)
            
            final_next_state = legal_next_states[action_index]
            next_state_key = env_ai.state_to_key(final_next_state)
            
            # Comprovacions de lògica de joc (Rei Negre, escac i mat, taules)
            bk_state = env_ai.getPieceState(final_next_state, 12)
            
            reward = 0
            is_checkmate = False
            is_draw = False

            if bk_state is None:
                reward = -500
                done = True
            else:
                is_checkmate = env_ai.isBlackInCheckMate(final_next_state)
                is_draw = env_ai.is_Draw(final_next_state)
                
                if reward_func == 'heuristic':
                    reward = env_ai.heuristica(final_next_state, step)
                elif reward_func == 'simple':
                    reward = -1

                if is_checkmate:
                    reward += 100
                    done = True
                
                if is_draw:
                    reward = -50
                    done = True

            total_reward += reward
            
            # L'agent aprèn
            agent_instance.learn(state_key, action_index, reward, next_state_key, done, num_actions)
            
            # Movem peces (encara que per simulació pura només necessitem l'estat, el board ho requereix per consistència interna)
            env_ai.chess.movePiece(current_state, final_next_state)
            current_state = final_next_state
            state_key = next_state_key
            
            if done:
                break
        
        # Reducció de paràmetres d'exploració (segons lògica original d'aichess)
        if episode % 200 == 0:
            agent_instance.reduce_exploration_rate_by_decrease_rate()
        if episode % 1000 == 0:
            agent_instance.reduce_exploration_rate_by_30_percent()

        rewards_history.append(total_reward)
        if episode > num_episodes-5 :
            print(f"Episodi {episode+1}/{num_episodes}, Recompensa total: {total_reward}")
    return rewards_history

# --- Main Configuration ---

alphas = [0.1, 0.3, 0.5, 0.7]
gammas = [0.5, 0.7, 0.9]
epsilons = [0.1, 0.3, 0.5, 0.9]

def main():
    convergences = []
    best_conv = float("-inf")
    best_alpha = best_gamma = best_epsilon = None
    best_reward_for_best_conv = 0.0
    best_itr = 200 # Assumim el màxim inicialment
    
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
                
                NUM_EPISODES = 3000
                rewards = training_simulation(game, agent_instance, num_episodes=NUM_EPISODES, reward_func='heuristic')
                
                # Comprovem convergència
                conv, iterations = check_convergence(rewards)
                
                # Si iterations és -1, significa que no ha trobat plateau, posem el màxim
                iterations_val = iterations if iterations != -1 else NUM_EPISODES
                
                convergences.append((alpha, gamma, epsilon, conv, iterations_val))
                
                current_best_reward = max(rewards) if len(rewards) > 0 else 0.0

                # Criteri: Major convergència (reward mig final) i menor nombre d'iteracions per estabilitzar-se
                if conv >= best_conv:
                    # Si la convergència és millor, o és igual però amb menys iteracions (més ràpid)
                    if conv > best_conv or (conv == best_conv and iterations_val < best_itr):
                        best_conv = conv
                        best_alpha, best_gamma, best_epsilon = alpha, gamma, epsilon
                        best_reward_for_best_conv = current_best_reward
                        best_itr = iterations_val
                        best_agent_instance = agent_instance

    print("\n" + "="*50)
    print("Simulació completada.")
    print("Totes les convergències:")
    for alpha, gamma, epsilon, conv, itr in convergences:
        print(f"alpha={alpha}, gamma={gamma}, epsilon={epsilon} -> conv={conv:.3f}, itr={itr}")

    print("\nMillor convergència trobada:")
    print(
        f"alpha={best_alpha}, gamma={best_gamma}, epsilon={best_epsilon} "
        f"-> convergència={best_conv:.3f}, max_reward={best_reward_for_best_conv}, iteracions={best_itr}"
    )
    
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