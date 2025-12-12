import agent
import environment

def training_loop(env, agent_instance, num_episodes=300):
    print("########------BEGIN------########")
    last_path = []
    paths = []
    rewards = []

    for episode in range(num_episodes):
        print("########------Episode ", episode, "------########")
        env.reset_environment(mode='init2') #Initalizng the enviroment with the penalties
        agent_instance.reduce_exploration_rate_by_decrease_rate()
        state = env.get_state()
        done = False

        episode_reward = 0.0 

        if episode == num_episodes - 1:
            last_path.append(state)

        while not done:
            action = agent_instance.think(state)
            next_state, reward, done = env.move_piece(action)
            agent_instance.learn(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward

            if episode % 20 == 0:
                last_path.append(state)

        paths.append(last_path)
        rewards.append(episode_reward)

    return paths, rewards


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
    start_index = max(1, min_episodes)
    
    for i in range(start_index, len(rewards)-1):
        # Comprovem si hi ha 3 valors idèntics consecutius
        if rewards[i-1] == rewards[i] == rewards[i+1]:
            # Només considerem estancament si el reward és raonable   
            if rewards[i] > -200: 
                iterations = i
            break
            
    # Si no troba convergència, retornem el total d'episodis com a "temps de convergència"
    if iterations == -1:
        iterations = len(rewards)
            
    return avg_last, iterations

alphas = [0.1, 0.3, 0.5, 0.7]

gammas = [0.5, 0.7, 0.9]

epsilons = [0.1, 0.3, 0.5, 0.9]


def main():
    convergences = []
    best_conv = float("-inf")
    best_alpha = best_gamma = best_epsilon = None
    best_reward_for_best_conv = 0.0
    iterations = 0
    best_itr = 200

    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                env = environment.Environment()
                agent_instance = agent.Agent(
                    learning_rate=alpha,
                    future_weight=gamma,
                    exploration_rate=epsilon
                )
                iterations=200

                paths, rewards = training_loop(env, agent_instance)

                conv, iterations = check_convergence(rewards)
                convergences.append((alpha, gamma, epsilon, conv, iterations))

                triplet_best_reward = max(rewards) if len(rewards) > 0 else 0.0

                if conv >= best_conv and iterations < best_itr:
                    best_conv = conv
                    best_alpha, best_gamma, best_epsilon = alpha, gamma, epsilon
                    best_reward_for_best_conv = triplet_best_reward
                    best_itr = iterations

    print("Simulation completed.")
    print("All convergences:")
    for alpha, gamma, epsilon, conv, itr in convergences:
        print(f"alpha={alpha}, gamma={gamma}, epsilon={epsilon} -> convergence={conv:.3f}, iterations={itr}")

    print("\nBest convergence:")
    print(
        f"alpha={best_alpha}, gamma={best_gamma}, epsilon={best_epsilon} "
        f"-> convergence={best_conv:.3f}, reward={best_reward_for_best_conv}, iterations={best_itr}"
    )


if __name__ == "__main__":
    main()

    
