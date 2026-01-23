import agent 
import aichess 
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(history_to_plot, iterations,conv_point, filename='convergence_plot.png'):
    """
    Creats the plot for convergence based on the difference in Q'table values (loss history).
    """
    plt.figure(figsize=(10, 6))
    
    #evolution of the diffrence in Q-table values
    plt.plot(history_to_plot, color='red', alpha=0.6, label='difference in Q-table values')

    plt.axvline(x=conv_point, color='blue', linestyle='--', linewidth=2, label=f'Point of convergence (x={conv_point})')
    plt.title("Agent convergence through episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(filename)
    print(f"saved as {filename}")
    plt.show()

def check_convergence_final(loss_history, rewards, min_episodes=50, loss_thresh=5.0, reward_std_thresh=80.0):
    """
    Calculates convergence based solely on the Q-table values (Loss).
    min_episodes: Minimum number of episodes before checking for stagnation.
    """
    if len(loss_history) == 0:
        return 0.0, -1, []
    
    iterations = -1
    #conv_list stores the loss history for plotting
    conv_list=[] 
    start_index = 1 if min_episodes < 50 else min_episodes
    avg_best = 0.0
    conv_point=None
    
    #we iterate respecting the number of q-tables
    for i in range(len(loss_history)): 
        
        if i < min_episodes:
            conv_list.append(loss_history[i])
            continue
        
        #calculate the average loss over the recent 50 episodes
        recent_loss = loss_history[(i-50):i]
        avg_loss = np.mean(recent_loss)
        
        # A. The brain is stable (Low Loss)
        is_brain_stable = avg_loss < loss_thresh
        
        #average loss
        conv_list.append(avg_loss)

        if conv_point is None: #we keep l;ooking until found the ocnvergeence pioint
            if is_brain_stable:
                start_window = max(0, i - min_episodes)
                avg_best = np.mean(rewards[start_window:i]) 
                conv_point = i
        
            
    #Case we have not found any convergence point, we return 0.0 and -1
    if conv_point is None:
        return 0.0, -1, conv_list
            
    return avg_best, conv_point, conv_list

def Convergence_sim(self, agent, num_episodes, max_steps_per_episode=200, reward_func='simple', stochasticity=0.0):
        """
        Simulatate the agent q-learning to check convergence.
        It's like Q-learning with minor changes to be able to store results and find the point whre the Q-table stabilizes.
        """
        initial_state = self.chess.getCurrentState() 
        initial_state_key = self.state_to_key(initial_state)
        rewards_history = []
        loss_history = []

        for episode in range(num_episodes):
            
            self.chess.reset_environment()
            current_state = self.chess.getCurrentState()
            self.listVisitedStates = []
            done = False
            total_reward = 0
            
            state_key = self.state_to_key(current_state)

            current_episode_losses = []

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
                current_episode_losses.append(agent.learn(state_key, action_index, reward, next_state_key, done, num_actions))
                
                #We actually move the piece in the board
                self.chess.movePiece(current_state, final_next_state)
                
                current_state = final_next_state
                state_key = next_state_key

                if done:
                    break
            
            rewards_history.append(total_reward)
            loss_history.append(np.sum(current_episode_losses)/len(current_episode_losses) if current_episode_losses else 0.0)
            
            #search for the optimal path found so far
            if total_reward > self.best_reward or (total_reward == self.best_reward and step + 1 < self.min_steps_for_best_reward):
                
                self.best_reward = total_reward
                self.min_steps_for_best_reward = step + 1
                self.optimal_path_key = initial_state_key

            if episode % 200 == 0: 
                agent.reduce_exploration_rate_by_decrease_rate() 

            if episode % 1000 == 0: 
                agent.reduce_exploration_rate_by_30_percent() 
        
        return loss_history, rewards_history


alphas = [0.3, 0.5, 0.7]
gammas = [0.5, 0.7, 0.9]
epsilons = [0.3, 0.5, 0.9]

def main():
    convergences = []
    best_conv = float("-inf")
    best_alpha = best_gamma = best_epsilon = None
    best_reward_for_best_conv = 0.0
    best_itr = 3000
    best_conv_list = []
    
    best_agent_instance = None 

    print("Convergence simulation...")
    
    total_iterations = len(alphas) * len(gammas) * len(epsilons)
    current_iter = 0

    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                current_iter += 1
                #Keep trak during all the simulations
                print(f"simulation {current_iter}/{total_iterations}: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
                
                #Initialize environment and agent
                #Using Aichess as it contains the Board and auxiliary logic
                game = aichess.Aichess() 
                agent_instance = agent.Agent(learning_rate=alpha, future_weight=gamma,exploration_rate=epsilon)
                
                num_episodes = 1000

                #Here is where the convergence is calculated it self, if any params needed for change HERE
                loss_history, rewards = Convergence_sim(game, agent_instance, num_episodes=num_episodes, reward_func='heruistic')
                conv, conv_point, conv_list = check_convergence_final(loss_history, rewards)
                
                #If there is a -1, the simulation hasn't reached the stability in Q differences, so we set the maximum
                conv_iteration = conv_point if conv_point != -1 else num_episodes
                
                current_best_reward = max(rewards) if len(rewards) > 0 else 0.0
                convergences.append((alpha, gamma, epsilon, conv, current_best_reward, conv_iteration))
                
                #we a deciding based on two factors, quality and speed, conv is quality, conv_iteration is speed
                if conv >= best_conv:
                                       
                    #if quality is strictly better (conv > best_conv),
                    #if quality is equal but speed is faster (conv_iteration < best_itr).
                    if conv > best_conv or (conv == best_conv and conv_iteration < best_itr):
                        best_conv = conv                            
                        best_alpha, best_gamma, best_epsilon = alpha, gamma, epsilon #best parameters
                        best_reward_for_best_conv = current_best_reward #maximum reward achieved
                        best_itr = conv_iteration #best convergence speed, number of episodes
                        best_conv_list = conv_list 

    print("\n" + "="*50)
    print("convergences")
    for alpha, gamma, epsilon, conv, rew, itr in convergences:
        print(f"alpha={alpha}, gamma={gamma}, epsilon={epsilon} ->max_reward={rew}, conv={conv:.3f}, itr={itr}")

    print("\nbest convergence:")
    print(
        f"alpha={best_alpha}, gamma={best_gamma}, epsilon={best_epsilon} "
        f"-> convergence={best_conv:.3f}, max_reward={best_reward_for_best_conv}, iteracions={best_itr}"
    )

    print("\nPlotting best convergence...")
    print(best_itr)
    print(len(best_conv_list))
    plot_convergence(best_conv_list, iterations=len(best_conv_list), conv_point=best_itr,filename='best_convergence_plot.png')

    if best_agent_instance:
        print("\nsaving Q-table of the best agent")
        agent_instance.save_qtable_to_json("best_agent_qtable.json")

        dummy_agent = agent.Agent()
        dummy_agent.load_qtable_from_json("best_agent_qtable.json")
        print(f"Size of loaded Q-table: {len(dummy_agent.q_table)} states.")

if __name__ == "__main__":
    main()