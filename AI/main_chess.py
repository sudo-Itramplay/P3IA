import agent
from QChessEnviroment import Environment as env
from copy import deepcopy

# Store snapshots of the Q-table at specific intervals for analysis
global Qtables
Qtables = []
loops = [0, 50, 200, 500] 


def training_loop(env, agent_instance, num_episodes=501):
    print(f"########------BEGIN TRAINING ({env.reward_mode} reward)------########")
    
    # Store the initial Q-table (empty)
    Qtables.append(deepcopy(agent_instance.getQtable()))
    
    for episode in range(1, num_episodes + 1): 
        
        # Reset environment and get initial state string
        state_string = env.reset_environment() 
        agent_instance.reduce_exploration_rate_by_decrease_rate()
        
        done = False
        episode_reward = 0
        
        while not done:
            
            # 1. Get all possible next states (actions) from the current state
            possible_actions = env.get_possible_actions(state_string)
            
            if not possible_actions:
                 # Should only happen if in Checkmate, Stalemate, or Draw state
                 break

            # 2. Agent decides the best action (next_state_string)
            action_string = agent_instance.think(state_string, possible_actions)
            
            # 3. Take a step (execute the move in the environment)
            next_state_string, reward, done, next_full_state_list = env.step(state_string, action_string)
            
            # 4. Get possible actions from the next state (needed for Bellman equation's max(Q(s', a')))
            # This is crucial for Q-Learning update
            possible_actions_next_state = env.get_possible_actions(next_state_string)

            # 5. Agent learns from the transition
            agent_instance.learn(
                state_string, 
                action_string, 
                reward, 
                next_state_string, 
                done,
                possible_actions_next_state
            )
            
            state_string = next_state_string
            episode_reward += reward

        if episode in loops:
            Qtables.append(deepcopy(agent_instance.getQtable()))
        
        if episode % 100 == 0:
             print(f"Episode {episode}: Total Reward = {episode_reward:.2f}, Epsilon = {agent_instance.epsilon:.4f}, States Learned = {len(agent_instance.getQtable())}")
            
    print("Simulation completed.")


def run_optimal_path(env, agent_instance):
    """Prints the final path found by the converged agent."""
    print("\n--- OPTIMAL PATH RECONSTRUCTION ---")
    
    # Reset environment to initial state
    current_state_string = env.reset_environment()

    # Temporarily disable exploration for reconstruction
    original_epsilon = agent_instance.epsilon
    agent_instance.epsilon = 0.0 
    
    print("\nInitial Board:")
    env.aichess_instance.chess.boardSim.print_board()

    for step in range(1, 20): # Limit steps for K+R vs K to find mate (usually less than 16)
        
        # Get current full state list for check
        current_full_state_list = env.aichess_instance.getCurrentSimState()
        
        # Check for checkmate/draw before getting next move
        # We check the transition from current_state_list -> current_state_list (for terminal state checks)
        reward, done = env.calculate_reward(current_full_state_list, current_full_state_list)
        if done and reward == env.REWARD_CHECKMATE:
            print(f"--- Step {step-1}: CHECKMATE ACHIEVED (Reward: {reward}) ---")
            break
        elif done:
            print(f"--- Step {step-1}: Game Over (Reward: {reward}) ---")
            break
        
        # 1. Get all possible actions and choose the best (max_Q)
        possible_actions = env.get_possible_actions(current_state_string)
        
        if not possible_actions:
            print("--- STALEMATE/NO MOVES ---")
            break

        best_action_string = agent_instance.max_Q(current_state_string, possible_actions)
        
        if best_action_string is None:
            print("--- NO OPTIMAL MOVE FOUND IN Q-TABLE ---")
            break

        # 2. Take the step and update state (simulates move on the board)
        
        # We need to know which piece moved and where for the sequence description
        current_white_state_list = env.stringToState(env.aichess_instance, current_state_string)
        best_next_white_state_list = env.stringToState(env.aichess_instance, best_action_string)
        
        movement = env.aichess_instance.getMovement(current_white_state_list, best_next_white_state_list)
        from_pos = movement[0][0:2]
        to_pos = movement[1][0:2]
        
        print(f"-> Move {step}: From {from_pos} to {to_pos}")
        
        current_state_string, _, done, _ = env.step(current_state_string, best_action_string)
        
        print("Resulting Board:")
        env.aichess_instance.newBoardSim(env.aichess_instance.getCurrentSimState())
        env.aichess_instance.chess.boardSim.print_board()
        
    # Restore exploration rate
    agent_instance.epsilon = original_epsilon

# --- Main Execution ---

def main():
    
    # --- 2.a: Simple Reward (-1 everywhere, 100 at checkmate) ---
    print("\n" + "="*80)
    print("= Running Exercise 2.a: Simple Reward (-1 / 100) =")
    print("="*80 + "\n")

    env_simple = env(reward_mode='simple')
    agent_simple = agent.Agent(
        learning_rate=0.5, 
        future_weight=0.9, 
        exploration_rate=0.9, 
        decrease_rate=0.005 
    )
    
    training_loop(env_simple, agent_simple, num_episodes=501)
    
    # Answer 2.a.i: Sequence of actions and Q-tables
    print("\n### Q-Tables (2.a.i) ###")
    for i in range(len(Qtables)):
        print(f"\n--- Q-table Snapshot (Episode {loops[i]}) ---")
        q_table = Qtables[i]
        print(f"Total States Learned: {len(q_table)}")
        if len(q_table) > 0:
            sample_state = next(iter(q_table))
            print(f"Sample State {sample_state}: {q_table[sample_state]}")
    
    run_optimal_path(env_simple, agent_simple)

    # Reset Qtables for 2.b
 
    Qtables = []
    
    # --- 2.b: Heuristic Reward (Adapted from A* heuristic) ---
    print("\n" + "="*80)
    print("= Running Exercise 2.b: Heuristic Reward (H(s') - H(s)) =")
    print("="*80 + "\n")
    
    env_heuristic = env(reward_mode='heuristic')
    agent_heuristic = agent.Agent(
        learning_rate=0.5, 
        future_weight=0.9,
        exploration_rate=0.9, 
        decrease_rate=0.005 
    )
    
    training_loop(env_heuristic, agent_heuristic, num_episodes=501)

    # Answer 2.b.i: Sequence of actions and Q-tables
    print("\n### Q-Tables (2.b.i) ###")
    for i in range(len(Qtables)):
        print(f"\n--- Q-table Snapshot (Episode {loops[i]}) ---")
        q_table = Qtables[i]
        print(f"Total States Learned: {len(q_table)}")
        if len(q_table) > 0:
            sample_state = next(iter(q_table))
            print(f"Sample State {sample_state}: {q_table[sample_state]}")
    
    run_optimal_path(env_heuristic, agent_heuristic)


if __name__ == "__main__":
    main()