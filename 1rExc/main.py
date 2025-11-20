import agent
import environment

def main():
    # Initialize the environment
    env = environment.Environment()
    
    # Initialize the agent
    agent_instance = agent.Agent()
    
    print("########------BEGIN------########")
    env.print_board()
    last_path=[]
    final_reward = 0
    # Run the simulation
    for episode in range(100):  # Number of episodes
        print("########------Episode ", episode, "------########")
        env.reset_environment()
        agent_instance.reduce_exploration_rate_by_decrease_rate()
        state = env.get_state()
        done = False
        
        if episode == 99:
            last_path.append(state)

        while not done:
            action = agent_instance.think(state)
            next_state, reward, done = env.move_piece(action)
            env.print_board()
            agent_instance.learn(state, action, reward, next_state, done)
            state = next_state

            if episode == 99:
                last_path.append(state)
                final_reward+=reward

            
    print("Simulation completed.")
    print("Final Path")
    print(last_path)
    print("Final Reward:")
    print(final_reward)

if __name__ == "__main__":
    main()
    
