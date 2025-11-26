import agent
import environment

def training_loop(env, agent_instance):
    print("########------BEGIN------########")
    env.print_board()
    last_path=[]
    paths=[]
    final_reward = 0
    rewards=[]
    # Run the simulation
    for episode in range(101):  # Number of episodes
        print("########------Episode ", episode, "------########")
        env.init2()
        env.print_board()
        agent_instance.reduce_exploration_rate_by_decrease_rate()
        state = env.get_state()
        done = False

        while not done:
            action = agent_instance.think(state)
            next_state, reward, done = env.move_piece(action)
            env.print_board()
            agent_instance.learn(state, action, reward, next_state, done)
            state = next_state

    return paths, rewards



def main():

    env = environment.Environment()
    env.init2()
    
    # Initialize the agent
    agent_instance = agent.Agent()
    paths, rewards = training_loop(env, agent_instance)
            
    print("Simulation completed.")



if __name__ == "__main__":
    main()

    
