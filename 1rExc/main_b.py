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
        env.reset_environment(mode='init2')
        agent_instance.reduce_exploration_rate_by_decrease_rate()
        state = env.get_state()
        done = False

        while not done:
            action = agent_instance.think(state)
            next_state, reward, done = env.move_piece(action)
            #env.print_board()
            agent_instance.learn(state, action, reward, next_state, done)
            state = next_state


        if episode%25==0 or episode == 100:
            if episode != 50:
                Qtables.append(agent_instance.getQtable())

    return paths, rewards


Qtables=[]
loops=[0, 25, 75, 100]

def main():

    env = environment.Environment()
    
    # Initialize the agent
    agent_instance = agent.Agent()
    paths, rewards = training_loop(env, agent_instance)
            
    print("Simulation completed.")
    print(len(Qtables))
    for i in range(4):
        print()
        print("---------------------------------------------------------")
        print("Qtable for loop ", loops[i])
        print("---------------------------------------------------------")
        print()
        print(Qtables[i])


if __name__ == "__main__":
    main()

    
