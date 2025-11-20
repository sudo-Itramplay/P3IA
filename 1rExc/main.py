import agent
import enviroment

def main():
    # Initialize the environment
    environment = enviroment.Enviroment()
    
    # Initialize the agent
    agent_instance = agent.Agent()
    
    print("########------BEGIN------########")
    environment.print_board()
    # Run the simulation
    for episode in range(100):  # Number of episodes
        environment.reset_enviroment()
        state = environment.get_state()
        done = False
        
        while not done:
            action = agent_instance.think(state)
            next_state, reward, done = environment.move_piece(action)
            environment.print_board()
            agent_instance.learn(state, action, reward, next_state, done)
            state = next_state

            
    print("Simulation completed.")

if __name__ == "__main__":
    main()
    
