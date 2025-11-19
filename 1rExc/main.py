import agent
import enviroment

def main():
    # Initialize the environment
    environment = enviroment.Enviroment()
    
    # Initialize the agent
    agent_instance = agent.Agent()
    
    # Run the simulation
    for episode in range(100):  # Number of episodes
        state = environment.get_enviroment()
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
    
