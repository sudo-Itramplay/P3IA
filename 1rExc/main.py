from agent import Agent as agent
from enviroment import Enviroment as env

def main():
    # Initialize the environment
    environment = env.Enviroment()
    
    # Initialize the agent
    agent_instance = agent.Agent()
    
    # Run the simulation
    for episode in range(100):  # Number of episodes
        state = environment.get_enviroment()
        done = False
        
        while not done:
            action = agent_instance.think(state)
            next_state, reward, done = environment.move_piece(action)
            agent_instance.learn(state, action, reward, next_state, done)
            state = next_state
            
    print("Simulation completed.")
    
