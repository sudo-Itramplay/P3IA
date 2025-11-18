import agent.py as agent
import environment.py as env

def main():
    # Initialize the environment
    environment = env.Environment()
    
    # Initialize the agent
    agent_instance = agent.Agent(environment)
    
    # Run the simulation
    for episode in range(100):  # Number of episodes
        state = environment.reset()
        done = False
        
        while not done:
            action = agent_instance.select_action(state)
            next_state, reward, done = environment.step(action)
            agent_instance.learn(state, action, reward, next_state, done)
            state = next_state
            
    print("Simulation completed.")
    
