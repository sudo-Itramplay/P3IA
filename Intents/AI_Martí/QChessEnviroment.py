import numpy as np
import agent
import aichess as aichess
import auxiliary_P3 as p3
from copy import deepcopy

#I had to create this to prevent a depndencies loop with aichess, that created an inifnite loop problem
_initializing_aichess = False

class Environment:
    """
    Environment wrapper for Aichess to support Q-learning for the K+R vs K scenario.
    The goal is for White (agent) to achieve checkmate on Black (non-agent, static).
    """
    
    #Initial board setup for chess environment
    INITIAL_BOARD_MATRIX = np.zeros((8, 8))
    INITIAL_BOARD_MATRIX[7][0] = 2 #White rook   
    INITIAL_BOARD_MATRIX[7][5] = 6 #White king   
    INITIAL_BOARD_MATRIX[0][5] = 12 #nigga king
    
    #We use the all -1 reward system for simplicity
    REWARD_CHECKMATE = 100
    REWARD_DEFAULT = -1
    REWARD_DRAW = -50 

    def __init__(self, reward_mode='simple'):
        #preventing dependencies loop
        global _initializing_aichess
        if _initializing_aichess:
            self.aichess_instance = None
            self.stateToString = p3.stateToString 
            self.stringToState = p3.stringToState
            self.getPieceState = None
        else:
            _initializing_aichess = True
            try:
                # Initialize the Aichess simulator with the P1 board state
                self.aichess_instance = aichess.Aichess(self.INITIAL_BOARD_MATRIX, myinit=True)
                # Utility functions for state encoding/decoding
                self.stateToString = p3.stateToString 
                self.stringToState = p3.stringToState
                self.getPieceState = self.aichess_instance.getPieceState
            finally:
                _initializing_aichess = False
        
        self.reward_mode = reward_mode
        
    def reset_environment(self):
        self.aichess_instance = aichess.Aichess(self.INITIAL_BOARD_MATRIX, myinit=True)
        current_state_list = self.aichess_instance.getCurrentSimState()
        return self.state_to_string(current_state_list)

    def state_to_string(self, state_list):
        white_pieces = self.aichess_instance.getWhiteState(state_list)
        return self.stateToString(self.aichess_instance, white_pieces)

    def string_to_state(self, state_string):
        return self.stringToState(self.aichess_instance, state_string)

    def get_possible_actions(self, state_string):
        """
        Returns a list of valid next state strings (actions) for the given state.
        """
        current_white_state = self.stringToState(self.aichess_instance, state_string)
        
        # 2. Get the current black state (needed for the board context)
        # Reconstruct the full board state with the decoded white state and current black state
        # This ensures the board simulator has the correct positions before we extract the black state
        current_full_state = self.aichess_instance.getCurrentSimState()
        current_black_state = self.aichess_instance.getBlackState(current_full_state)
        # Reconstruct the board with the correct white and black states, filtering None values
        reconstructed_state = current_white_state + current_black_state
        reconstructed_state_clean = [p for p in reconstructed_state if p is not None]
        self.aichess_instance.newBoardSim(reconstructed_state_clean)
        #Get the possible states
        next_states_list_of_lists = self.aichess_instance.getListNextStatesW(current_white_state) 
        
        next_state_strings = []
        
        for candidate_white_move_list in next_states_list_of_lists:
            # a. Check for legality (White King not in check AFTER the move)
            # Filter out None values before passing to is_legal_transition
            current_white_state_clean = [p for p in current_white_state if p is not None]
            current_black_state_clean = [p for p in current_black_state if p is not None]
            is_legal, next_full_node = self.aichess_instance.is_legal_transition(
                current_white_state_clean, 
                current_black_state_clean, 
                candidate_white_move_list, 
                color=True
            )

            if is_legal:
                #Convert the legal full state list to a string for the Q-table
                next_state_string = self.state_to_string(next_full_node)
                next_state_strings.append(next_state_string)
                
        return next_state_strings


    def calculate_reward(self, prev_state_list, next_state_list):
        """Calculates the reward based on the transition to the new state."""
        
        # Pass the full state list to aichess methods (should contain all pieces)
        # 1. Check for checkmate (Goal state)
        if self.aichess_instance.isBlackInCheckMate(next_state_list):
            return self.REWARD_CHECKMATE, True 
        
        # 2. Check for draw 
        if self.aichess_instance.is_Draw(next_state_list):
             return self.REWARD_DRAW, True 
        
        # 3. Apply Simple Reward Function (2.a)
        if self.reward_mode == 'simple':
            return self.REWARD_DEFAULT, False 
        
        # 4. Apply Heuristic Reward Function (2.b) - Delta Heuristic R(s, a) = H(s') - H(s)
        elif self.reward_mode == 'heuristic':
            h_next = self.aichess_instance.heuristica(next_state_list, True) # White's perspective
            h_prev = self.aichess_instance.heuristica(prev_state_list, True)
            
            # Reward is the change in heuristic value
            reward = h_next - h_prev 
            return reward, False


    def step(self, state_string, action_string):
        """
        Takes one step/move given an action (next state string).
        
        Returns: next_state_string, reward, done, next_full_state_list
        """
        
        # 1. Reconstruct the full board state from the current state_string
        current_white_state = self.stringToState(self.aichess_instance, state_string)
        # Get the board's current black state
        current_board_state = self.aichess_instance.getCurrentSimState()
        current_black_state = self.aichess_instance.getBlackState(current_board_state)
        # Rebuild the full state - keep original with potential None values
        current_full_state = current_white_state + current_black_state
        # Filter only for newBoardSim which can't handle None
        current_full_state_clean = [p for p in current_full_state if p is not None]
        self.aichess_instance.newBoardSim(current_full_state_clean)
        
        # Get the proper previous state from the board (before the move)
        prev_full_state_list = self.aichess_instance.getCurrentSimState()
        
        # White's state derived from the *action_string*
        next_white_state_list = self.stringToState(self.aichess_instance, action_string)
        current_white_state_list = self.aichess_instance.getWhiteState(current_full_state)

        # 2. Find the actual move that corresponds to this state transition
        # Filter out None values before passing to getMovement
        current_white_state_clean = [p for p in current_white_state_list if p is not None]
        next_white_state_clean = [p for p in next_white_state_list if p is not None]
        movement = self.aichess_instance.getMovement(
            current_white_state_clean, 
            next_white_state_clean
        )
        
        # Extract (r, c) for start and end positions
        start_pos = movement[0][0:2]
        to_pos = movement[1][0:2]
        
        # 3. Execute the move on the Aichess simulator
        # Reset board to current state, then apply moveSim
        self.aichess_instance.newBoardSim(current_full_state_clean)
        self.aichess_instance.chess.moveSim(start_pos, to_pos, verbose=False) 
        
        # 4. Get the resulting state in Aichess list format (must include all pieces)
        next_full_state_list = self.aichess_instance.getCurrentSimState()
        
        # 5. Calculate reward and check for done condition
        # Use the proper states from getCurrentSimState
        reward, done = self.calculate_reward(prev_full_state_list, next_full_state_list)
        
        # 6. Convert the resulting state back to the string key for the Q-Table
        next_state_string = self.state_to_string(next_full_state_list)

        return next_state_string, reward, done, next_full_state_list