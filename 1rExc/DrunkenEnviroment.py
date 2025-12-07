import numpy as np
import AI_BASE_P2.piece as piece 
import random 

class DrunkenEnvironment:
    initState = None
    
    rows = 0
    cols = 0
        
    board = []
    #coord Agent (White king)
    currentStateW = ()
    #coord final reward (Goal)
    currentStateB = ()
    #coord obstacle
    currentObs = ()
    #general penalty per movement
    reward = -1
    #Final reward
    treasure = 100
    #Wall penalization
    wall_penalization = -100
    
    def __init__(self, rows=3, cols=4, currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1)):

        if self.initState is None:
            self.rows = rows
            self.cols = cols

            self.board = np.full((rows, cols), -1, dtype=object)
            
            self.currentStateW = currentStateW
            self.currentStateB = currentStateB
            self.currentObs = currentObs
            #Inicialzation fo the special postiions n the enviroemnt
            self.board[self.currentStateB] = self.treasure
            self.board[self.currentStateW] = piece.King(True) 
            self.board[self.currentObs] = self.wall_penalization
            self.initState = 1

    def init2(self, rows=3, cols=4,currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1)): 
        #Different incializaiton for the 1.b exercise, with special and different penalties
        if self.initState is None:
            self.rows = rows
            self.cols = cols

            self.board = np.empty((rows, cols), dtype=object)

            self.currentStateW = currentStateW
            self.currentStateB = currentStateB
            self.currentObs   = currentObs

            #The distributioon has a logic
            goal_r, goal_c = self.currentStateB
            for r in range(rows):
                for c in range(cols):
                    dist = abs(r - goal_r) + abs(c - goal_c)
                    if dist == 0:
                        self.board[r, c] = self.treasure #goal
                    else:
                        self.board[r, c] = -dist#distribution of penalties

            self.board[self.currentObs] = self.wall_penalization
            self.board[self.currentStateW] = piece.King(True)
            self.initState = 1


    def get_environment(self):
        return self.board 

    def get_state(self):
        return self.currentStateW

    def print_board(self):
        print("-" * (self.cols * 4 + 1))
        for r in range(self.rows):
            row_display = "| "
            for c in range(self.cols):
                cell_value = self.board[r, c]
                
                if (r, c) == self.currentStateW:
                    symbol = "K " 
                elif (r, c) == self.currentStateB:
                    symbol = "100" 
                elif cell_value == -1:
                    symbol = "- "
                else:
                    symbol = "? " 
                
                row_display += f"{symbol} | " if symbol != "100" else f"{symbol}| "
            print(row_display)
            print("-" * (self.cols * 4 + 1))
        

    def is_finish(self):
        if self.currentStateB == self.currentStateW[:2]:
            return True
        return False
    
    def move_piece(self, action):
        """
        Moves the piece to the acton recieved form the agent
        +
        a drunken sailor proability of deviating from intended action, 1%|99%
        """
        current_r, current_c = self.currentStateW
        
        #Drunk sailor stochasticity logic
        intended_action = action
        possible_actions = [0, 1, 2, 3]
        
        if random.random() < 0.01: #1% chance of being drunk
            #look for the ohter possible actions, excluding the intended one
            other_actions = [a for a in possible_actions if a != intended_action]
            #Sailor takes a random action from the unintended directions
            if other_actions:
                actual_action = random.choice(other_actions)
            else:
                actual_action = intended_action
        else: #99% chance of success
            actual_action = intended_action
        #End of the drunken sailor logic

        deltas = {
            0: (-1, 0),  #Up
            1: (1, 0),   #Down
            2: (0, 1),   #Right
            3: (0, -1)   #Left
        }

        #intented action and the actual adction taken
        idr, idc = deltas.get(intended_action, (0, 0))
        adr, adc = deltas.get(actual_action, (0, 0))
        #idr is intened 
        #adc is actual
        intended_pos = (current_r + idr, current_c + idc)
        new_r = current_r + adr
        new_c = current_c + adc
        new_pos = (new_r, new_c)

        #DEBUGGING Report if drunken deviation occurred
        #In case the executed action has been different from the intended one
        if actual_action != intended_action:
            action_names = {0: 'Up', 1: 'Down', 2: 'Right', 3: 'Left'}
            try:
                intended_name = action_names[intended_action]
                actual_name = action_names[actual_action]
            except Exception:
                intended_name = str(intended_action)
                actual_name = str(actual_action)
            print(f"THE AGENT IS SO DRUNK: he wanted to go {intended_name} from {self.currentStateW} to {intended_pos}, "
                  f"but fell to the {actual_name} to {new_pos}.")
               
        reward = self.reward
        done = False        

        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            return self.currentStateW, self.wall_penalization, done

        #has the agent reached the goal?
        if new_pos == self.currentStateB:
            #as it could be 100 or -dist
            reward = self.treasure
            done = True
        #has the agent hit the obstacle?
        if new_pos == self.currentObs:
            return self.currentStateW, self.wall_penalization, done
        
        # IMPORTANT: If the new position is NOT the goal/obstacle/wall, 
        # the reward is the value stored in the board (e.g., -1 or -3, if init2 was used)
        if not done:
            # We fetch the specific penalty for the new position
            reward = self.board[new_pos] 
            # Note: This logic assumes that the reward for stepping on a clean tile is defined by the board value. 
            # This correctly handles both the simple -1/100 (init) and the Manhattan distance (init2).

        # Recuperem l'objecte Rei
        king_obj = self.board[self.currentStateW]
        self.board[self.currentStateW] = reward
        self.board[self.currentStateW] = -1 
        self.currentStateW = new_pos
        self.board[new_pos] = king_obj
        
        return self.currentStateW, reward, done
    
    def reset_environment(self, rows=3, cols=4, currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1), mode='default'):
        #This is fine, it just re-initializes the board
        if mode == 'init2':
            prev = self.initState
            self.initState = None
            self.init2(rows=rows, cols=cols, currentStateW=currentStateW, currentStateB=currentStateB, currentObs=currentObs)
            self.initState = 1 if prev is None else prev
            return

        self.rows = rows
        self.cols = cols
        self.board = np.full((rows, cols), -1, dtype=object)
        self.currentStateW = currentStateW
        self.currentStateB = currentStateB
        self.currentObs = currentObs
        self.board[self.currentStateB] = self.treasure
        self.board[self.currentStateW] = piece.King(True)
        self.board[self.currentObs] = self.wall_penalization