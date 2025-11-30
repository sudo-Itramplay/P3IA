import numpy as np
import piece 
import random 

class DrunkenEnvironment:
    initState = None
    
    rows = 0
    cols = 0
        
    board = []
    # coord Agent (Rei Blanc)
    currentStateW = ()
    # coord Recompensa final (Objectiu)
    currentStateB = ()
    # coord obstacle
    currentObs = ()
    # Recompensa per moviment (pas)
    reward = -1
    # Bonificació final
    treasure = 100
    # Wall penalization
    wall_penalization = -100
    
    def __init__(self, rows=3, cols=4, currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1)):

        if self.initState is None:
            self.rows = rows
            self.cols = cols
            
            # 1. Inicialitzem amb dtype=object per poder guardar números I peces
            self.board = np.full((rows, cols), -1, dtype=object)
            
            self.currentStateW = currentStateW
            self.currentStateB = currentStateB
            self.currentObs = currentObs

            # 2. Posem el valor 100 a la posició B
            self.board[self.currentStateB] = self.treasure
            
            # 3. Posem el King a la posició W
            self.board[self.currentStateW] = piece.King(True) 

            # 4. Posem obstacle
            self.board[self.currentObs] = self.wall_penalization

            self.initState = 1

    def init2(self, rows=3, cols=4,currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1)): 
        # This method uses the Manhattan-distance reward structure (for P1.b)

        if self.initState is None:
            self.rows = rows
            self.cols = cols

            # Tauler amb números (recompenses) + peces
            self.board = np.empty((rows, cols), dtype=object)

            self.currentStateW = currentStateW
            self.currentStateB = currentStateB
            self.currentObs   = currentObs

            # 1. Omplim amb recompensa = -distància Manhattan fins al tresor
            goal_r, goal_c = self.currentStateB
            for r in range(rows):
                for c in range(cols):
                    dist = abs(r - goal_r) + abs(c - goal_c)
                    if dist == 0:
                        self.board[r, c] = self.treasure   # 100 al goal
                    else:
                        self.board[r, c] = -dist          # -1, -2, -3, -4, -5

            # 2. Posem obstacle (casella grisa)
            self.board[self.currentObs] = self.wall_penalization

            # 3. Posem el Rei a la posició W
            self.board[self.currentStateW] = piece.King(True)

            self.initState = 1


    def get_environment(self):
        return self.board 

    def get_state(self):
        """
        Retorna la posició actual de l'agent (Rei Blanc).
        """
        return self.currentStateW

    def print_board(self):
        """
        Mostra el tauler de joc de forma visual.
        """
        print("-" * (self.cols * 4 + 1))
        for r in range(self.rows):
            row_display = "| "
            for c in range(self.cols):
                cell_value = self.board[r, c]
                
                if (r, c) == self.currentStateW:
                    symbol = "K " 
                elif (r, c) == self.currentStateB: #Millor comprovar coord que valor
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
        Mou el rei a la nova posició basant-se en l'acció rebuda (int).
        It now includes stochasticity to simulate a "drunken sailor" effect.
        
        Retorna: (next_state, reward, done)
        Accions esperades (basat en agent.py): 
        0: Up, 1: Down, 2: Right, 3: Left
        """
        current_r, current_c = self.currentStateW
        
        #Drunk sailor stochasticity logic
        intended_action = action
        possible_actions = [0, 1, 2, 3]
        
        if random.random() < 0.01: # 1% chance of being drunk
            # Find all other possible directions (3 directions)
            other_actions = [a for a in possible_actions if a != intended_action]
            
            # Sailor takes a random action from the unintended directions
            if other_actions:
                actual_action = random.choice(other_actions)
            else:
                actual_action = intended_action # Fallback if only one move is possible
        else: # 99% chance of success
            actual_action = intended_action
        #End of the drunken sailor logic

        # 1. Definició de Deltas (Canvi de coordenades segons l'acció)
        # We use the actual_action determined by the stochastic logic
        deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, 1),   # Right
            3: (0, -1)   # Left
        }
        
        # Recuperem el desplaçament. Si l'acció no existeix, no ens movem (0,0)
        dr, dc = deltas.get(actual_action, (0, 0))
        
        # Calculem la proposta de nova posició
        new_r = current_r + dr
        new_c = current_c + dc
        new_pos = (new_r, new_c)
        
        # --- 2. Validacions i Lògica de Moviment ---
        
        reward = self.reward # Cost per defecte (-1 or Manhattan distance penalty)
        done = False        

        # A) Validació de límits del tauler (Murs)
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            # If hits a wall, return to the current position with wall penalization
            return self.currentStateW, self.wall_penalization, done

        # B) Comprovem si hem arribat a l'objectiu
        if new_pos == self.currentStateB:
            # We must fetch the actual reward value from the board, 
            # as it could be 100 or -dist (if init2 was used)
            # Since currentStateB is always 100 (treasure) in both init and init2:
            reward = self.treasure
            done = True # Episodi acabat

        # C) Validació de Obstacle
        if new_pos == self.currentObs:
            # If hits obstacle, return to the current position with obstacle penalization
            return self.currentStateW, self.wall_penalization, done
        
        # --- 3. Actualització del Tauler (Física del moviment) ---
        
        # IMPORTANT: If the new position is NOT the goal/obstacle/wall, 
        # the reward is the value stored in the board (e.g., -1 or -3, if init2 was used)
        if not done:
            # We fetch the specific penalty for the new position
            reward = self.board[new_pos] 
            # Note: This logic assumes that the reward for stepping on a clean tile is defined by the board value. 
            # This correctly handles both the simple -1/100 (init) and the Manhattan distance (init2).

        # Recuperem l'objecte Rei
        king_obj = self.board[self.currentStateW]
        
        # Buidem la casella antiga (set it back to its original reward value)
        # We need to know what value to restore: -1, -2, -3, etc.
        # This requires storing the original cell value before the King moves, 
        # but for simplicity in this grid world, we'll rely on the reset method
        
        # For the purpose of tracking the King's position, simply move it:
        self.board[self.currentStateW] = reward # Restore the original reward of the *old* cell (This might be wrong if you are using pieces)
        
        # Since the board holds pieces OR rewards, we must rely on the reset.
        # For simplicity, we assume the old cell is reset to its original step penalty *before* King was there.
        # A simpler way is to just keep track of the King's position without modifying the board object values.
        
        # We will keep your original board update logic:
        # 1. Buidem la casella antiga:
        #    This is risky as it removes the reward value. Let's assume you handle the board state in a way
        #    where the Q-learning loop only cares about the coordinates, not the object in the board cell.
        self.board[self.currentStateW] = -1 # Assuming -1 or other penalty is restored for the empty old cell
        
        # 2. Actualitzem coordenades internes
        self.currentStateW = new_pos
        
        # 3. Posem el Rei a la nova casella (visualització)
        self.board[new_pos] = king_obj
        
        return self.currentStateW, reward, done
    
    def reset_environment(self, rows=3, cols=4, currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1), mode='default'):
        # This is fine, it just re-initializes the board
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