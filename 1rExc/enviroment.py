import numpy as np
import piece 

class Enviroment:
    initState = None
    
    rows = 0
    cols = 0
        
    board = []
    # coord Agent (Rei Blanc)
    currentStateW = []
    # coord Recompensa final (Objectiu)
    currentStateB = []
    # Recompensa per moviment (pas)
    reward = -1
    # Bonificació final
    treasure = 100
    # Wall penalization
    wall_penalization = -100
    
    def __init__(self, rows=3, cols=4, currentStateW=(2,0), currentStateB=(2,3)):

        if self.initState is None:
            self.rows = rows
            self.cols = cols
            
            # 1. Inicialitzem amb dtype=object per poder guardar números I peces
            self.board = np.full((rows, cols), -1, dtype=object)
            
            self.currentStateW = currentStateW
            self.currentStateB = currentStateB

            # 2. Posem el valor 100 a la posició B
            self.board[self.currentStateB] = 100
            
            # 3. Posem el King a la posició W
            self.board[self.currentStateW] = piece.King(True) 
            
            self.initState = 1

    def get_enviroment(self):
        return self.board # Retornem el tauler o l'estat rellevant

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
                elif (r, c) == self.currentStateB: # Millor comprovar coord que valor
                    symbol = "100" 
                elif cell_value == -1:
                    symbol = "- "
                else:
                    symbol = "? " 
                
                row_display += f"{symbol} | " if symbol != "100" else f"{symbol}| "
            print(row_display)
            print("-" * (self.cols * 4 + 1))
        
    def move_piece(self, action):
        """
        Mou el rei a la nova posició basant-se en l'acció rebuda (int).
        Retorna: (next_state, reward, done)
        Accions esperades (basat en agent.py): 
        0: Up, 1: Down, 2: Right, 3: Left
        """
        current_r, current_c = self.currentStateW
        
        # 1. Definició de Deltas (Canvi de coordenades segons l'acció)
        # Format: (delta_fila, delta_columna)
        deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, 1),   # Right
            3: (0, -1)   # Left
        }
        
        # Recuperem el desplaçament. Si l'acció no existeix, no ens movem (0,0)
        dr, dc = deltas.get(action, (0, 0))
        
        # Calculem la proposta de nova posició
        new_r = current_r + dr
        new_c = current_c + dc
        new_pos = (new_r, new_c)
        
        # --- 2. Validacions i Lògica de Moviment ---
        
        reward = self.reward # Cost per defecte (-1)
        done = False         # Per defecte no hem acabat

        # A) Validació de límits del tauler (Murs)
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            # Si surt del tauler: Penalització forta i NO es mou
            return self.currentStateW, self.wall_penalization, done

        # B) Comprovem si hem arribat a l'objectiu
        if new_pos == self.currentStateB:
            reward = self.treasure
            done = True # Episodi acabat

        # --- 3. Actualització del Tauler (Física del moviment) ---

        # Recuperem l'objecte Rei
        king_obj = self.board[self.currentStateW]
        
        # Buidem la casella antiga
        self.board[self.currentStateW] = -1 
        
        # Actualitzem coordenades internes
        self.currentStateW = new_pos
        
        # Posem el Rei a la nova casella (visualització)
        # Nota: Si és l'objectiu, tècnicament el 'mengem', però visualment posem el rei igualment
        self.board[new_pos] = king_obj
        
        return self.currentStateW, reward, done