import numpy as np
import piece 

class Enviroment:
    initState = None
    
    rows = 0
    cols = 0
        
    board = []
    # Agent
    currentStateW = []
    # Recompensa   
    currentStateB = []
    
    def __init__(self, initState=None, rows=3, cols=4, currentStateW=(2,0), currentStateB=(2,3)):
        self.rows = rows
        self.cols = cols
        
        # 1. Inicialitzem amb dtype=object per poder guardar números I peces
        self.board = np.full((rows, cols), -1, dtype=object)
        
        self.currentStateW = currentStateW
        self.currentStateB = currentStateB

        # (S'ha eliminat el bucle for/append redundant i erroni)

        if initState is None:
            # 2. Posem el valor 100 a la posició B
            self.board[self.currentStateB] = 100
            
            # 3. Posem el King a la posició W (això sobreescriu el valor, actuant com a punt 0/Inici)
            self.board[self.currentStateW] = piece.King(True) 
            
            self.initState = 1
        else:
            return


    def get_state(self):
        """
        Retorna la posició actual de l'agent (Rei Blanc).
        """
        return self.currentStateW

    def get_target_pos(self):
        """
        Retorna la posició de l'objectiu (Casella 100).
        """
        return self.currentStateB

    def set_target_pos(self, new_pos):
        """
        Defineix una nova posició per a l'objectiu (B).
        Això requereix actualitzar el tauler i l'estat intern.
        """
        new_r, new_c = new_pos
        
        # 1. Validació de límits
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            print(f"Error: Posició objectiu fora de límits: {new_pos}")
            return False

        # 2. Esborrem l'objectiu antic (-1) i actualitzem la posició B
        self.board[self.currentStateB] = -1
        self.currentStateB = new_pos
        
        # 3. Definim el nou objectiu amb valor 100
        self.board[self.currentStateB] = 100
        
        return True
          
    def print_board(self):
        """
        Mostra el tauler de joc de forma visual, utilitzant K per al Rei 
        i 100 per a l'objectiu.
        """
        print("-" * (self.cols * 4 + 1))
        for r in range(self.rows):
            row_display = "| "
            for c in range(self.cols):
                cell_value = self.board[r, c]
                
                if (r, c) == self.currentStateW:
                    # Si la posició coincideix amb l'estat actual W, mostrem el Rei
                    symbol = "K " 
                elif cell_value == 100:
                    # Si el valor és 100, és l'objectiu
                    symbol = "100" 
                elif cell_value == -1:
                    # La casella és buida (default)
                    symbol = "- "
                else:
                    # En cas de tenir un altre objecte (però amb la lògica actual, no hauria de passar)
                    symbol = "? " 
                
                row_display += f"{symbol} | " if symbol != "100" else f"{symbol}| "
            print(row_display)
            print("-" * (self.cols * 4 + 1))
        
    def move_piece(self, new_pos):
        """
        Mou el rei a la nova posició si és vàlid.
        Retorna la recompensa obtinguda.
        """
        current_r, current_c = self.currentStateW
        new_r, new_c = new_pos

        # --- 1. Validacions ---
        
        # A) Validació de límits del tauler
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            print(f"Moviment invàlid: Fora de límits {new_pos}")
            return -10 # Penalització forta

        # B) Validació de geometria del Rei (distància màxima de 1 en qualsevol eix)
        # Chebyshev distance: max(|x1-x2|, |y1-y2|)
        if max(abs(new_r - current_r), abs(new_c - current_c)) > 1:
            print(f"Moviment invàlid: El Rei no pot saltar a {new_pos}")
            return -10 

        # --- 2. Càlcul de Recompensa ---
        
        reward = -1 # Cost estàndard per pas (per incentivar camins curts)
        
        if new_pos == self.currentStateB:
            reward = 100 # Recompensa final

        # --- 3. Actualització del Tauler (Swap) ---

        # Recuperem l'objecte Rei de la posició actual
        king_obj = self.board[self.currentStateW]
        
        # Buidem la casella antiga (tornem a posar -1 o 0 segons la teva lògica de buit)
        self.board[self.currentStateW] = -1
        
        # Posem el Rei a la nova casella
        self.board[new_pos] = king_obj
        
        # Actualitzem les coordenades internes de l'agent
        self.currentStateW = new_pos
        
        return reward
