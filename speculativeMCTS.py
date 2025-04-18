import math
import random
import pyspiel
import time
import ray
import numpy as np
from collections import deque

# ========================
#      CONFIG SECTION     
# ========================
GAME_NAME = "checkers"
NUM_GAMES = 20               # Number of games to play for comparison
NUM_WORKERS = 30            # Number of parallel workers for speculation
TOTAL_ITERATIONS = 10000       # Total MCTS simulations per move
SPLIT_RATIO = 0.90            # Ratio of simulations for speculation vs verification
CACHE_ENABLED = True         # Enable/disable neural network cache simulation
# ========================

# Initialize Ray
ray.init()

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.is_expanded = False
        self.visit_count = 0
        self.total_reward = 0.0

    def expand(self):
        """Expand node with all legal actions"""
        for action in self.state.legal_actions():
            child_state = self.state.clone()
            child_state.apply_action(action)
            self.children.append(MCTSNode(child_state, self, action))
        self.is_expanded = True

class MCTSBot:
    def __init__(self, game, player_id):
        self.game = game
        self.player_id = player_id
        self.root = None
        self.cache = {} if CACHE_ENABLED else None

    def get_action(self, state, num_iterations):
        """Safe action selection with fallback"""
        if state.is_terminal():
            raise ValueError("Cannot choose action for terminal state")
            
        if self.root is None or self.root.state != state:
            self.root = MCTSNode(state.clone())
        
        for _ in range(num_iterations):
            node = self.root
            path = [node]

            # Selection phase
            while not node.state.is_terminal() and node.is_expanded:
                node = self._select_child(node)
                path.append(node)

            # Expansion
            if not node.state.is_terminal():
                if not node.is_expanded:
                    node.expand()
                    node.is_expanded = True
                node = random.choice(node.children)
                path.append(node)

            # Simulation with caching
            reward = self._simulate_with_cache(node.state)

            # Backpropagation
            for n in path:
                n.visit_count += 1
                n.total_reward += reward
        if not self.root.children:
            return random.choice(state.legal_actions())
        return max(self.root.children, key=lambda c: c.visit_count).action

    def _simulate_with_cache(self, state):
        """Simulate with simple cache implementation"""
        if self.cache is not None:
            state_str = str(state)
            if state_str in self.cache:
                return self.cache[state_str]
            
        reward = self._simulate(state)
        
        if self.cache is not None:
            self.cache[state_str] = reward
        return reward

    def _select_child(self, node):
        """UCB1 selection with exploration constant"""
        best_score = -np.inf
        best_child = None
        for child in node.children:
            if child.visit_count == 0:
                score = np.inf
            else:
                exploitation = child.total_reward / child.visit_count
                exploration = math.sqrt(math.log(node.visit_count + 1) / child.visit_count)
                score = exploitation + 0.3 * exploration
            if score > best_score:
                best_score, best_child = score, child
        return best_child

    def _simulate(self, state):
        """Random rollout simulation"""
        current_state = state.clone()
        while not current_state.is_terminal():
            action = random.choice(current_state.legal_actions())
            current_state.apply_action(action)
        return current_state.rewards()[self.player_id]

@ray.remote
class SpeculativeWorker:
    def __init__(self, game_name):
        self.game = pyspiel.load_game(game_name)
        self.bot = MCTSBot(self.game, None)
    
    def compute_move(self, state, player_id, num_iterations):
        try:
            if state.is_terminal():
                return None  # No action for terminal states
                
            self.bot.player_id = player_id
            return self.bot.get_action(state, num_iterations)
        except:
            return None

class SpeculativeMCTS:
    def __init__(self, game_name, num_workers=NUM_WORKERS):
        self.game = pyspiel.load_game(game_name)
        self.num_workers = num_workers
        self.workers = [SpeculativeWorker.remote(game_name) for _ in range(num_workers)]
        self.partial_iterations = int(TOTAL_ITERATIONS * SPLIT_RATIO)
        self.remaining_iterations = TOTAL_ITERATIONS - self.partial_iterations

    def play_games(self, num_games):
        total_time = 0
        for game_num in range(1, num_games+1):
            start_time = time.time()
            state = self.game.new_initial_state()
            speculative_futures = []
            
            while not state.is_terminal():
                current_player = state.current_player()
                
                # Phase 1: Partial MCTS
                bot = MCTSBot(self.game, current_player)
                try:
                    tentative_action = bot.get_action(state.clone(), self.partial_iterations)
                except ValueError:
                    break
                
                # Speculative execution only for non-terminal states
                if len(speculative_futures) < self.num_workers:
                    spec_state = state.clone()
                    try:
                        spec_state.apply_action(tentative_action)
                    except pyspiel.SpielError:
                        break
                        
                    if not spec_state.is_terminal():
                        future = self.workers[len(speculative_futures)].compute_move.remote(
                            spec_state, current_player, TOTAL_ITERATIONS
                        )
                        speculative_futures.append((future, tentative_action))

                # Phase 2: Complete MCTS
                try:
                    final_action = bot.get_action(state.clone(), self.remaining_iterations)
                except ValueError:
                    break
                
                # Validate speculation
                used_speculation = False
                for i, (future, action) in enumerate(speculative_futures):
                    if action == final_action:
                        try:
                            next_action = ray.get(future)
                            if next_action is not None:
                                state.apply_action(final_action)
                                state.apply_action(next_action)
                                used_speculation = True
                                speculative_futures.pop(i)
                                break
                        except:
                            pass
                
                if not used_speculation:
                    try:
                        state.apply_action(final_action)
                    except pyspiel.SpielError:
                        break
                
                speculative_futures = []
            
            game_time = time.time() - start_time
            total_time += game_time
            print(f"Game {game_num} completed in {game_time:.2f} seconds")
        
        return total_time


def run_baseline(num_games):
    """Run baseline MCTS without speculation"""
    total_time = 0
    game = pyspiel.load_game(GAME_NAME)
    for game_num in range(1, num_games+1):
        start_time = time.time()
        state = game.new_initial_state()
        bot = MCTSBot(game, 0)
        
        while not state.is_terminal():
            action = bot.get_action(state.clone(), TOTAL_ITERATIONS)
            state.apply_action(action)
        
        game_time = time.time() - start_time
        total_time += game_time
        print(f"Game {game_num} completed in {game_time:.2f} seconds")
    
    return total_time

if __name__ == "__main__":
    
    spec_start = time.time()
    spec_time = SpeculativeMCTS(GAME_NAME).play_games(NUM_GAMES)
    spec_total = time.time() - spec_start

    print(f"Total Time for {NUM_GAMES} games: {spec_total:.2f} seconds")
    print(f"Average Time for {NUM_GAMES} games: {(spec_total/NUM_GAMES):.2f} seconds")