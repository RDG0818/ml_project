import math
import time
import ray
import pyspiel
import numpy as np
from collections import defaultdict

# Initialize Ray once at startup
ray.init()

@ray.remote
class MCTSActor:
    def __init__(self, game_name, player_id, exploration=0.3):
        self.game = pyspiel.load_game(game_name)
        self.player_id = player_id
        self.exploration = exploration
        self.node_cache = {}
        
    class Node:
        __slots__ = ('visit_count', 'total_reward', 'children')
        def __init__(self):
            self.visit_count = 0
            self.total_reward = 0.0
            self.children = None

    def search(self, root_state, num_iterations=500):
        """Perform a chunk of MCTS iterations"""
        root = self.Node()
        state = root_state.clone()
        legal_actions = state.legal_actions()
        
        for _ in range(num_iterations):
            node = root
            path = [node]
            current_state = state.clone()
            
            # Selection
            while node.children is not None:
                best_score = -math.inf
                best_child = None
                for action, child in node.children.items():
                    score = self._ucb_score(child, node.visit_count)
                    if score > best_score:
                        best_score = score
                        best_action = action
                        best_child = child
                
                node = best_child
                path.append(node)
                current_state.apply_action(best_action)
                
                if current_state.is_terminal():
                    break

            # Expansion
            if not current_state.is_terminal() and node.children is None:
                node.children = {
                    action: self.Node() 
                    for action in current_state.legal_actions()
                }
                
            # Simulation
            reward = self._simulate(current_state)
            
            # Backpropagation
            for n in reversed(path):
                n.visit_count += 1
                n.total_reward += reward
        
        # Return root action statistics
        return {
            action: (child.visit_count, child.total_reward)
            for action, child in (root.children or {}).items()
        }

    def _ucb_score(self, node, parent_visits):
        if node.visit_count == 0:
            return math.inf
        return (node.total_reward / node.visit_count) + \
               self.exploration * math.sqrt(math.log(parent_visits + 1) / node.visit_count)

    def _simulate(self, state):
        """Fast simulation using pre-allocated state"""
        while not state.is_terminal():
            action = np.random.choice(state.legal_actions())
            state.apply_action(action)
        return state.rewards()[self.player_id]

class ParallelMCTSBot:
    def __init__(self, game_name, player_id, num_iterations=500, num_actors=5):
        self.game_name = game_name
        self.player_id = player_id
        self.num_iterations = num_iterations
        self.num_actors = num_actors
        
        # Create actor pool
        self.actors = [
            MCTSActor.remote(game_name, player_id)
            for _ in range(num_actors)
        ]

    def get_action(self, state):
        """Execute parallel MCTS search"""
        chunks = self._split_iterations()
        root_state = ray.put(state.clone())  # Share state efficiently
        
        # Distribute work to actors
        futures = [
            actor.search.remote(root_state, chunk_size)
            for actor, chunk_size in zip(self.actors, chunks)
        ]
        
        # Collect and aggregate results
        results = ray.get(futures)
        action_stats = defaultdict(lambda: [0, 0.0])
        
        for result in results:
            for action, (visits, reward) in result.items():
                action_stats[action][0] += visits
                action_stats[action][1] += reward
                
        return max(action_stats.items(), key=lambda x: x[1][0])[0]

    def _split_iterations(self):
        """Distribute iterations between actors"""
        base = self.num_iterations // self.num_actors
        remainder = self.num_iterations % self.num_actors
        return [base + (1 if i < remainder else 0) for i in range(self.num_actors)]

def benchmark(bot_class, game_name, num_runs=5):
    """Benchmark different bot configurations"""
    game = pyspiel.load_game(game_name)
    state = game.new_initial_state()
    
    configs = [
        ("Single Process", {"num_actors": 1}),
        ("4 Processes", {"num_actors": 4}),
        ("8 Processes", {"num_actors": 8})
    ]
    
    for label, params in configs:
        times = []
        for _ in range(num_runs):
            bot = bot_class(game_name, 0, num_iterations=10000, **params)
            start = time.time()
            bot.get_action(state)
            times.append(time.time() - start)
        
        print(f"{label}: {np.mean(times):.2f}s ± {np.std(times):.2f}")

if __name__ == "__main__":
    game_name = "go"
    num_games = 20
    
    
    bots = {
        0: ParallelMCTSBot(game_name, 0, num_actors=30),
        1: ParallelMCTSBot(game_name, 1, num_actors=30)
    }
    
    start_time = time.time()
    for i in range(20):
        game = pyspiel.load_game(game_name)
        state = game.new_initial_state()
        start_game_time = time.time()

        while not state.is_terminal():
            current_player = state.current_player()
            action = bots[current_player].get_action(state)
            state.apply_action(action)
        
        end_game_time = time.time()

        print(f"Game {i+1} finished in {(end_game_time - start_game_time):.2f} seconds")
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal Time for {num_games} games: {total_time:.2f} seconds")