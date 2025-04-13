import math
import random
import pyspiel
import time

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
        """Expands the node by creating child nodes for all legal actions"""
        for action in self.state.legal_actions():
            child_state = self.state.clone()
            child_state.apply_action(action)
            self.children.append(MCTSNode(child_state, self, action))
        self.is_expanded = True

class MCTSBot:
    def __init__(self, game, player_id, num_iterations=100000):
        self.game = game
        self.player_id = player_id
        self.num_iterations = num_iterations

    def get_action(self, state):
        """Performs MCTS search and returns the best action"""
        root = MCTSNode(state.clone())

        for _ in range(self.num_iterations):
            node = root
            path = [node]

            # Selection phase
            while not node.state.is_terminal() and node.is_expanded:
                node = self._select_child(node)
                path.append(node)

            # Expansion and simulation
            if not node.state.is_terminal():
                if not node.is_expanded:
                    node.expand()
                    node.is_expanded = True

                node = random.choice(node.children)
                path.append(node)
                reward = self._simulate(node.state)
            else:
                reward = node.state.rewards()[self.player_id]

            # Backpropagation
            for n in path:
                n.visit_count += 1
                n.total_reward += reward

        return max(root.children, key=lambda c: c.visit_count).action

    def _select_child(self, node):
        """Selects child node using UCB1 exploration policy"""
        best_score = -float('inf')
        best_child = None

        for child in node.children:
            if child.visit_count == 0:
                score = float('inf')
            else:
                exploitation = child.total_reward / child.visit_count
                exploration = math.sqrt(math.log(node.visit_count) / child.visit_count)
                score = exploitation + 0.3 * exploration  # UCB1 constant

            if score > best_score:
                best_score, best_child = score, child

        return best_child

    def _simulate(self, state):
        """Simulates random playout from given state to terminal"""
        current_state = state.clone()
        while not current_state.is_terminal():
            action = random.choice(current_state.legal_actions())
            current_state.apply_action(action)
        return current_state.rewards()[self.player_id]

if __name__ == "__main__":
    game_name = "connect_four"
    num_games = 20
    
    print("\nRunning sample match:")
    
    bots = {
        0: MCTSBot(game_name, 0),
        1: MCTSBot(game_name, 1)
    }
    
    start_time = time.time()
    for i in range(20):
        print(f"Playing Game {i+1}")
        game = pyspiel.load_game(game_name)
        state = game.new_initial_state()

        while not state.is_terminal():
            current_player = state.current_player()
            action = bots[current_player].get_action(state)
            state.apply_action(action)
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal Time for {num_games} games: {total_time:.2f} seconds")