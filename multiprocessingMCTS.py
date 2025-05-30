import math
import random
import pyspiel
import multiprocessing
import time

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state         # an OpenSpiel state object
        self.parent = parent       
        self.move = move           
        self.children = {}         
        self.visit_count = 0
        self.total_value = 0.0

    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        if self.is_terminal():
            return True
        legal_moves = self.state.legal_actions()
        return len(self.children) == len(legal_moves)

    def value(self):
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

class MCTS:
    def __init__(self, exploration_weight=1.0, gamma=1.0):
        """
        exploration_weight: controls the trade-off between exploration and exploitation.
        gamma: discount factor (set to 1.0 for undiscounted rollouts).
        """
        self.exploration_weight = exploration_weight
        self.gamma = gamma

    def search(self, initial_state, num_simulations):
        # Use the current player of the root state as the perspective for rewards.
        root_player = initial_state.current_player()
        root = Node(initial_state)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # 1. Selection: Traverse the tree until reaching a node that is not fully expanded or is terminal.
            while node.is_fully_expanded() and not node.is_terminal():
                move, node = self._select_child(node)
                search_path.append(node)

            # 2. Expansion: If the node is non-terminal, expand one of its untried children.
            if not node.is_terminal():
                legal_moves = node.state.legal_actions()
                untried_moves = [m for m in legal_moves if m not in node.children]
                if untried_moves:
                    move = random.choice(untried_moves)
                    next_state = node.state.clone()
                    next_state.apply_action(move)
                    child_node = Node(next_state, parent=node, move=move)
                    node.children[move] = child_node
                    node = child_node
                    search_path.append(node)

            # 3. Simulation (Rollout): Run a random simulation from the new node until terminal.
            reward = self._rollout(node.state, root_player)

            # 4. Backpropagation: Update each node along the path with the obtained reward.
            self._backpropagate(search_path, reward)

        # Choose the move from the root with the highest average reward.
        best_move = None
        best_value = -float('inf')
        for move, child in root.children.items():
            child_value = child.value()
            if child_value > best_value:
                best_value = child_value
                best_move = move

        return best_move, best_value

    def _select_child(self, node):
        total_visits = sum(child.visit_count for child in node.children.values())
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in node.children.items():
            exploitation = child.value()
            exploration = self.exploration_weight * math.sqrt(math.log(total_visits) / child.visit_count)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _rollout(self, state, root_player):
        current_state = state.clone()
        discount = 1.0
        cumulative_reward = 0.0

        # In tic tac toe, intermediate rewards are zero; we only get a reward at terminal states.
        while not current_state.is_terminal():
            legal_moves = current_state.legal_actions()
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            current_state.apply_action(move)

        # Get the terminal reward from the perspective of the root player.
        terminal_reward = current_state.returns()[root_player]
        cumulative_reward += discount * terminal_reward

        return cumulative_reward

    def _backpropagate(self, search_path, reward):
        # Propagate the reward back up the tree.
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += reward
            reward *= self.gamma

def play_full_game(game_id):
    game = pyspiel.load_game("go")
    state = game.new_initial_state()
    start_time = time.time()
    mcts = MCTS(exploration_weight=0.5, gamma=.98)

    while not state.is_terminal():
        best_move, best_value = mcts.search(state, num_simulations=500)
        state.apply_action(best_move)

    end_time = time.time()

    print(f"Game {game_id} finished in {(end_time-start_time):.2f} seconds")
    

def run_multiple_games(num_games=20):
    # time how long to complete all games
    start_time = time.time()
    # Create a pool of processes, each running a game
    with multiprocessing.Pool(processes=num_games) as pool:
        # Map the function to the number of games, running each game in a separate process
        pool.map(play_full_game, range(num_games))

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal Time for {num_games} games: {total_time:.2f} seconds")

if __name__ == "__main__":
    # change num_games to process that many games
    run_multiple_games(num_games=20)