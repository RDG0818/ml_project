import math
import random
import pyspiel

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state         # an open_spiel state object
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
        root = Node(initial_state)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # 1. Selection: Traverse until reaching a node that is not fully expanded or is terminal.
            while node.is_fully_expanded() and not node.is_terminal():
                move, node = self._select_child(node)
                search_path.append(node)

            # 2. Expansion: If the node is non-terminal, expand one of its unvisited children.
            if not node.is_terminal():
                legal_moves = node.state.legal_actions()
                untried_moves = [m for m in legal_moves if m not in node.children]
                if untried_moves:
                    move = random.choice(untried_moves)
                    next_state = node.state.clone()  # clone to avoid mutating the original state
                    next_state.apply_action(move)
                    child_node = Node(next_state, parent=node, move=move)
                    node.children[move] = child_node
                    node = child_node
                    search_path.append(node)

            # 3. Simulation (Rollout): Perform a random rollout from the new node until a terminal state.
            reward = self._rollout(node.state)

            # 4. Backpropagation: Update the nodes along the path.
            self._backpropagate(search_path, reward)

        # Choose the best move based on the highest average value.
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

    def _rollout(self, state):
        """
        Perform a random rollout (simulation) from the given state until a terminal state is reached.
        In many OpenSpiel games, rewards are only given at terminal states.
        """
        current_state = state.clone()  # clone to avoid side-effects
        cumulative_reward = 0.0
        discount = 1.0

        # If the game supports intermediate rewards, you could accumulate them here.
        # For most OpenSpiel games, intermediate rewards are zero.
        while not current_state.is_terminal():
            legal_moves = current_state.legal_actions()
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            current_state.apply_action(move)
            # Assume no intermediate reward (or set reward=0) in most OpenSpiel games.
            reward = 0.0
            cumulative_reward += discount * reward
            discount *= self.gamma

        # At terminal state, get the reward from the state's returns.
        # Here we assume a single-agent or a zero-sum game and use player 0's reward.
        terminal_reward = current_state.returns()[0]
        cumulative_reward += discount * terminal_reward

        return cumulative_reward

    def _backpropagate(self, search_path, reward):
        """
        Propagate the reward back up the search path.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += reward
            reward *= self.gamma

def test_mcts():
    # Load an OpenSpiel game. Here we use tic_tac_toe as an example.
    game = pyspiel.load_game("tic_tac_toe")
    initial_state = game.new_initial_state()

    mcts = MCTS(exploration_weight=1.0, gamma=1.0)
    best_move, best_value = mcts.search(initial_state, num_simulations=1000)

    print(f"Best move: {best_move}")
    print(f"Best value: {best_value}")

    # Apply the best move to a clone of the initial state.
    next_state = initial_state.clone()
    next_state.apply_action(best_move)
    print("Next state:")
    print(next_state)

if __name__ == "__main__":
    test_mcts()
