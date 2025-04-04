import math
import random
import pyspiel

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
        # Ensure parent visit count is at least 1 for log calculation
        # (it should be > 0 if we are selecting a child)
        log_parent_visits = math.log(max(1, node.visit_count)) # Use parent's visits

        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in node.children.items():
            if child.visit_count == 0:
                 score = child.value() + self.exploration_weight * float('inf') 
            else:
                exploitation = child.value()
                exploration = self.exploration_weight * math.sqrt(log_parent_visits / child.visit_count)
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        if best_child is None and node.children:
             best_move = random.choice(list(node.children.keys()))
             best_child = node.children[best_move]

        return best_move, best_child

    def _rollout(self, state, root_player):
        current_state = state.clone()
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
        cumulative_reward += terminal_reward

        return cumulative_reward

    def _backpropagate(self, search_path, reward):
        # Propagate the reward back up the tree.
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += reward
            reward *= self.gamma

def play_full_game():
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    mcts = MCTS(exploration_weight=0.5, gamma=.98)

    print("Starting Tic Tac Toe Game")
    while not state.is_terminal():
        print("Current state:")
        print(state)
        # Each turn, the current player uses MCTS to choose a move.
        best_move, best_value = mcts.search(state, num_simulations=20000)
        print(f"Player {state.current_player()} selects move: {best_move} (value: {best_value})\n")
        state.apply_action(best_move)

    print("Final state:")
    print(state)
    print("Returns:", state.returns())

if __name__ == "__main__":
    play_full_game()
