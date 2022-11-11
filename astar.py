from math import dist
from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueue, manhattanDistance


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """

    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules())
    )


def getNumCapsules(state):
    return len(state.getCapsules())


def heuristic(state, list_food, remaining_dist):
    """Given a situation returns an approximation of number of turn left.

    Arguments:
        A game state and the list containing all the food positions.

    Returns:
        A value of the approximation of number of turn left
    """

    pacman_pos = state.getPacmanPosition()

    # If there is no food left, return 0
    if len(list_food) == 0:
        return 0

    min_dist = float("inf")

    for i in range(len(list_food)):

        # Distance between pacman and food
        dist = manhattanDistance(pacman_pos, list_food[i]) + remaining_dist[i]

        if dist < min_dist:
            min_dist = dist
            if min_dist == 0:
                break

    return (min_dist)


def dist_food(last_food, last_list_food):
    """
    Arguments:
        A food's position and the list of all the food positions.

    Returns:
        A heuristic value of the distance left before winning the game after
    eating the food.
    """
    tot_dist = 0
    list_food = last_list_food.copy()
    list_food.remove(last_food)
    while list_food != []:

        min_dist = float("inf")
        min_food = (-1, -1)

        for food in list_food:
            dist = manhattanDistance(last_food, food)
            if (dist < min_dist):
                min_dist = dist
                min_food = food
                if min_dist == 1:
                    break

        tot_dist += min_dist
        list_food.remove(min_food)
        last_food = min_food

    return (tot_dist)


class PacmanAgent(Agent):
    """Pacman agent based on ASTAR."""

    def __init__(self):
        super().__init__()

        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """

        if self.moves is None:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def astar(self, state):
        """Given a Pacman game state, returns a list of legal moves to solve
        the search layout.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """

        path = []
        fringe = PriorityQueue()
        fringe.push((state, path, 0), 0)
        closed = set()

        # We compute the list of all the foods positions
        list_all_food = []
        food = state.getFood()
        for x in range(food.width):
            for y in range(food.height):
                if food[x][y]:
                    list_all_food.append((x, y))

        while True:
            if fringe.isEmpty():
                return []

            _, item = fringe.pop()
            current, path, last_g_cost = item

            if current.isWin():
                return path

            current_key = key(current)
            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            food = current.getFood()
            list_food = []
            for x, y in list_all_food:
                if food[x][y]:
                    list_food.append((x, y))

            remaining_dist = []
            for food in list_food:
                remaining_dist.append(dist_food(food, list_food))

            current_score = current.getScore()
            for successor, action in current.generatePacmanSuccessors():

                h_cost = heuristic(successor, list_food, remaining_dist)

                next_score = successor.getScore()
                if next_score < current_score:
                    turn_cost = current_score - next_score
                else:
                    turn_cost = 1

                g_cost = last_g_cost + turn_cost
                state_value = g_cost + h_cost

                fringe.push((successor, path + [action], g_cost), state_value)

        return path
