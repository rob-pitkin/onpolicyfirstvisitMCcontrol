import numpy as np
import copy


def printGrid(temp_grid):
    """
    Nicely prints the racetrack grid.

    :param temp_grid: Copy of the racetrack grid to alter
    :return: None
    """
    print("|", end='')
    for i in range(len(temp_grid[0])):
        print("-", end='')
    print("|")
    for r in temp_grid:
        print("|", end='')
        for c in r:
            if c == 0:
                print(" ", end='')
            elif c == 1:
                print("W", end='')
            elif c == 3:
                print("E", end='')
            elif c == 2:
                print("S", end='')
            elif c == 4:
                print("R", end='')
        print("|")
    print("|", end='')
    for i in range(len(temp_grid[0])):
        print("-", end='')
    print("|")


class Racetrack:
    """
    Racetrack environment class. Maintains agent's
    position and velocity (i.e. the state).
    """

    def __init__(self, grid):
        self.grid_test = [[0, 1, 1, 3, 0],
                          [0, 1, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 2, 0, 0, 0]]
        self.grid = grid
        self.velocity = [0, 0]
        self.pos = self.getRandomStartPos()
        self.finishEdge = -1
        for r in self.grid:
            if r[-2] == 3:
                self.finishEdge += 1
            else:
                break

    def getRandomStartPos(self):
        """
        Generates random starting position for the racecar.

        :return: Position of racecar
        """
        indices = [i for i, x in enumerate(self.grid[-1]) if x == 2]
        return len(self.grid) - 1, np.random.choice(indices)

    def winCond(self):
        """
        Checks if the racecar has passed the finish line.

        :return: True if racecar has crossed the finish line, otherwise false.
        """
        if self.grid[self.pos[0]][self.pos[1]] == 3:
            return True
        return False

    def move(self, y_component, x_component):
        """
        Changes the racecar's position according to current position,
        current velocity, and the selected change in velocity (action).

        :param y_component: Vertical component of the selected action
        :param x_component: Horizontal component of the selected action
        :return: True if the racecar is still on the track, else false
        """
        if (self.velocity[0] + y_component, self.velocity[1] + x_component) != (0, 0):
            self.velocity[0] += y_component
            self.velocity[0] = max(0, self.velocity[0])
            self.velocity[0] = min(5, self.velocity[0])

            self.velocity[1] += x_component
            self.velocity[1] = max(0, self.velocity[1])
            self.velocity[1] = min(5, self.velocity[1])

        newPos = [self.pos[0] - self.velocity[0], self.pos[1] + self.velocity[1]]
        if 0 <= newPos[0] <= self.finishEdge and newPos[1] >= len(self.grid[0]) - 2:
            self.pos = [newPos[0], len(self.grid[0]) - 2]
            return True
        if newPos[0] >= len(self.grid) or newPos[0] < 0 \
                or newPos[1] >= len(self.grid[0]) or newPos[1] < 0 \
                or self.grid[newPos[0]][newPos[1]] == 0:
            return False
        else:
            self.pos = newPos
            return True

    def getVelocity(self):
        """
        Gets the agent's velocity.

        :return: Agent's velocity
        """
        return self.velocity

    def getPos(self):
        """
        Gets the agent's position.

        :return: Agent's position
        """
        return self.pos

    def printCurrGrid(self):
        """
        Prints the grid with the current agent's position.

        :return: None
        """
        temp_grid = copy.deepcopy(self.grid)
        temp_grid[self.pos[0]][self.pos[1]] = 4
        printGrid(temp_grid)
