import numpy as np
from racetrack import Racetrack
from racetrack import printGrid
import copy
import sys
import time
import matplotlib.pyplot as plt


def showTrajectory(ep, r):
    """
    Displays the racecar's trajectory in a given episode using Matplotlib.

    :param ep: Episode to track
    :param r: Current racetrack object
    :return:
    """
    temp_grid = copy.deepcopy(r.grid)
    for e in ep:
        temp_grid[e[0][0]][e[0][1]] = 4
    if ep[-1][2] == 100:
        delta_y = max(0, (ep[-1][0][2] + ep[-1][1][0]))
        delta_y = min(5, delta_y)
        delta_x = max(0, (ep[-1][0][3] + ep[-1][1][1]))
        delta_x = min(5, delta_x)
        final_pos = (ep[-1][0][0] - delta_y, ep[-1][0][1] + delta_x)
        temp_grid[final_pos[0]][min(len(r.grid[0]) - 2, final_pos[1])] = 4
    im = np.array(temp_grid) * 50

    plt.imshow(im)

    plt.show()


def printTrajectory(ep, r):
    """
    Prints the racecar's trajectory in a given episode to the command line.

    :param ep: Episode to track
    :param r: Current racetrack object
    :return:
    """
    temp_grid = copy.deepcopy(r.grid)
    for e in ep:
        temp_grid[e[0][0]][e[0][1]] = 4
    if ep[-1][2] == 100:
        final_pos = (ep[-1][0][0] - (ep[-1][0][2] + ep[-1][1][0]), ep[-1][0][1] + (ep[-1][0][3] + ep[-1][1][1]))
        temp_grid[final_pos[0]][min(len(r.grid[0]) - 2, final_pos[1])] = 4
    printGrid(temp_grid)


# Human generated episodes for the experiment extension
human_episodes = [
    [((31, 8, 0, 0), (1, 0), -1), ((30, 8, 1, 0), (1, 0), -1), ((28, 8, 2, 0), (1, 0), -1), ((25, 8, 3, 0), (1, 0), -1),
     ((21, 8, 4, 0), (1, 0), -1), ((16, 8, 5, 0), (1, 0), -1), ((11, 8, 5, 0), (-1, 1), -1),
     ((7, 9, 4, 1), (-1, 1), -1), ((4, 11, 3, 2), (-1, 1), -1), ((2, 14, 2, 3), (-1, 1), 100)],
    [((31, 8, 0, 0), (1, 0), -1), ((30, 8, 1, 0), (1, 1), -1), ((28, 9, 2, 1), (1, -1), -1), (
        (25, 9, 3, 0), (1, 0), -1), ((21, 9, 4, 0), (1, 0), -1), ((16, 9, 5, 0), (-1, 0), -1), (
         (12, 9, 4, 0), (-1, 0), -1), ((9, 9, 3, 0), (0, 1), -1), ((6, 10, 3, 1), (-1, 1), -1), (
         (4, 12, 2, 2), (-1, 1), -1), ((3, 15, 1, 3), (-1, 1), 100)],
    [((31, 5, 0, 0), (1, 0), -1), ((30, 5, 1, 0), (1, 0), -1), ((28, 5, 2, 0), (1, 0), -1), (
        (25, 5, 3, 0), (1, 0), -1), ((21, 5, 4, 0), (1, 0), -1), ((16, 5, 5, 0), (-1, 1), -1), (
         (12, 6, 4, 1), (-1, 1), -1), ((9, 8, 3, 2), (0, 0), -1), ((6, 10, 3, 2), (-1, 1), -1), (
         (4, 13, 2, 3), (-1, 1), 100)],
    [((31, 9, 0, 0), (1, 0), -1), ((30, 9, 1, 0), (1, 0), -1), ((28, 9, 2, 0), (1, 0), -1), (
        (25, 9, 3, 0), (1, 0), -1), ((21, 9, 4, 0), (1, 0), -1), ((16, 9, 5, 0), (-1, 0), -1), (
         (12, 9, 4, 0), (-1, 0), -1), ((9, 9, 3, 0), (0, 1), -1), ((6, 10, 3, 1), (-1, 1), -1), (
         (4, 12, 2, 2), (-1, 1), -1), ((3, 15, 1, 3), (-1, 1), 100)],
    [((31, 7, 0, 0), (1, 0), -1), ((30, 7, 1, 0), (1, 0), -1), ((28, 7, 2, 0), (1, 0), -1), (
        (25, 7, 3, 0), (1, 0), -1), ((21, 7, 4, 0), (1, 0), -1), ((16, 7, 5, 0), (0, 1), -1), (
         (11, 8, 5, 1), (-1, 0), -1), ((7, 9, 4, 1), (-1, 1), -1), ((4, 11, 3, 2), (-1, 1), -1), (
         (2, 14, 2, 3), (-1, 1), 100)],
    [((31, 4, 0, 0), (1, 0), -1), ((30, 4, 1, 0), (1, 0), -1), ((28, 4, 2, 0), (1, 0), -1), (
        (25, 4, 3, 0), (1, 1), -1), ((21, 5, 4, 1), (1, 1), -1), ((16, 7, 5, 2), (-1, -1), -1), (
         (12, 8, 4, 1), (-1, -1), -1), ((9, 8, 3, 0), (0, 1), -1), ((6, 9, 3, 1), (-1, 1), -1), (
         (4, 11, 2, 2), (-1, 1), -1), ((3, 14, 1, 3), (-1, 1), 100)],
    [((31, 6, 0, 0), (1, 0), -1), ((30, 6, 1, 0), (1, 0), -1), ((28, 6, 2, 0), (1, 0), -1), (
        (25, 6, 3, 0), (1, 0), -1), ((21, 6, 4, 0), (0, 1), -1), ((17, 7, 4, 1), (0, 0), -1),
     ((13, 8, 4, 1), (0, 0), -1), (
         (9, 9, 4, 1), (-1, 0), -1), ((6, 10, 3, 1), (-1, 1), -1), ((4, 12, 2, 2), (-1, 1), -1), (
         (3, 15, 1, 3), (-1, 1), 100)],
    [((31, 5, 0, 0), (1, 0), -1), ((30, 5, 1, 0), (1, 0), -1), ((28, 5, 2, 0), (1, 0), -1), (
        (25, 5, 3, 0), (1, 0), -1), ((21, 5, 4, 0), (1, 0), -1), ((16, 5, 5, 0), (-1, 1), -1), (
         (12, 6, 4, 1), (-1, 1), -1), ((9, 8, 3, 2), (0, 0), -1), ((6, 10, 3, 2), (-1, 1), -1), (
         (4, 13, 2, 3), (-1, 1), 100)],
    [((31, 7, 0, 0), (1, 0), -1), ((30, 7, 1, 0), (1, 0), -1), ((28, 7, 2, 0), (1, 0), -1), (
        (25, 7, 3, 0), (1, 0), -1), ((21, 7, 4, 0), (1, 0), -1), ((16, 7, 5, 0), (-1, 0), -1), (
         (12, 7, 4, 0), (-1, 1), -1), ((9, 8, 3, 1), (0, 1), -1), ((6, 10, 3, 2), (-1, 1), -1), (
         (4, 13, 2, 3), (-1, 1), 100)],
    [((31, 6, 0, 0), (1, 0), -1), ((30, 6, 1, 0), (1, 0), -1), ((28, 6, 2, 0), (1, 0), -1), (
        (25, 6, 3, 0), (1, 0), -1), ((21, 6, 4, 0), (1, 0), -1), ((16, 6, 5, 0), (-1, 1), -1), (
         (12, 7, 4, 1), (-1, 1), -1), ((9, 9, 3, 2), (0, -1), -1), ((6, 10, 3, 1), (-1, 1), -1), (
         (4, 12, 2, 2), (-1, 1), -1), ((3, 15, 1, 3), (-1, 1), 100)]
]


class Agent:
    """
    Monte Carlo Control Agent. Implements first-visit on-policy MC control to
    solve the racetrack problem presented in exercise 5.12 of Sutton and Barto
    """

    def __init__(self, epsilon, gamma):
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = None
        self.returns = None
        self.n = None
        self.total_reward = 0
        self.running_ten_avg = 0
        self.running_hundred_avg = 0
        self.actions = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 0), (0, 1),
                        (1, -1), (1, 0), (1, 1)]

    def onPolicyFirstVisitMCControl(self, r, include_human=False):
        """
        On-policy first-visit MC control implementation, pseudocode found on
        page 101 of Sutton and Barto.

        :param r: Current racetrack object
        :param include_human: Whether to include human generated episodes
        :return: None, modifies the agent's Q table
        """
        grid_height = len(r.grid)
        grid_width = len(r.grid[0])
        # Q value function that takes in the state,
        # {y pos, x pos, vertical speed, horizontal speed}, and an
        # action, a, and returns the value
        self.q = {((y, x, v, h), a): 0
                  for y in range(grid_height)
                  for x in range(grid_width)
                  for v in range(6)
                  for h in range(6)
                  for a in self.actions}

        # Returns of the agent, stored by state action pairs
        self.returns = {((y, x, v, h), a): 0
                        for y in range(grid_height)
                        for x in range(grid_width)
                        for v in range(6)
                        for h in range(6)
                        for a in self.actions}

        # Visit count table, takes in the state and action and returns
        # the number of visits
        self.n = {((y, x, v, h), a): 0
                  for y in range(grid_height)
                  for x in range(grid_width)
                  for v in range(6)
                  for h in range(6)
                  for a in self.actions}

        # First visit on-policy Monte Carlo control, loop until the
        # difference in weighted avg of past 10 and 100 total rewards
        # from an episode to detect convergence
        # episode_rewards = [0]
        i = 0
        while abs(self.running_ten_avg - self.running_hundred_avg) > 1e-6 or i < 1000:
            if i % 10000 == 0:
                print("At episode:", str(i))
            np.random.seed(i)

            # Generating an episode
            curr = Racetrack(r.grid)
            self.total_reward = 0
            if include_human and i < len(human_episodes):
                ep = human_episodes[i]
                for triple in ep:
                    self.total_reward += triple[2]
            else:
                ep = self.generateEpisode(curr)

            # episode_rewards.append(episode_rewards[-1] + (1 / (len(episode_rewards) + 1) *
            #                        (self.total_reward - episode_rewards[-1])))

            # Calculating approximate running weighted averages of the
            # past 10 and 100 total rewards to detect when we've converged
            self.running_ten_avg += 0.9 * (self.total_reward - self.running_ten_avg)
            self.running_hundred_avg += 0.99 * (self.total_reward - self.running_hundred_avg)

            # Calculating the returns backwards starting from the last return
            G = 0
            for j in range(len(ep) - 1, -1, -1):
                G = self.gamma * G + ep[j][2]
                self.returns[(ep[j][0], ep[j][1])] = G
                self.n[(ep[j][0], ep[j][1])] += 1

            # Now setting Q values going forwards
            visited = {}
            for j in range(len(ep)):
                # If we've already seen this state, action pair, ignore it
                if (ep[j][0], ep[j][1]) in visited:
                    continue

                # Update the Q value with the sample average of the returns at
                # the given state, action pair
                visited[(ep[j][0], ep[j][1])] = True
                self.q[(ep[j][0], ep[j][1])] += (1 / self.n[(ep[j][0], ep[j][1])]) * \
                                                (self.returns[(ep[j][0], ep[j][1])] - self.q[(ep[j][0], ep[j][1])])
            i += 1
        print(f"Maximum episode: {i}")
        # plt.plot(episode_rewards[1:])
        # plt.show()

    def generateEpisode(self, r, show_grid=False, include_noise=True):
        """
        Generates an episode of the racetrack problem for the agent.

        :param r: Current racetrack object
        :param show_grid: Flag used to display episode's trajectory
        :param include_noise: Whether to include random noise
        :return: Episode of the racetrack problem
        """
        ep = []
        while True:
            S = (r.getPos()[0], r.getPos()[1], r.getVelocity()[0], r.getVelocity()[1])
            q_vals = [self.q[(r.getPos()[0], r.getPos()[1], r.getVelocity()[0], r.getVelocity()[1]), a] for a in
                      self.actions]
            # Exploiting
            random_val = np.random.rand()
            if random_val > self.epsilon:
                m = max(q_vals)
                max_indices = [i for i, x in enumerate(q_vals) if x == m]
                A = np.random.choice(max_indices)
            # If we are exploring
            else:
                A = np.random.randint(0, len(self.actions))
            # We choose the action (0,0) to make the env more
            # challenging with probability 0.1 - see exercise 5.12
            if include_noise:
                challenge_val = np.random.rand()
                if challenge_val < 0.1:
                    A = 3
            y_vel, x_vel = self.actions[A]

            # We are out of bounds
            if not r.move(y_vel, x_vel):
                R = -100
                self.total_reward += R
                ep.append((S, self.actions[A], R))
                break

            # We've crossed the finish line
            if r.winCond():
                R = 100
                self.total_reward += R
                ep.append((S, self.actions[A], R))
                break
            R = -1
            self.total_reward += R
            ep.append((S, self.actions[A], R))
        if show_grid:
            showTrajectory(ep, r)
        return ep


# Driver loop of the simulator
def main():
    grid = []
    with open(sys.argv[1]) as f:
        while True:
            line = f.readline()
            if not line:
                break
            grid.append([int(x) for x in line.split()])

    print("Welcome to the on-policy Monte Carlo control racetrack simulator!")
    print("Would you like to select the agent's epsilon value? [y/n] Default value is 0.1")
    yes = input()
    if yes == 'y':
        print("What would you like the agent's epsilon value to be?")
        epsilon = float(input())
    else:
        epsilon = 0.1
    print("Would you like to select the agent's gamma value? [y/n] Default value is 0.9")
    yes = input()
    if yes == 'y':
        print("What would you like the agent's gamma value to be?")
        gamma = float(input())
    else:
        gamma = 0.9

    a = Agent(epsilon, gamma)
    r = Racetrack(grid)
    a.onPolicyFirstVisitMCControl(r, include_human=False)

    # Setting our epsilon to 0 to be purely greedy
    a.epsilon = 0.0
    print("Now generating five episodes with the learned policy...")
    for i in range(5):
        print(i)
        np.random.seed(int(time.time()))
        curr = Racetrack(grid)
        ep = a.generateEpisode(curr, show_grid=True, include_noise=False)
        print(ep)


if __name__ == '__main__':
    main()
