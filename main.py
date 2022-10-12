from racetrack import Racetrack
import sys


def main():
    grid = []
    with open(sys.argv[1]) as f:
        while True:
            line = f.readline()
            if not line:
                break
            grid.append([int(x) for x in line.split()])

    print("Initializing racetrack...")
    r = Racetrack(grid)
    r.printCurrGrid()
    print("Ready to begin? [y/n]:")
    start = input()
    ep = []
    while start != 'q' and start != 'n':
        S = (r.getPos()[0], r.getPos()[1], r.getVelocity()[0], r.getVelocity()[1])
        print(f"Your current velocity is: {r.getVelocity()[0]}, {r.getVelocity()[1]}")
        print("Input your changes to the vertical and horizontal components [-1, 0, 1]:")
        new_vel = input().split(',')
        y_vel, x_vel = int(new_vel[0]), int(new_vel[1])
        if not r.move(y_vel, x_vel):
            r.getPos()
            ep.append((S, (y_vel, x_vel), -100))
            break
        r.printCurrGrid()
        if r.winCond():
            print("You won!")
            ep.append((S, (y_vel, x_vel), 100))
            break
        print("Want to make another move? [y/n]:")
        start = input()
        ep.append((S, (y_vel, x_vel), -1))
    print(ep)


if __name__ == '__main__':
    main()
