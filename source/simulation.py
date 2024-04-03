import argparse


class Simulation:
    def __init__(self):
        self.t = 0

    def setup(self, settings_file: str):
        filename = './settings/' + settings_file
        SETTINGS = ''
        with open(filename) as f:
            for line in f.readlines():
                SETTINGS += line.strip() + ' '

        parser = argparse.ArgumentParser()
        parser.add_argument('-g', dest='g', type=float)
        parser.add_argument('-N_bodies', dest='N_bodies', type=int)

        self.args = parser.parse_known_args(SETTINGS.split())[0]
        print(self.args.N_bodies)
