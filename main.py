from source.simulation import Simulation
import sys
import numpy as np

if __name__ == "__main__":
    sim = Simulation()
    sim.setup(sys.argv[1])
    sim.run()
