from source.simulation import Simulation
import sys

if __name__ == "__main__":
    sim = Simulation()
    sim.setup(sys.argv[1], sys.argv[2])
    sim.run()
