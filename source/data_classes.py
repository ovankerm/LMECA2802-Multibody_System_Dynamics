from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class body_data:
    m: float = 0.
    Iz: float = 0.
    dii: ArrayLike = field(default_factory=lambda: np.array([0., 0.]))
    joint_type: str = "pri"


@dataclass
class sim_data:
    g: float = -9.81

    N_bodies: int = 1
    inbody: ArrayLike = field(default_factory=lambda: np.zeros(1))

    t0: float = 0.
    tf: float = 1.


if __name__ == "__main__":
    data = sim_data()
    print(data)
    body = body_data()

    print(body)