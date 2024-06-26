from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class body_data:
    """
    Data class containing all the data related to a body

    Parameters
    ----------
    m : float
        mass of the body
    Iz : float
        inertia of the body along the z axis
    dii : numpy array
        position of the body's CG wrt to the attachement point
    dik : numpy array
        numpy array of size (N, 2) containing the attachement point of the body's children
    joint_type : string
        gives the type of the joint linked to this body, the options are:
            - 'rev' for a revolute joint
            - 'prix' for a prismatic joint along the x-direction
            - 'priy' for a prismatic joint along the y-direction
    children : numpy array
        array containing the indices of the childen bodies
    """
    m: float = 0.
    Iz: float = 0.
    dii: ArrayLike = field(default_factory=lambda: np.array([0., 0.]))
    dik: ArrayLike = field(default_factory=lambda: [])
    joint_type: str = 'prix'
    children: ArrayLike = field(default_factory=lambda: np.array([]))
    dis: ArrayLike = field(default_factory=lambda: np.array([]))
    force_index: ArrayLike = field(default_factory=lambda: np.array([]))
