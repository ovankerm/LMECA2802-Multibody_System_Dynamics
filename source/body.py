import numpy as np
from source.data_classes import body_data
from copy import deepcopy


class Body:
    def __init__(self, data: body_data, joint_force: str = 'none', q0: float = 0., qd0: float = 0) -> None:
        """
        Class representing a body

        Parameters
        ----------
        data : body_data
            Instance of body_data containing the data linked to this body
        q0 : float, optional
            Initial value for the generalized coordinate linked to the joint

        Returns
        -------
        None
        """
        self.data = deepcopy(data)
        self.joint_force = joint_force
        self.q = q0
        self.qd = qd0
        self.Fext = np.zeros(2)
        self.Lext = 0

    def get_R(self) -> np.ndarray:
        """
        Gives the 2D rotation matrix

        Parameters
        ----------
        None

        Returns
        -------
        R : numpy array
            (2x2) numpy array representing the rotation matrix between the body and its parent
        """
        if self.data.joint_type == 'rev': return np.array([[np.cos(self.q), np.sin(self.q)], [-np.sin(self.q), np.cos(self.q)]])
        else: return np.eye(2)

    def get_R_inv(self) -> np.ndarray:
        """
        Gives the 2D rotation matrix

        Parameters
        ----------
        None

        Returns
        -------
        R : numpy array
            (2x2) numpy array representing the rotation matrix between the body and its parent
        """
        if self.data.joint_type == 'rev': return np.array([[np.cos(self.q), -np.sin(self.q)], [np.sin(self.q), np.cos(self.q)]])
        else: return np.eye(2)

    def get_z(self) -> np.ndarray:
        """
        Gives the z vector

        Parameters
        ----------
        None

        Returns
        -------
        z : numpy array
            numpy array representing the joint displacement
        """
        if self.data.joint_type == 'prix': return np.array([self.q, 0.])
        elif self.data.joint_type == 'priy': return np.array([0., self.q])
        else: return np.zeros(2)

    def get_z_dot(self) -> np.ndarray:
        """
        Gives the z dot vector

        Parameters
        ----------
        None

        Returns
        -------
        z_dot : numpy array
            numpy array representing the joint velocity
        """
        if self.data.joint_type == 'prix': return np.array([self.qd, 0.])
        elif self.data.joint_type == 'priy': return np.array([0., self.qd])
        else: return np.zeros(2)

    def get_diiz(self) -> np.ndarray:
        """
        Gives the diiz vector

        Parameters
        ----------
        None

        Returns
        -------
        diiz : numpy array
            numpy array representing the augmented CG position vector
        """
        return self.data.dii + self.get_z()

    def get_dikz(self, children_index) -> np.ndarray:
        """
        Gives the vectors of the children attachement points

        Parameters
        ----------
        None

        Returns
        -------
        dikz : numpy array
            numpy array of size (N, 2) containing the position of the attachement point of the N children
        """
        return self.data.dik[np.where(self.data.children == children_index)[0][0]] + self.get_z()

    def get_omega(self) -> float:
        """
        Gives the relative angular velocity vector

        Parameters
        ----------
        None

        Returns
        -------
        Omega : float
            relative angular velocity vector between the body and its parent
        """
        if self.data.joint_type == 'rev': return self.qd
        else: return 0.

    def get_phi(self) -> int:
        return int(self.data.joint_type == 'rev')

    def get_psi(self) -> np.ndarray:
        return np.array([self.data.joint_type == 'prix', self.data.joint_type == 'priy'], dtype=int)

    def get_children(self) -> np.ndarray:
        return self.data.children

    def get_m(self):
        return self.data.m

    def get_Iz(self):
        return self.data.Iz

    def get_Fext(self):
        return self.Fext

    def get_Lext(self):
        return self.Lext

    def get_joint_force(self):
        if self.joint_force == 'none':
            return 0.
        elif self.joint_force == 'beam':
            return -8.5e6 * self.q - 1e4 * self.qd
        elif self.joint_force == 'fixed':
            return -1e10 * self.q - 1e5 * self.qd

    def add_Fext(self, Fext):
        self.Fext += Fext

    def add_Lext(self, Lext):
        self.Lext += Lext

    def reset_Fext(self):
        self.Fext = np.zeros(2)

    def reset_Lext(self):
        self.Lext = 0





if __name__ == "__main__":
    body_d = body_data(joint_type='priy', dik=np.array([[1., 1.], [1., 5.]]))
    body = Body(body_d, q0=np.pi/4)

    print(body_d)
    print(body.get_dikz())
    print(body.get_omega())
