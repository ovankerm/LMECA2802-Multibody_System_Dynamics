import numpy as np
from data_classes import body_data


class Body:
    def __init__(self, data: body_data, q0: float = 0., qd0: float = 0) -> None:
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
        self.data = data
        self.q = q0
        self.qd = qd0

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

    def get_dikz(self) -> np.ndarray:
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
        to_return = np.copy(self.data.dik)
        for i in to_return:
            i += self.get_z()
        return to_return

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

    def get_phi(self) -> float:
        return float(self.data.joint_type == 'rev')

    def get_psi(self) -> np.ndarray:
        return np.array([self.data.joint_type == 'prix', self.data.joint_type == 'priy'], dtype=float)



if __name__ == "__main__":
    body_d = body_data(joint_type='priy', dik=np.array([[1., 1.], [1., 5.]]))
    body = Body(body_d, q0=np.pi/4)

    print(body_d)
    print(body.get_dikz())
    print(body.get_omega())
