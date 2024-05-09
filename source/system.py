from source.body import Body
from source.data_classes import body_data
import numpy as np
from numpy.linalg import solve

class System:
    def __init__(self, N_bodies, g=np.array([0, -9.81])):
        self.N_bodies = N_bodies
        self.bodies = np.empty(N_bodies + 1, dtype=Body)

        base = body_data()
        self.bodies[0] = Body(base)
        self.first_index = 1

        self.inbody = np.zeros(N_bodies + 1, dtype=int)
        self.q = np.zeros(N_bodies, dtype=float)
        self.qd = np.zeros(N_bodies, dtype=float)
        self.Q = np.zeros(self.N_bodies)

        # arrays for the forwards loop
        self.omega = np.zeros(N_bodies + 1, dtype=float)
        self.omega_c_dot = np.zeros(N_bodies + 1, dtype=float)
        self.beta_c = np.zeros((N_bodies + 1, 2, 2), dtype=float)
        self.alpha_c = np.zeros((N_bodies + 1, 2), dtype=float)
        self.alpha_c[0] = -g
        self.O_M = np.zeros((N_bodies + 1, N_bodies + 1), dtype=float)
        self.A_M = np.zeros((N_bodies + 1, N_bodies + 1, 2), dtype=float)

        # arrays for the backwards loop
        self.F_c = np.zeros((N_bodies + 1, 2), dtype=float)
        self.L_c = np.zeros(N_bodies + 1, dtype=float)
        self.F_M = np.zeros((N_bodies + 1, N_bodies + 1, 2), dtype=float)
        self.L_M = np.zeros((N_bodies + 1, N_bodies + 1), dtype=float)

        self.M = np.zeros((N_bodies + 1, N_bodies + 1), dtype=float)
        self.c = np.zeros(N_bodies + 1, dtype=float)

    def push_q(self):
        for i in range(self.N_bodies):
            self.bodies[i+1].q = self.q[i]
            self.bodies[i+1].qd = self.qd[i]

    def make_beam(self, N_sections: int, parent: int, length: float = 2., mass: float = 50., Iz: float = 50., fixed: bool = False, dik: np.ndarray = np.array([0, 0])):
        if N_sections % 2 != 1:
            raise ValueError('N_sections should be odd')
        if self.first_index + N_sections + 2 > self.N_bodies + 1:
            raise MemoryError('Not enough storage to create the beam')

        joint_forces = 'fixed' if fixed else 'none'

        # add the correct data to the parent body
        self.bodies[parent].data.dik = np.append(self.bodies[parent].data.dik, dik)
        self.bodies[parent].data.children = np.append(self.bodies[parent].data.children, int(self.first_index))

        # make the T1 joint
        prix_data = body_data(joint_type='prix', dii=np.array([0, 0]), dik=np.array([[0, 0]]), children=np.array([self.first_index+1]))
        self.bodies[self.first_index] = Body(prix_data, joint_force=joint_forces)
        self.inbody[self.first_index] = parent
        posx_ind = self.first_index - 1
        self.first_index += 1

        # make the T2 joint
        priy_data = body_data(joint_type='priy', dii=np.array([0, 0]), dik=np.array([[0, 0]]), children=np.array([self.first_index+1]))
        self.bodies[self.first_index] = Body(priy_data, joint_force=joint_forces)
        self.inbody[self.first_index] = self.first_index - 1
        posy_ind = self.first_index - 1
        self.first_index += 1

        half_length = N_sections//2
        # make the center body
        c_body_data = body_data(m=mass, Iz=Iz, joint_type='rev', dii=np.array([0, 0]), dik=np.array([[-length/2, 0], [length/2, 0]]), children=np.array([self.first_index+1, self.first_index+1+half_length]))
        self.bodies[self.first_index] = Body(c_body_data, joint_force=joint_forces)
        self.inbody[self.first_index] = self.first_index - 1
        angle_ind = self.first_index - 1

        l_body_data = body_data(m=mass, Iz=Iz, joint_type='rev', dii=np.array([-length/2, 0]), dik=np.array([[-length, 0]]), children=np.array([self.first_index+2]))
        self.bodies[self.first_index + 1] = Body(l_body_data, joint_force='beam')
        self.inbody[self.first_index + 1] = self.first_index

        r_body_data = body_data(m=mass, Iz=Iz, joint_type='rev', dii=np.array([length/2, 0]), dik=np.array([[length, 0]]), children=np.array([self.first_index+2+half_length]))
        self.bodies[self.first_index + 1 + half_length] = Body(r_body_data, joint_force='beam')
        self.inbody[self.first_index + 1 + half_length] = self.first_index

        for i in range(1, half_length - 1):
            l_body_data = body_data(m=mass, Iz=Iz, joint_type='rev', dii=np.array([-length/2, 0]), dik=np.array([[-length, 0]]), children=np.array([self.first_index+2+i]))
            self.bodies[self.first_index + 1 + i] = Body(l_body_data, joint_force='beam')
            self.inbody[self.first_index + 1 + i] = self.first_index + i

            r_body_data = body_data(m=mass, Iz=Iz, joint_type='rev', dii=np.array([length/2, 0]), dik=np.array([[length, 0]]), children=np.array([self.first_index+2+i+half_length]))
            self.bodies[self.first_index + 1 + half_length + i] = Body(r_body_data, joint_force='beam')
            self.inbody[self.first_index + 1 + half_length + i] = self.first_index + half_length + i

        l_body_data = body_data(m=mass, Iz=Iz, joint_type='rev', dii=np.array([-length/2, 0]))
        self.bodies[self.first_index + half_length] = Body(l_body_data, joint_force='beam')
        self.inbody[self.first_index + half_length] = self.first_index + half_length - 1

        r_body_data = body_data(m=mass, Iz=Iz, joint_type='rev', dii=np.array([length/2, 0]))
        self.bodies[self.first_index + 2 * half_length] = Body(r_body_data, joint_force='beam')
        self.inbody[self.first_index + 2 * half_length] = self.first_index + 2 * half_length - 1

        self.first_index += N_sections

        return posx_ind, posy_ind, angle_ind

    def make_bodies(self):
        base = body_data(dik=np.array([0, 0]), children=np.array([1]))
        self.bodies[0] = Body(base)
        """
        body_d = body_data(joint_type='rev', dii=np.array([0, l/2]), m=m, Iz=m*l*l/12)
        self.bodies[1] = Body(body_d)
        self.inbody[1] = 0
        """
        body_ = body_data(joint_type='prix', dii=np.array([0, 0]), dik=np.array([[0, 0]]), children=np.array([2]))
        self.bodies[1] = Body(body_)
        self.inbody[1] = 0
        body_d = body_data(joint_type='rev', dii=np.array([0, 0]), m=50, Iz=50, dik=np.array([[2, 0]]), children=np.array([3]))
        self.bodies[2] = Body(body_d)
        self.inbody[2] = 1
        body_dd = body_data(joint_type='rev', dii=np.array([1, 0]), m=50, Iz=50, dik=np.array([[2, 0]]), children=np.array([4]))
        self.bodies[3] = Body(body_dd)
        self.inbody[3] = 2
        body_ddd = body_data(joint_type='rev', dii=np.array([1, 0]), m=50, Iz=50, dik=np.array([[2, 0]]), children=np.array([5]))
        self.bodies[4] = Body(body_ddd)
        self.inbody[4] = 3
        body_dddd = body_data(joint_type='rev', dii=np.array([1, 0]), m=50, Iz=50, dik=np.array([[2, 0]]), children=np.array([6]))
        self.bodies[5] = Body(body_dddd)
        self.inbody[5] = 4
        body_ddddd = body_data(joint_type='rev', dii=np.array([1, 0]), m=50, Iz=50)
        self.bodies[6] = Body(body_ddddd)
        self.inbody[6] = 5

    def get_qdd(self):
        for i in range(1, self.N_bodies + 1):
            h = self.inbody[i]
            self.omega[i] = self.omega[h] + self.bodies[i].get_phi() * self.bodies[i].qd
            self.omega_c_dot[i] = self.omega_c_dot[h]
            self.beta_c[i] = np.array([[-self.omega[i]*self.omega[i], -self.omega_c_dot[i]],
                                       [self.omega_c_dot[i], -self.omega[i]*self.omega[i]]])
            psi_i = self.bodies[i].get_psi()
            dhi = self.bodies[h].get_dikz(i)
            self.alpha_c[i] = self.bodies[i].get_R() @ (self.alpha_c[h] + self.beta_c[h] @ dhi) + 2 * self.bodies[i].qd * self.omega[i] * np.array([-psi_i[1], psi_i[0]])
            for k in range(1, i+1):
                self.O_M[i, k] = self.O_M[h, k] + (k == i) * self.bodies[i].get_phi()
                self.A_M[i, k] = self.bodies[i].get_R() @ (self.A_M[h, k] + self.O_M[h, k] * np.array([-dhi[1], dhi[0]])) + (k == i) * self.bodies[i].get_psi()

        for i in range(self.N_bodies, 0, -1):
            W_c = self.bodies[i].get_m() * (self.alpha_c[i] + self.beta_c[i] @ self.bodies[i].get_diiz()) - self.bodies[i].get_Fext()
            F_c = np.zeros(2)
            L_c = 0
            for j in self.bodies[i].get_children():
                temp = self.bodies[j].get_R() @ self.F_c[j]
                dijz = self.bodies[i].get_dikz(j)
                F_c += temp
                L_c += self.L_c[j] + np.cross(dijz, temp)

            self.F_c[i] = F_c + W_c
            self.L_c[i] = L_c + np.cross(self.bodies[i].get_diiz(), W_c) - self.bodies[i].get_Lext() + self.bodies[i].get_Iz() * self.omega_c_dot[i]

            for k in range(1, i+1):
                W_M = self.bodies[i].get_m() * (self.A_M[i, k] + self.O_M[i, k] * self.bodies[i].get_diiz())
                F_M = np.zeros(2)
                L_M = 0
                for j in self.bodies[i].get_children():
                    temp = self.bodies[j].get_R() @ self.F_M[j, k]
                    dijz = self.bodies[i].get_dikz(j)
                    F_M += temp
                    L_M += self.L_M[j, k] + np.cross(dijz, temp)

                self.F_M[i, k] = F_M + W_M
                self.L_M[i, k] = L_M + np.cross(self.bodies[i].get_diiz(), W_M) + self.bodies[i].get_Iz() * self.O_M[i, k]

        for i in range(1, self.N_bodies + 1):
            self.c[i] = np.dot(self.bodies[i].get_psi(), self.F_c[i]) + self.bodies[i].get_phi() * self.L_c[i]
            for j in range(1, i + 1):
                self.M[i, j] = np.dot(self.bodies[i].get_psi(), self.F_M[i, j]) + self.bodies[i].get_phi() * self.L_M[i, j]

        for i in range(self.N_bodies):
            self.Q[i] = self.bodies[i+1].get_joint_force()

        return solve(self.M[1:, 1:], -self.c[1:] + self.Q)

    def RHS(self, t, y):
        print(t)
        self.q = np.copy(y[:self.N_bodies])
        self.qd = np.copy(y[self.N_bodies:])

        self.push_q()

        to_return = np.zeros(2 * self.N_bodies, dtype=float)
        to_return[:self.N_bodies] = np.copy(self.qd)
        qdd = self.get_qdd()
        # print(f't = %.4f, qdd = {qdd}'%(t))
        to_return[self.N_bodies:] = qdd

        return to_return


if __name__ == "__main__":
    sys = System(10)

    print(sys.bodies)
