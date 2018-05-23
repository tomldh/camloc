"""
Investigation on Kabsch Algorithm
=================================

Autograd vs. Finite Difference Method

"""

import torch
import cv2
from math import cos, sin, pow, pi

import time
import matplotlib.pyplot as plt

########################################
# Data class
# ----------
#
# This is a class that supports generation of 3D scene point data (X) and
# its corresponding transformed data (P) by a sequence of rotations and
# translations.
#


class SceneDataset():
    """Scene dataset"""

    def __init__(self, numPt=4):
        self.X = torch.ones([3, numPt], dtype=torch.float32)
        self.R = torch.eye(3, dtype=torch.float32)
        self.P = []


##########################################
# Transform class
# ---------------
#
# This class supports routines related to 3D data transformations, such as
# rotation and translation.
#

class Transformation():
    """Transformation"""

    def __init__(self, rep=0,
                 rots=None,
                 trans=None):
        """
        Args:
            rep: rotation representation
                0 - matrix
                1 - axis-angle
            rots: user-defined rotation, [angleX, angleY, angleZ]
                rep=0: 0 <= angle <= 2*PI
                rep=1: -PI <= angle <= PI
            trans: user-defined translation
        """
        self.rep = int(rep)

        if self.rep == 0:
            if rots is None:
                rots = torch.rand(3, dtype=torch.float32) * 2 * pi

            # FIXME: use opencv functions?
            rotX = torch.FloatTensor(
                [[1, 0, 0],
                 [0, cos(rots[0]), -sin(rots[0])],
                 [0, sin(rots[0]), cos(rots[0])]])

            rotY = torch.FloatTensor(
                [[cos(rots[1]), 0, -sin(rots[1])],
                 [0, 1, 0],
                 [sin(rots[1]), 0, cos(rots[1])]])

            rotZ = torch.FloatTensor(
                [[cos(rots[2]), -sin(rots[2]), 0],
                 [sin(rots[2]), cos(rots[2]), 0],
                 [0, 0, 1]])

            self.R = torch.mm(rotZ, torch.mm(rotY, rotX))

        elif self.rep == 1:
            if rots is None:
                rots = torch.rand(3, dtype=torch.float32) * 2 * pi - pi

            self.R = rots.clone()

        else:
            print('Error: undefined rotation representation.')

        self.T = trans

    def getRotation(self, rep=None):
        if rep is None or self.rep == rep:
            return self.R

        if self.rep == 0 and rep == 1:
            return torch.from_numpy(cv2.Rodrigues(self.R.numpy())[0])

        elif self.rep == 1 and rep == 0:
            return torch.from_numpy(cv2.Rodrigues(self.R.numpy())[0])

    def getTranslation(self):
        return self.T

    def getRepresentation(self):
        return self.rep

    def setRotation(self, another):
        self.R = another

    def setRep(self, another):
        self.rep = another

    def compareTranslation(self, another):
        """
        calculates averaged element-wise square of differences
        """
        if torch.numel(self.T) != torch.numel(another):
            print('Error in comparison: number of elements does not match.')
            return -1

        return torch.sum(torch.pow((self.T-another), 2), 0) / torch.numel(self.T)

    def compareRotation(self, another):
        """
        calculates averaged element-wise square of differences
        """
        if torch.numel(self.R) != torch.numel(another):
            print('Error in comparison: number of elements does not match.')
            return -1

        if self.rep == 0:
            return torch.sum(torch.sum(torch.pow((self.R-another), 2), 1), 0) / torch.numel(self.R)

        elif self.rep == 1:
            return torch.sum(torch.pow((self.R-another), 2), 0) / torch.numel(self.R)

        return -1


########################################################
# Custom autograd for cv2.Rodrigues()
#
class Rodrigues(torch.autograd.Function):
    """
    This class implements the forward and backward passes for Rodrigues fcn
    """

    @staticmethod
    def forward(ctx, input, use_cuda):
        if use_cuda:
            input = input.cpu()
        r, jac = cv2.Rodrigues(input.detach().numpy())
        r = torch.from_numpy(r)
        jac = torch.from_numpy(jac)
        if use_cuda:
            input = input.cuda()
            # r = r.cuda()
            jac = jac.cuda()
        ctx.save_for_backward(input, jac)
        return r

    def backward(ctx, grad_output):
        # print("grad_output size: {}".format(grad_output.size()))
        input, jac = ctx.saved_tensors
        # print("input size: {}".format(input.size()))
        # print(input)
        # print("jac size: {}".format(jac.size()))
        # print(jac)
        # FIXME: determine grad_input size dynamically
        grad_input = torch.mm(jac, grad_output).view(3, 3)

        return grad_input, None


########################################################
# Function that performs kabsch algorithm and differentitation using autograd.
#
def kabsch_autograd(P, X, jacobian=None, use_cuda=False):
    """
    Args:
        calc_grad(bool): controls whether to use autograd

    Return:
        R (tensor): 3x3 rotation that satisfies P = RX
        jacobian (tensor) 9x3N jacobian matrix of R w.r.t scene point coord.
    """
    print("Running Kabsch (Autograd)...")

    if use_cuda:
        X = X.cuda()
        P = P.cuda()

    if jacobian is None:
        R, t = kabsch(P, X)
        return R

    print("\tComputing jacobian...")
    X.requires_grad = True

    tx = torch.mean(X, 0)
    tp = torch.mean(P, 0)
    Xc = X.sub(tx+0.1)
    Pc = P.sub(tp+0.1)

    A = torch.mm(torch.t(Pc), Xc)

    U, S, V = torch.svd(A)

    Vt = torch.t(V)

    d = torch.det(torch.mm(U, Vt))

    D = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    if use_cuda:
        D = D.cuda()

    R = torch.mm(U, torch.mm(D, Vt))

    rod = Rodrigues.apply

    r = rod(R, use_cuda)

    numelR = torch.numel(r)

    for i in range(numelR):
        onehot = torch.zeros(numelR, dtype=torch.float32)
        onehot[i] = 1

        if use_cuda:
            r.backward(onehot.view(r.size()).cuda(), retain_graph=True)
        else:
            r.backward(onehot.view(r.size()), retain_graph=True)
        jacobian[i, :] = X.grad.data.view(-1)

        X.grad.data.zero_()

        if use_cuda:
            tx.backward(onehot.view(tx.size()).cuda(), retain_graph=True)
        else:
            tx.backward(onehot.view(tx.size()), retain_graph=True)
        jacobian[i+3, :] = X.grad.data.view(-1)

        X.grad.data.zero_()

    return r, tx


########################################################
# Function that performs kabsch algorithm and differentiation using
# using central finite differences.
# Returns:
# - estimated rotation tensor
# - estimated translation tensor
# - jacobian matrix of rotation w.r.t each data point coordinate
# - jacobian matrix of translation w.r.t each data point coordinate
#
def kabsch_fd(P, X, eps=0.1, jacobian=None):
    """
    Args:
        P (tensor): Measurements
        X (tensor): Scene points
        eps (float): Epsilon used in finite difference approximation
        calc_grad (bool): Flag to indicate whether to calculate jacobian

    Return:
        R (tensor): 3x3 rotation that satisfies P = RX
        jacobian (tensor) 9x3N jacobian matrix of R w.r.t scene point coord.
    """
    print("Running Kabsch (Finite Difference)...")

    if P.size() != X.size():
        print("\tError: Dimension of P and X is not the same.")

    if X.size()[1] != 3:
        print("\tError: Expected 3D coordinates, but got {}.".format(
            X.size()[0]))

    R, t = kabsch(P, X)

    # Return rotation matrix only if no gradient is required
    if jacobian is None:
        return R, t

    print("\tComputing jacobian...")
    # calculates partial derivatives
    # create data containers
    # FIXME: use axis-angle

    for i in range(X.size()[0]):
        for j in range(3):

            X[i, j] += eps
            fwdR, fwdT = kabsch(P, X)

            X[i, j] -= 2*eps
            bwdR, bwdT = kabsch(P, X)

            X[i, j] += eps

            diffR = (fwdR-bwdR)/(2*eps)

            diffT = (fwdT-bwdT)/(2*eps)

            # place derivatives to column (w.r.t X[i, j]) in jacobian matrix
            jacobian[:3, i*3+j] = diffR.view(-1)
            jacobian[3:, i*3+j] = diffT.view(-1)

    return R, t


########################################################
# Helper functions
#
def kabsch(P, X):

    tx = torch.mean(X, 0)
    tp = torch.mean(P, 0)

    Xc = X.sub(tx+0.1)
    Pc = P.sub(tp+0.1)

    A = torch.mm(torch.t(Pc), Xc)

    U, S, V = torch.svd(A)

    Vt = torch.t(V)

    d = torch.det(torch.mm(U, Vt))

    D = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    R = torch.mm(U, torch.mm(D, Vt))

    return torch.from_numpy(cv2.Rodrigues(R.numpy())[0]), tx


def createData(N, seed=-1, transform=None):
    """create 3D scene point data

    Args:
        N - number of points
        seed - seed for random value generation. If seed<0, use default points.

    Return:
        data - Nx3 scene point matrix
        tfData - Nx3 transformed data as fake measurement
    """
    data = torch.empty(N, 3, dtype=torch.float32)
    tfData = torch.empty(N, 3, dtype=torch.float32)

    if seed < 0:
        data = torch.FloatTensor(
            [[1., 0., 0.],
             [0., 0., 0.],
             [0., 1., 0.],
             [1., 1., 0.]
             ])
    else:
        torch.manual_seed(seed)

        data[:, 0] = torch.randn(N, dtype=torch.float32)
        data[:, 1] = torch.randn(N, dtype=torch.float32)
        data[:, 2] = torch.randn(N, dtype=torch.float32)

    if transform is not None:
        tfData = torch.t(torch.mm(transform.getRotation(rep=0),
                                  torch.t(data+transform.getTranslation())))
    else:
        tfData = data.clone()

    return data, tfData


def compareJacobian(A, B):

    return torch.sum(torch.sum(torch.pow((A-B), 2), 1), 0) / torch.numel(A)


########################################################
# Main routine to run the analysis
#
if __name__ == '__main__':

    # list for statistics
    # powers = [2, 4, 6, 8, 10, 12]
    powers = [2]
    N = []
    fd_time = []
    ag_time = []
    ag_cuda_time = []

    if torch.cuda.is_available():
        use_cuda = True

    # create known a rotation and translation
    tf = Transformation(trans=torch.FloatTensor([10., 2., 0.]))
    tf.setRotation(tf.getRotation(rep=1))
    tf.setRep(1)
    print("Rotation matrix:\n{}".format(tf.getRotation(rep=0)))
    print("Rotation vector:\n{}".format(tf.getRotation(rep=1)))
    print("Translation:\n{}".format(tf.getTranslation()))

    for k, p in enumerate(powers):
        N.append(int(pow(2, p)))
        # create sample scene point and correspondences
        # scenePts, measurePts = createData(4, transform=tf)
        scenePts, measurePts = createData(N[k], seed=10, transform=tf)
        # print("Samples:\n{}".format(scenePts))
        # print("Measurements:\n{}".format(measurePts))

        fdJac = torch.zeros([6, torch.numel(scenePts)], dtype=torch.float32)
        agJac = torch.zeros([6, torch.numel(scenePts)], dtype=torch.float32)
        agJac_cuda = torch.zeros(
            [6, torch.numel(scenePts)], dtype=torch.float32)

        # finite differences
        fd_begin = time.time()
        fdRot, fdT = kabsch_fd(measurePts, scenePts.clone(), jacobian=fdJac)
        fd_time.append(time.time()-fd_begin)
        # print(fdRot)
        # print(fdT)
        print(fdJac)

        # autograd
        ag_begin = time.time()
        agRot, agT = kabsch_autograd(
            measurePts, scenePts.clone(), agJac, False)
        ag_time.append(time.time()-ag_begin)
        # print(agRot)
        # print(agT)
        print(agJac)

        agc_begin = time.time()
        agRot_cuda, agT_cuda = kabsch_autograd(
            measurePts, scenePts.clone(), agJac_cuda, True)
        ag_cuda_time.append(time.time()-agc_begin)
        # print(agRot_cuda)
        # print(agT_cuda)
        print(agJac_cuda)

        # statistics
        print("Summary of analysis:")
        print("Number of scene points: {}".format(N[k]))
        print("Finite difference used {0:.6f} sec.".format(fd_time[k]))
        print("Autograd used {0:.6f} sec.".format(ag_time[k]))
        print("Autograd(cuda) used {0:.6f} sec.".format(ag_cuda_time[k]))

        print("Difference of rotation matrices:")
        print("\tfinite difference: {}".format(tf.compareRotation(fdRot)))
        print("\tautograd_cpu: {}".format(tf.compareRotation(agRot)))
        print("\tautograd_cuda: {}".format(
            tf.compareRotation(agRot_cuda.cpu())))

        print("Difference of translation vectors:")
        print("\tfinite difference: {}".format(tf.compareTranslation(fdT)))
        print("\tautograd_cpu: {}".format(tf.compareTranslation(agT)))
        print("\tautograd_cuda: {}".format(
            tf.compareTranslation(agT_cuda.cpu())))

        print("Difference of jacobian matrices fdJac and agJac: {}".format(
            compareJacobian(fdJac, agJac)))
        print("Difference of jacobian matrices fdJac and agJac_cuda: {}".format(
            compareJacobian(fdJac, agJac_cuda)))
        print("========================================\n")

    # print(N)
    # print(fd_time)
    # print(ag_time)
    # print(ag_cuda_time)
    #
    # plt.figure()
    # plt.loglog(N, fd_time, marker="x", color="r", label="finite difference")
    # plt.loglog(N, ag_time, marker="x", color="b", label="autograd CPU")
    # plt.loglog(N, ag_cuda_time, marker="x", color="g", label="autograd GPU")
    # plt.xlabel("N")
    # plt.ylabel("Runtime/sec")
    # plt.legend(loc=0)
    # plt.show()
