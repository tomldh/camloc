"""
Investigation on Kabsch Algorithm
=================================

Autograd vs. Finite Difference Method

"""

import torch
import numpy as np
import cv2
from math import cos, sin, pow, pi

import time
import matplotlib.pyplot as plt
import utility as utl

# file names
FNAME_ACCURACY = "test_accuray.txt"
FNAME_RUNTIME = "test_runtime.txt"
FNAME_DATA = "problem_data.txt"

##########################################
# Transform class
# ---------------
#
# This class supports routines related to 3D data transformations, such as
# rotation and translation.
#


class Transformation():
    """Transformation"""

    def __init__(self,
                 rots=None,
                 trans=None):
        """
        Args:
            rep: rotation representation
                0 - matrix
                1 - axis-angle
            rots: user-defined rotational angles, [angleX, angleY, angleZ]
                rep=0: 0 <= angle <= 2*PI
                rep=1: -PI <= angle <= PI
            trans: user-defined translation
        """

        if rots is None:
            self.rots = torch.rand(3, dtype=torch.float32) * 2 * pi
        else:
            self.rots = rots

        if trans is None:
            self.T = torch.randn(3, dtype=torch.float32)
        else:
            self.T = trans

        self.R = cv2.Rodrigues(self.__computeRotationMatrix__())[0]

    def __computeRotationMatrix__(self):

        rotX = np.array(
            [[1, 0, 0],
             [0, cos(self.rots[0]), -sin(self.rots[0])],
             [0, sin(self.rots[0]), cos(self.rots[0])]])

        rotY = np.array(
            [[cos(self.rots[1]), 0, -sin(self.rots[1])],
             [0, 1, 0],
             [sin(self.rots[1]), 0, cos(self.rots[1])]])

        rotZ = np.array(
            [[cos(self.rots[2]), -sin(self.rots[2]), 0],
             [sin(self.rots[2]), cos(self.rots[2]), 0],
             [0, 0, 1]])

        return np.dot(rotZ, np.dot(rotY, rotX))

    def getRotationAngles(self):
        return self.rots

    def getRotationVector(self):
        return torch.from_numpy(self.R).float()

    def getRotationMatrix(self):
        return torch.from_numpy(cv2.Rodrigues(self.R)[0]).float()

    def getTranslation(self):
        return self.T

    def setRotation(self, rots):
        self.rots = rots
        self.R = cv2.Rodrigues(self.__computeRotationMatrix__())[0]

    def setTranslation(self, trans):
        self.T = trans


########################################################
# Custom autograd for cv2.Rodrigues()
#
class Rodrigues(torch.autograd.Function):
    """
    This class implements the forward and backward passes for Rodrigues fcn
    """

    @staticmethod
    def forward(ctx, input):
        # keep track of original tensor type
        isCuda = input.is_cuda

        # cuda tensor will be converted to cpu tensor
        # no change if tensor is already cpu
        input = input.cpu()

        r, jac = cv2.Rodrigues(input.detach().numpy())
        r = torch.from_numpy(r)  # convert numpy to tensor
        jac = torch.from_numpy(jac)  # convert numpy to tensor

        # convert back to the original tensor type
        if isCuda:
            input = input.cuda()
            r = r.cuda()
            jac = jac.cuda()

        ctx.save_for_backward(input, jac)

        return r

    @staticmethod
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
        r, t = kabsch(P, X)
        return r, t

    print("\tComputing jacobian...")

    X.requires_grad = True
    # X.register_hook(utl.Hook_X)

    # tf = Transformation(rots=torch.FloatTensor([0.15, 0.15, 0.15]), trans=torch.FloatTensor([0., 0., 0.]))

    tx = torch.mean(X, 0)
    tp = torch.mean(P, 0)

    t = tp - tx  # translation vector
    print(tx)
    print(tp)

    Xc = X.sub(tx)
    Pc = P.sub(tp)

    print("Xc")
    print(Xc)

    Xc.register_hook(utl.Hook_Xc)

    # Pcp = torch.t(torch.mm(tf.getRotationMatrix(), torch.t(Pc+tf.getTranslation())))
    # Xcp = torch.t(torch.mm(tf.getRotationMatrix(), torch.t(Xc+tf.getTranslation())))
    # print("Xcp")
    # print(Xcp)
    A = torch.mm(torch.t(Pc), Xc)
    print("A:")
    print(A)
    A.register_hook(utl.Hook_A)
    U, S, V = torch.svd(A)
    # S.register_hook(utl.Hook_S)
    U.register_hook(utl.Hook_U)
    V.register_hook(utl.Hook_V)
    print("U:")
    print(U)
    print("S:")
    print(S)
    print("V")
    print(V)
    # numelU = torch.numel(A)
    # jacU = torch.zeros([9,12], dtype=torch.float)
    # for i in range(numelU):
    #     print("Gradient w.r.t. U element {}".format(i))
    #     onehot = torch.zeros(numelU, dtype=torch.float32)
    #     onehot[i] = 1
    #
    #     if use_cuda:
    #         U.backward(onehot.view(U.size()).cuda(), retain_graph=True)
    #     else:
    #         U.backward(onehot.view(U.size()), retain_graph=True)
    #     jacU[i, :] = X.grad.data.view(-1)
    #     X.grad.data.zero_()
    # print("Autograd Jacobian U:")
    # print(jacU)

    Vt = torch.t(V)
    # Vt.register_hook(utl.Hook_Vt)
    d = torch.det(torch.mm(U, Vt))
    # d.register_hook(utl.Hook_d)
    D = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, d]])
    # D.register_hook(utl.Hook_D)
    if use_cuda:
        D = D.cuda()

    R = torch.mm(U, torch.mm(D, Vt))
    # R.register_hook(utl.Hook_R)
    rod = Rodrigues.apply

    r = rod(R)  # rotation vector

    numelR = torch.numel(r)

    for i in range(numelR):
        print("Gradient w.r.t. element {}".format(i))
        onehot = torch.zeros(numelR, dtype=torch.float32)
        onehot[i] = 1

        if use_cuda:
            r.backward(onehot.view(r.size()).cuda(), retain_graph=True)
        else:
            r.backward(onehot.view(r.size()), retain_graph=True)
        jacobian[i, :] = X.grad.data.view(-1)

        X.grad.data.zero_()

        if use_cuda:
            t.backward(onehot.view(t.size()).cuda(), retain_graph=True)
        else:
            t.backward(onehot.view(t.size()), retain_graph=True)
        jacobian[i+3, :] = X.grad.data.view(-1)

        X.grad.data.zero_()

    return r, t


########################################################
# Function that performs kabsch algorithm and differentiation using
# using central finite differences.
# Returns:
# - estimated rotation tensor
# - estimated translation tensor
# - jacobian matrix of rotation w.r.t each data point coordinate
# - jacobian matrix of translation w.r.t each data point coordinate
#
def kabsch_fd(P, X, jacobian=None, eps=0.01):
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
            X.size()[1]))

    r, t, U, A = kabsch(P, X)

    # Return rotation matrix only if no gradient is required
    if jacobian is None:
        return r, t

    print("\tComputing jacobian...")

    jacU = torch.zeros([9, 12], dtype=torch.float32)
    jacA = torch.zeros([9, 12], dtype=torch.float32)

    for i in range(X.size()[0]):
        for j in range(3):

            # forward step
            X[i, j] += eps
            fwdR, fwdT, fwdU, fwdA = kabsch(P, X)

            # backward step
            X[i, j] -= 2*eps
            bwdR, bwdT, bwdU, bwdA = kabsch(P, X)

            # return to original
            X[i, j] += eps

            diffR = (fwdR-bwdR)/(2*eps)
            diffT = (fwdT-bwdT)/(2*eps)

            # diffU = (fwdU-bwdU) / (2*eps)
            # diffA = (fwdA-bwdA) / (2*eps)
            #
            # jacU[:, i*3+j] = diffU.view(-1)
            # jacA[:, i*3+j] = diffA.view(-1)
            # place derivatives to column (w.r.t X[i, j]) in jacobian matrix
            jacobian[:3, i*3+j] = diffR.view(-1)
            jacobian[3:, i*3+j] = diffT.view(-1)

    # print("Jacobian U:")
    # print(jacU)
    # print("Jacobian A:")
    # print(jacA)

    return r, t


########################################################
# Function that performs kabsch algorithm with autograd and finite differences
# Non-degenerate case: autograd
# degenerate case: finite difference
#
def kabsch_stable(P, X, jacobian=None, use_cuda=False, eps=0.1):
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

    print("Running Kabsch (Stable)...")

    if use_cuda:
        X = X.cuda()
        P = P.cuda()

    if jacobian is None:
        r, t = kabsch(P, X)
        return r, t

    print("\tComputing jacobian...")

    X.requires_grad = True

    # compute centroid as average of coordinates
    tx = torch.mean(X, 0)
    tp = torch.mean(P, 0)

    t = tp - tx  # translation vector

    # move centroid to origin
    Xc = X.sub(tx)
    Pc = P.sub(tp)

    A = torch.mm(torch.t(Pc), Xc)

    U, S, V = torch.svd(A)
    # print("S")
    # print(S)

    # flag for degeneracy
    degenerate = False

    # degenerate if any singular value is zero
    if torch.numel(torch.nonzero(S)) != torch.numel(S):
        # print(torch.nonzero(S).size())
        # print(S.size())
        # print("zero singular")
        degenerate = True

    # degenerate if singular values are not distinct
    if torch.abs(S[0]-S[1]) < 1e-8 or torch.abs(S[0]-S[2]) < 1e-8 or torch.abs(S[1]-S[2]) < 1e-8:
        # print("non distinct singular")
        degenerate = True

    # if degenerate, resort back to finite difference for stability
    if degenerate is True:
        X.requires_grad = False
        return None, None

    # non-degenerate, continue with kabsch algorithm with autograd
    Vt = torch.t(V)

    d = torch.det(torch.mm(U, Vt))

    D = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    R = torch.mm(U, torch.mm(D, Vt))

    rod = Rodrigues.apply

    r = rod(R)  # rotation vector

    numelR = torch.numel(r)

    # compute jacobian matrix
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
            t.backward(onehot.view(t.size()).cuda(), retain_graph=True)
        else:
            t.backward(onehot.view(t.size()), retain_graph=True)
        jacobian[i+3, :] = X.grad.data.view(-1)

        X.grad.data.zero_()

    return r, t


########################################################
# Helper functions
#
def kabsch(P, X):
    # tf = Transformation(rots=torch.FloatTensor([0.1, 0.1, 0.1]), trans=torch.FloatTensor([0., 0., 0.]))
    #
    # Xcp = torch.t(torch.mm(tf.getRotation(rep=0), torch.t(X)))
    # Pcp = torch.t(torch.mm(tf.getRotation(rep=0), torch.t(P)))

    tx = torch.mean(X, 0)
    tp = torch.mean(P, 0)

    t = tp - tx  # translation vector

    # print(tx)
    # print(tp)

    Xc = X.sub(tx)
    Pc = P.sub(tp)

    A = torch.mm(torch.t(Pc), Xc)

    U, S, V = torch.svd(A)

    Vt = torch.t(V)

    d = torch.det(torch.mm(U, Vt))

    D = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    R = torch.mm(U, torch.mm(D, Vt))

    r = torch.from_numpy(cv2.Rodrigues(R.numpy())[0])  # rotation vector

    return r, t, U, A


def createScene(N, seed=None):
    """create 3D scene point data

    Args:
        N - number of points
        seed - seed for random value generation

    Return:
        data - Nx3 scene point matrix
        tfData - Nx3 transformed data used as measurement
    """
    data = torch.empty(N, 3, dtype=torch.float32)

    if seed is None:
        data = torch.FloatTensor(
            [[1., 0., 0.],
             [0., 0., 0.],
             [0., 1., 0.],
             [1., 1., 0.]
             ])
        # data = torch.FloatTensor(
        #     [[0.5, -0.5, 0.],
        #      [-0.5, -0.5, 0.],
        #      [-0.5, 0.5, 0.],
        #      [0.5, 0.5, 0.]
        #      ])
    else:
        if seed > 0:
            torch.manual_seed(seed)

        data[:, 0] = torch.randn(N, dtype=torch.float32)
        data[:, 1] = torch.randn(N, dtype=torch.float32)
        data[:, 2] = torch.randn(N, dtype=torch.float32)

    return data


def createMeasurements(data, transform, fixed=False):
    """Given scene points, create measurements by some transformations

    Args:
    data - scene points
    transform - class object that contains rotation and translation
    fixed - if true, existing transformation will be overriden by random transformation. If false, use existing transformation.

    """

    assert data is not None
    assert transform is not None
    assert data.size()[1] == 3

    if fixed is False:
        # set a new transformation
        transform.setRotation(torch.rand(3, dtype=torch.float32) * 2 * pi)
        transform.setTranslation(torch.randn(3, dtype=torch.float32))

    # P = RX + T
    return torch.t(torch.mm(transform.getRotationMatrix(), torch.t(data))) + transform.getTranslation()


def compareMatrix2D(A, B):
    """Compute averaged sum of squared differences"""

    assert A.size() == B.size()
    return torch.sum(torch.sum(torch.pow((A-B), 2), 1), 0) / torch.numel(A)


def compareVector(A, B):
    """Compute averaged sum of squared differences"""

    assert A.size() == B.size()
    return torch.sum(torch.pow((A-B), 2), 0) / torch.numel(A)


########################################################
# Tests
#

def test_runtime():
    """
    Comparison of methods in terms of
    Runtime vs Number of scene points

    """
    fileLog = open(FNAME_RUNTIME, "w")
    fileLog.write("#Points\t#Trials\tFinite Difference\tAutograd\tAutograd Cuda\n")
    fileLog.close()

    if torch.cuda.is_available():
        use_cuda = True

    # create known a rotation and translation
    tf = Transformation(rots=torch.FloatTensor([0., pi/2, 0.]), trans=torch.FloatTensor([10., 10., 0.]))
    # tf = Transformation(trans=torch.FloatTensor([0., 0., 0.]))

    # list for statistics
    # powers = [2, 4, 6, 8, 10, 12]
    powers = [2]
    N = []
    trials = 1

    sumFdTime = 0
    sumAgTime = 0
    sumAgCudaTime = 0

    timeFd = []
    timeAg = []
    timeAgCuda = []

    for k, p in enumerate(powers):
        N.append(int(pow(2, p)))
        # create sample scene points
        scenePts = createScene(N[k], seed=10)
        # create measurements
        measurePts = createMeasurements(scenePts, transform=tf, fixed=True)

        # print("Samples:\n{}".format(scenePts))
        # print("Measurements:\n{}".format(measurePts))

        # create jacobian matrices
        jacFd = torch.zeros([6, N[k]*3], dtype=torch.float32)
        jacAg = torch.zeros([6, N[k]*3], dtype=torch.float32)
        jacAgCuda = torch.zeros([6, N[k]*3], dtype=torch.float32)

        for trial in range(trials):
            # finite differences
            beginFd = time.time()
            rFd, tFd = kabsch_fd(measurePts, scenePts.clone(), jacFd)
            sumFdTime += (time.time()-beginFd)
            # timeFd.append(time.time()-beginFd)
            # print(rFd)
            # print(tFd)
            # print(jacFd)

            # autograd
            beginAg = time.time()
            rAg, tAg = kabsch_autograd(
                measurePts, scenePts.clone(), jacAg, False)
            sumAgTime += (time.time()-beginAg)
            # timeAg.append(time.time()-beginAg)
            # print(rAg)
            # print(tAg)
            # print(jacAg)

            # autograd with CUDA
            # FIXME: warm-up GPU
            beginAgCuda = time.time()
            rAgCuda, tAgCuda = kabsch_autograd(
                measurePts, scenePts.clone(), jacAgCuda, True)
            sumAgCudaTime += (time.time()-beginAgCuda)
            timeAgCuda.append(time.time()-beginAgCuda)
            print(rAgCuda)
            print(tAgCuda)
            # print(jacAgCuda)

        # calculate average runtime
        timeFd.append(sumFdTime/trials)
        timeAg.append(sumAgTime/trials)
        timeAgCuda.append(sumAgCudaTime/trials)

        sumFdTime = 0
        sumAgTime = 0
        sumAgCudaTime = 0

        # write runtime statistics for the current dataset size
        fileLog = open(FNAME_RUNTIME, "a")
        fileLog.write("{0}\t\t{1}\t\t{2:.6f}\t\t\t{3:.6f}\t{4:.6f}".format(N[k], trials, timeFd[k], timeAg[k], timeAgCuda[k]))
        fileLog.close()

        # statistics
        print("Summary of analysis:")
        print("Difference of rotation matrices:")
        print("Original rotation")
        print(tf.getRotationVector())
        print("\tfinite difference: {}".format(compareVector(rFd, tf.getRotationVector())))
        print(rFd)
        print("\tautograd: {}".format(compareVector(rAg, tf.getRotationVector())))
        print(rAg)
        print("\tautograd_cuda: {}".format(
            compareVector(rAgCuda.cpu(), tf.getRotationVector())))

        print("Difference of translation vectors:")
        print("Original")
        print(tf.getTranslation())
        print("\tfinite difference: {}".format(compareVector(tFd, tf.getTranslation())))
        print(tFd)
        print("\tautograd: {}".format(compareVector(tFd, tf.getTranslation())))
        print(tAg)
        # print("\tautograd_cuda: {}".format(
        #     tf.compareTranslation(agT_cuda.cpu())))

        print("Difference of jacobian matrices fdJac and agJac: {}".format(
            compareMatrix2D(jacFd, jacAg)))
        print(jacFd)
        print(jacAg)
        # print("Difference of jacobian matrices fdJac and agJac_cuda: {}".format(
        #     compareJacobian(jacFd, jacAgCuda)))
        print("========================================\n")

    # plot graph of runtime vs size of dataset
    # plt.figure()
    # plt.loglog(N, timeFd, marker="x", color="r", label="finite difference")
    # plt.loglog(N, timeAg, marker="x", color="b", label="autograd CPU")
    # plt.loglog(N, timeAgCuda, marker="x", color="g", label="autograd GPU")
    # plt.xlabel("N")
    # plt.ylabel("Runtime/sec")
    # plt.legend(loc=0)
    # plt.show()


def test_accuracy():
    """Tests for accuracy of gradient under all possible scenarios

    Test Input:
    1. Randomly generated scene points
    2. Various dataset size
    3. Multiple repetitions for each size

    Test Output:
    1. Percentage of degenerate cases
    2. Percentage of accuracy
    3.
    """

    if torch.cuda.is_available():
        use_cuda = True

    fileLog = open(FNAME_ACCURACY, "w")
    fileLog.write("#Points\t#Trials\t#Degeneracy\t#Jacobian errors\t#Translation errors\t#Rotation errors\t#Rotation errors (Finite Difference)\n")
    fileLog.close()

    fileLog = open(FNAME_DATA, "w")
    fileLog.write("\n")
    fileLog.close()

    powers = [2, 4, 6, 8, 10]
    N = []
    trials = 50  # trials per dataset size
    etol = 1e-4  # error tolerance of results
    delta = 0.01

    cntErrJac = 0
    cntErrRot = 0
    cntErrTrans = 0
    cntErrRotFD = 0
    cntDegen = 0

    sumErrJac = 0
    sumErrRot = 0
    sumErrTrans = 0
    sumErrRotFD = 0
    sumDegen = 0

    # create known a rotation and translation
    tf = Transformation()

    for k, p in enumerate(powers):
        N.append(int(pow(2, p)))
        # create sample scene point
        # scenePts = createScene(N[k], seed=None)
        # print(scenePts)
        # scenePts = createScene(4, seed=None)
        # create jacobian matrix
        jacAg = torch.zeros([6, N[k]*3], dtype=torch.float32)
        jacFd = torch.zeros([6, N[k]*3], dtype=torch.float32)

        for trial in range(trials):
            scenePts = createScene(N[k], seed=-1)
            # print(scenePts)
            measurePts = createMeasurements(scenePts, transform=tf, fixed=False)

            rAg, tAg = kabsch_stable(measurePts, scenePts.clone(), jacAg)

            if rAg is None and tAg is None:
                cntDegen += 1
                rAg, tAg = kabsch_fd(measurePts, scenePts.clone(), jacAg, delta)

            rFd, tFd = kabsch_fd(measurePts, scenePts.clone(), jacFd, delta)

            errJac = compareMatrix2D(jacAg, jacFd)
            errRot = compareVector(tf.getRotationVector(), rAg)
            errTrans = compareVector(tf.getTranslation(), tAg)
            errRotFD = compareVector(tf.getRotationVector(), rFd)

            # print("Original rotation: ")
            # print(tf.getRotationVector())
            # print("Estimated rotation: ")
            # print(r)
            # print("Jacobian FD:")
            # print(jacFd)
            # print("Jacobian autograd")
            # print(jacAg)

            if errJac > etol:
                cntErrJac += 1
                fileLog = open(FNAME_DATA, "a")
                fileLog.write("epsilon={0}, use_cuda={1}\n".format(delta), use_cuda)
                fileLog.write("Scene Points:\n")
                fileLog.write("{}\n".format(scenePts))
                fileLog.write("Original rotation angles: \n")
                fileLog.write("{}\n".format(tf.getRotationAngles()))
                fileLog.write("Original rotation vector: \n")
                fileLog.write("{}\n".format(tf.getRotationVector()))
                fileLog.write("est. Rotation: \n")
                fileLog.write("{}\n".format(rAg))
                fileLog.write("Original translation: \n")
                fileLog.write("{}\n".format(tf.getTranslation()))
                fileLog.write("est. Translation: \n")
                fileLog.write("{}\n".format(tAg))
                fileLog.write("Finite diff Jacobian: \n")
                fileLog.write("{}\n".format(jacFd))
                fileLog.write("est. Jacobian: \n")
                fileLog.write("{}\n".format(jacAg))
                fileLog.write("\n=========================================\n")
                fileLog.close()

            if errRot > etol:
                cntErrRot += 1

            if errTrans > etol:
                cntErrTrans += 1

            if errRotFD > etol:
                cntErrRotFD += 1

            sumErrJac += errJac
            sumErrRot += errRot
            sumErrTrans += errTrans
            sumErrRotFD += errRotFD

        sumErrJac /= trials
        sumErrRot /= trials
        sumErrTrans /= trials

        fileLog = open(FNAME_ACCURACY, "a")
        fileLog.write("{0}\t\t{1}\t\t{2}\t\t\t{3}({7})\t{4}({8})\t{5}({9})\t{6}\n\n".format(N[k],trials,cntDegen, cntErrJac, cntErrTrans, cntErrRot, cntErrRotFD, sumErrJac, sumErrTrans, sumErrRot))
        fileLog.close()

        sumErrJac = 0
        sumErrRot = 0
        sumErrTrans = 0
        sumErrRotFD = 0

        cntDegen = 0
        cntErrJac = 0
        cntErrRot = 0
        cntErrTrans = 0
        cntErrRotFD = 0


########################################################
# Main routine to run the analysis
#
if __name__ == '__main__':

    test_runtime()
    # test_accuracy()
