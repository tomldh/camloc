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

# file names
FNAME_ACCURACY = "test_accuray.txt"
FNAME_RUNTIME = "test_runtime.txt"
FNAME_DATA = "problem_data.txt"


##########################################
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
            rots (tensor): user-defined rotational angles, [angleX, angleY, angleZ], 0 <= angle < 2*PI

            trans (tensor): user-defined translation
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
        """ Returns a rotation matrix based on rotational angles """

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
        """ return angles w.r.t to current rotation matrix """
        return self.rots

    def getRotationVector(self):
        """ return angle-axis rotation vector """
        return torch.from_numpy(self.R).float()

    def getRotationMatrix(self):
        """ return rotation matrix """
        return torch.from_numpy(cv2.Rodrigues(self.R)[0]).float()

    def getTranslation(self):
        """ return translation vector """
        return self.T

    def setRotation(self, rots):
        """ set rotation vector
        Args:
            rots - angles of rotations
        """
        self.rots = rots
        self.R = cv2.Rodrigues(self.__computeRotationMatrix__())[0]

    def setTranslation(self, trans):
        """ set translation vector """
        self.T = trans


########################################################
# Custom autograd for cv2.Rodrigues()
#
class Rodrigues(torch.autograd.Function):
    """ This class implements the forward and backward passes for Rodrigues fcn
    """

    @staticmethod
    def forward(ctx, input):
        # keep track of original tensor type
        isCuda = input.is_cuda

        # cuda tensor will be converted to cpu tensor
        # no change if tensor is already cpu
        input = input.cpu()

        # use opencv function to convert rotation matrix to vector
        r, jac = cv2.Rodrigues(input.detach().numpy())
        r = torch.from_numpy(r)  # convert numpy to tensor type
        jac = torch.from_numpy(jac)  # convert numpy to tensor type

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
# Function that performs kabsch algorithm and differentiation using
# using central finite differences.
#
def kabsch_fd(P, X, jacobian=None, eps=0.01):
    """
    Args:
        P (tensor): Measurements
        X (tensor): Scene points
        jacobian (tensor): 6x3N jacobian matrix of r|t w.r.t scene point coord.
        eps (float): Epsilon used in finite difference approximation

    Return:
        r (tensor): 3x1 rotation vector that satisfies P = RX+T
        t (tensor): 3x1 translation vector that satisfies P = RX+T
    """
    print("Running Kabsch (Finite Difference)...")

    if P.size() != X.size():
        print("\tError: Dimension of P and X is not the same.")

    if X.size()[1] != 3:
        print("\tError: Expected 3D coordinates, but got {}.".format(
            X.size()[1]))

    r, t = kabsch(P, X)

    # Return rotation matrix only if no gradient is required
    if jacobian is None:
        return r, t

    print("\tComputing jacobian...")

    for i in range(X.size()[0]):
        for j in range(3):
            # forward step
            X[i, j] += eps
            fwdR, fwdT = kabsch(P, X)

            # backward step
            X[i, j] -= 2*eps
            bwdR, bwdT = kabsch(P, X)

            # return to original
            X[i, j] += eps

            diffR = (fwdR-bwdR)/(2*eps)
            diffT = (fwdT-bwdT)/(2*eps)

            # place derivatives to column (w.r.t X[i, j]) in jacobian matrix
            jacobian[:3, i*3+j] = diffR.view(-1)
            jacobian[3:, i*3+j] = diffT.view(-1)

    return r, t


########################################################
# Function that performs kabsch algorithm with autograd and finite differences
# Non-degenerate case: autograd
# degenerate case: finite difference
#
def kabsch_autograd(P, X, jacobian=None, use_cuda=False, eps=0.01):
    """
    Args:
        P (tensor): Measurements
        X (tensor): Scene points
        jacobian (tensor) 6x3N jacobian matrix of r&t w.r.t scene point coord.
        use_cuda (bool): Flag to indicate whether to calculate jacobian
        eps (float): Epsilon used in finite difference approximation

    Return:
        r (tensor): 3x1 rotation vector that satisfies P = RX + T
        t (tensor): 3x1 translation vector that satisfies P = RX + T
    """

    print("Running Kabsch (Autograd)")

    if use_cuda:
        print("\tUsing CUDA")
        X = X.cuda()
        P = P.cuda()

    if jacobian is None:
        r, t = kabsch(P, X)
        return r, t

    print("\tComputing jacobian")

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

    # flag for degeneracy
    degenerate = False

    # degenerate if any singular value is zero
    if torch.numel(torch.nonzero(S)) != torch.numel(S):
        degenerate = True

    # degenerate if singular values are not distinct
    if torch.abs(S[0]-S[1]) < 1e-8 or torch.abs(S[0]-S[2]) < 1e-8 or torch.abs(S[1]-S[2]) < 1e-8:
        degenerate = True

    # if degenerate, use finite difference for stability
    if degenerate is True:
        X.requires_grad = False
        return None, None

    # non-degenerate case, continue with Kabsch algorithm with autograd
    Vt = torch.t(V)

    d = torch.det(torch.mm(U, Vt))

    D = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    if use_cuda:
        D = D.cuda()

    R = torch.mm(U, torch.mm(D, Vt))

    rod = Rodrigues.apply

    r = rod(R)  # rotation vector

    numelR = torch.numel(r)

    # compute jacobian matrix
    for i in range(numelR):
        onehot = torch.zeros(numelR, dtype=torch.float32)
        onehot[i] = 1

        # jacobian of an element of r w.r.t X
        if use_cuda:
            r.backward(onehot.view(r.size()).cuda(), retain_graph=True)
        else:
            r.backward(onehot.view(r.size()), retain_graph=True)
        jacobian[i, :] = X.grad.data.view(-1)

        # zero the gradient for next element
        X.grad.data.zero_()

        # jacobian of an element of t w.r.t X
        if use_cuda:
            t.backward(onehot.view(t.size()).cuda(), retain_graph=True)
        else:
            t.backward(onehot.view(t.size()), retain_graph=True)
        jacobian[i+3, :] = X.grad.data.view(-1)

        # zero the gradient for next element
        X.grad.data.zero_()

    return r.detach(), t.detach()


########################################################
# Helper functions
#
def kabsch(P, X):
    """ Kabsch algorithm without gradient calculations

    Args:
        P (tensor): measurement points
        X (tensor): scene points

    Return:
        r (tensor): 3x1 rotation vector that satisfies P = RX + T
        t (tensor): 3x1 translation vector that satisfies P = RX + T
    """

    tx = torch.mean(X, 0)
    tp = torch.mean(P, 0)

    t = tp - tx  # translation vector

    # move centroid to origin
    Xc = X.sub(tx)
    Pc = P.sub(tp)

    A = torch.mm(torch.t(Pc), Xc)

    U, S, V = torch.svd(A)

    Vt = torch.t(V)

    d = torch.det(torch.mm(U, Vt))

    D = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    R = torch.mm(U, torch.mm(D, Vt))

    r = torch.from_numpy(cv2.Rodrigues(R.numpy())[0])  # rotation vector

    return r, t


def createScene(N, seed=None):
    """create 3D scene point data

    Args:
        N (int): number of points
        seed (int): seed for random value generation
                (None: fixed value, <=0: random seed, >0: fixed seed)

    Return:
        data (tensor): Nx3 scene point matrix
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
        data (tensor): scene points
        transform (class object): object which contains rotation and translation
        fixed (bool):
            - if True, existing transformation will be overriden by random transformation
            - if False, use existing transformation

    Return:
        tfData (tensor): Nx3 transformed scene points used as measurement

    """

    assert data is not None
    assert transform is not None
    assert data.size()[1] == 3

    # if not fixed, set a new transformation with random values
    if fixed is False:
        # set rotational angles between 0 and 2*PI
        transform.setRotation(torch.rand(3, dtype=torch.float32) * 2 * pi)
        # translation vector drawn from normal distribution
        transform.setTranslation(torch.randn(3, dtype=torch.float32))

    # P = RX + T
    return torch.t(torch.mm(transform.getRotationMatrix(), torch.t(data))) + transform.getTranslation()


def compareMatrix2D(A, B):
    """Compute averaged sum of squared differences"""

    assert A.size() == B.size()
    return float(torch.sum(torch.sum(torch.pow((A-B), 2), 1), 0) / torch.numel(A))


def compareVector(A, B):
    """Compute averaged sum of squared differences"""

    assert A.size() == B.size()
    return float(torch.sum(torch.pow((A-B), 2), 0) / torch.numel(A))


def writeToFile(filename, mode, format, contents):
    """ Write the contents into file in table format

    Args:
        filename (string): name of output file
        contents (OrderedDict): contents of the file
        mode (string): "w", "a", etc.

    """
    fh = open(filename, mode)

    if format == "table":

        for k in range(len(contents)):
            # different precision for different datatypes
            val = contents[k][1]
            if type(val) is int:
                fh.write("{}\t".format(val))
            elif type(val) is float:
                fh.write("{:6f}\t".format(val))

        fh.write("\n")

    elif format == "default":
        for k in range(len(contents)):
            fh.write("{}\n".format(contents[k][0]))
            fh.write("\t{}\n\n".format(contents[k][1]))

    fh.close()


def cudaWarmUp():
    """ warm-up GPU by data copying between GPU and CPU """

    data = torch.zeros(100, dtype=torch.float32)

    data = data.cuda()

    data = data.cpu()

    return


########################################################
# Test functions
#
def test_runtime():
    """Comparison of runtime for different methods

    Test Input:
        Dataset of different sizes

    Test Output:
        1. Runtime of finite difference, PyTorch autograd and autograd(CUDA)
        2. Graph of runtime vs scene points

    """
    # write column headers for file contents
    fh = open(FNAME_RUNTIME, "w")
    fh.write("#Points\t#Trials\tFinite Difference(sec)\tAutograd(sec)\tAutograd_GPU(sec)\n")
    fh.close()

    # writeToFile(FNAME_RUNTIME, )
    if torch.cuda.is_available():
        use_cuda = True

    # create known a rotation and translation
    tf = Transformation()

    powers = [2, 4, 6, 8, 10]
    N = []  # number of points by 2^power
    trials = 100  # number of repetitions for each dataset size

    # sum of time for all the trials of each data dataset size
    sumFdTime = 0
    sumAgTime = 0
    sumAgCudaTime = 0

    # averaged runtime for each data dataset size
    timeFd = []
    timeAg = []
    timeAgCuda = []

    # CUDA warm-up for more stable runtime performance
    cudaWarmUp()

    for k, p in enumerate(powers):
        # compute dataset size
        N.append(int(pow(2, p)))
        # create sample scene points
        scenePts = createScene(N[k], seed=10)
        # create measurements
        measurePts = createMeasurements(scenePts, transform=tf, fixed=True)

        # create jacobian matrices
        jacFd = torch.zeros([6, N[k]*3], dtype=torch.float32)
        jacAg = torch.zeros([6, N[k]*3], dtype=torch.float32)
        jacAgCuda = torch.zeros([6, N[k]*3], dtype=torch.float32)

        # run some reptitions for each dataset size
        for trial in range(trials):
            # finite differences
            beginFd = time.time()
            rFd, tFd = kabsch_fd(measurePts, scenePts.clone(), jacFd)
            sumFdTime += (time.time()-beginFd)

            # autograd
            beginAg = time.time()
            rAg, tAg = kabsch_autograd(
                measurePts, scenePts.clone(), jacAg, False)
            sumAgTime += (time.time()-beginAg)

            # autograd with CUDA
            # FIXME: warm-up GPU
            beginAgCuda = time.time()
            rAgCuda, tAgCuda = kabsch_autograd(
                measurePts, scenePts.clone(), jacAgCuda, True)
            sumAgCudaTime += (time.time()-beginAgCuda)

        # calculate average runtime
        timeFd.append(sumFdTime/trials)
        timeAg.append(sumAgTime/trials)
        timeAgCuda.append(sumAgCudaTime/trials)

        sumFdTime = 0
        sumAgTime = 0
        sumAgCudaTime = 0

        # write runtime statistics for the current dataset size
        writeToFile(FNAME_RUNTIME, "a", "table", [("#Points", N[k]), ("#Trials", trials), ("Finite Difference", timeFd[k]), ("Autograd", timeAg[k]), ("Autograd(GPU)", timeAgCuda[k])])

    # plot graph of runtime vs size of dataset
    plt.figure()
    plt.loglog(N, timeFd, marker="x", color="r", label="finite difference")
    plt.loglog(N, timeAg, marker="x", color="b", label="autograd CPU")
    plt.loglog(N, timeAgCuda, marker="x", color="g", label="autograd GPU")
    plt.xlabel("Number of Points")
    plt.ylabel("Runtime/sec")
    plt.legend(loc=0)
    # plt.show()
    plt.savefig("runtime.png")


def test_accuracy():
    """Tests for accuracy of gradient under all possible scenarios

    Test Input:
    1. Randomly generated scene points
    2. Various dataset size
    3. Multiple repetitions for each size

    Test Output:
    1. Number of degenerate cases
    2. Errors of rotation, translation, jacobian matrix
    3. Problematic dataset which has wrongly computed jacobian matrix

    """

    if torch.cuda.is_available():
        use_cuda = True

    # write column headers for file contents
    fh = open(FNAME_ACCURACY, "w")
    fh.write("#Points\t#Trials\t#Degeneracy\t#Jacobian errors\t#Translation errors\t#Rotation errors\t#Rotation errors (Finite Difference)\tJacobian error\tTranslation error\tRotation error\n")
    fh.close()

    fh = open(FNAME_DATA, "w")
    fh.write("\n")
    fh.close()

    powers = [2, 4, 6, 8, 10]
    N = []  # number of points by 2^power
    trials = 100  # trials per dataset size
    etol = 1e-3  # error tolerance of results FIXME: use different tolerance?
    delta = 0.01  # used in finite difference calculation

    # number of cases in all trials per dataset size
    cntErrJac = 0
    cntErrRot = 0
    cntErrTrans = 0
    cntErrRotFD = 0
    cntDegen = 0

    # sum of errors in all trials per dataset size
    sumErrJac = 0
    sumErrRot = 0
    sumErrTrans = 0
    sumErrRotFD = 0

    # create known a rotation and translation
    tf = Transformation()

    for k, p in enumerate(powers):
        # compute dataset size
        N.append(int(pow(2, p)))

        # scene point created here will be fixed in the trials later
        # scenePts = createScene(N[k], seed=None)

        # create jacobian matrix
        jacAg = torch.zeros([6, N[k]*3], dtype=torch.float32)
        jacFd = torch.zeros([6, N[k]*3], dtype=torch.float32)

        # run some reptitions for each dataset size
        for trial in range(trials):
            # scene point created here will change in every trial
            scenePts = createScene(N[k], seed=-1)

            # create measurements with random rotation and translation
            measurePts = createMeasurements(scenePts, transform=tf, fixed=False)

            # run Kabsch with autograd
            rAg, tAg = kabsch_autograd(measurePts, scenePts.clone(), jacAg)

            # run finite difference version if degenerate case is encountered
            if rAg is None and tAg is None:
                cntDegen += 1
                rAg, tAg = kabsch_fd(measurePts, scenePts.clone(), jacAg, delta)

            rFd, tFd = kabsch_fd(measurePts, scenePts.clone(), jacFd, delta)

            errJac = compareMatrix2D(jacAg, jacFd)
            errRot = compareVector(tf.getRotationVector(), rAg)
            errTrans = compareVector(tf.getTranslation(), tAg)
            errRotFD = compareVector(tf.getRotationVector(), rFd)

            if errJac > etol:
                cntErrJac += 1

                writeToFile(FNAME_DATA, "a", "default", [("epsilon:", delta), ("use_cuda:", use_cuda), ("scene points:", scenePts), ("original rotation angles:", tf.getRotationAngles()), ("Original rotation vector", tf.getRotationVector()), ("Estimated Rotation:", rAg), ("Original translation:", tf.getTranslation()), ("Estimated translation", tAg), ("Jacobian(Finite Difference)", jacFd), ("Estimated Jacobian:", jacAg), ("NEW DATASET", "========================================")])

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

        # write statistics to file
        writeToFile(FNAME_ACCURACY, "a", "table", [("#Points", N[k]), ("#Trials", trials), ("#Degeneracy", cntDegen), ("#Jacobian errors", cntErrJac), ("#Translation errors", cntErrTrans), ("#Rotation errors", cntErrRot), ("#Rotation errors (Finite Difference)", cntErrRotFD), ("Jacobian error", sumErrJac), ("Translation error", sumErrTrans), ("Rotation error", sumErrRot)])

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

    # test_runtime()
    test_accuracy()
