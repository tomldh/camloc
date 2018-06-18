import torch
import math

# https://j-towns.github.io/papers/svd-derivative.pdf
#
# This makes no assumption on the signs of sigma.


def svd_backward(grads, self, some, raw_u, sigma, raw_v):
    m = self.size()[0]
    n = self.size()[1]
    k = sigma.size()[0]
    gsigma = grads[1]

    u = raw_u
    v = raw_v
    gu = grads[0]
    gv = grads[2]

    guDefined = True  # FIXME: to replace "gu.defined()"
    gvDefined = True  # FIXME: to replace "gv.defined()"
    gsDefined = False  # FIXME: to replace "gs.defined()"

    if not some:
        # We ignore the free subspace here because possible base vectors cancel
        # each other, e.g., both - v and +v are valid base for a dimension.
        # Don't assume behavior of any particular implementation of svd.
        u = raw_u.narrow(1, 0, k)
        v = raw_v.narrow(1, 0, k)
        if guDefined:  # gu.defined():
            gu = gu.narrow(1, 0, k)

        if gvDefined:  # gv.defined()
            gv = gv.narrow(1, 0, k)

    vt = v.t()

    if gsDefined:  # gsigma.defined():
        sigma_term = u.mm(gsigma.diag()).mm(vt)
    else:
        sigma_term = torch.zeros(1, dtype=torch.float).expand_as(self)

    # in case that there are no gu and gv, we can avoid the series of kernel
    # calls below
    # if not gv.defined() and not gu.defined():
    if not gvDefined and not guDefined:
        return sigma_term

    ut = u.t()

    im = torch.eye(m, dtype=torch.float)
    inn = torch.eye(n, dtype=torch.float)
    sigma_mat = sigma.diag()
    print("sigma_mat")
    print(sigma_mat)
    sigma_mat_inv = sigma.pow(-1).diag()
    # for i in range(sigma_mat_inv.size()[0]):
    #     for j in range(sigma_mat_inv.size()[1]):
    #         if math.isinf(sigma_mat_inv[i, j]):
    #             sigma_mat_inv[i, j] = 0
    print("sigma_mat_inv")
    print(sigma_mat_inv)
    sigma_expanded_sq = sigma.pow(2).expand_as(sigma_mat)
    print("sigma_expanded_sq")
    print(sigma_expanded_sq)
    F = sigma_expanded_sq - sigma_expanded_sq.t()
    print("F")
    print(F)
    # The following two lines invert values of F, and fills the diagonal with 0s.
    # Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
    # first to prevent nan from appearing in backward of this function.
    # F.as_strided(k, k + 1).fill_(INFINITY)
    F[0, 0] = math.inf  # FIXME: to replace as_strided()
    F[1, 1] = math.inf  # FIXME: to replace as_strided()
    F[2, 2] = math.inf  # FIXME: to replace as_strided()
    F = F.pow(-1)
    # for i in range(F.size()[0]):
    #     for j in range(F.size()[1]):
    #         if math.isinf(F[i, j]):
    #             F[i, j] = 0
    print("F inverse")
    print(F)
    if guDefined:  # gu.defined():
        u_term = u.mm(F.mul(ut.mm(gu) - gu.t().mm(u))).mm(sigma_mat)
        if m > k:
            u_term = u_term + (im - u.mm(ut)).mm(gu).mm(sigma_mat_inv)

        u_term = u_term.mm(vt)
    else:
        u_term = torch.zeros(self.type(), {1}).expand_as(self)

    if gvDefined:  # gv.defined():
        gvt = gv.t()
        v_term = sigma_mat.mm(F.mul(vt.mm(gv) - gvt.mm(v))).mm(vt)
        if n > k:
            v_term = v_term + sigma_mat_inv.mm(gvt.mm(inn - v.mm(vt)))

        v_term = u.mm(v_term)
    else:
        v_term = torch.zeros(1, dtype=torch.float).expand_as(self)

    return u_term + sigma_term + v_term


def workingset():
    # loss(u11) w.r.t output (U)
    gU = torch.FloatTensor([[0.0000,  0.0000,  0.0000],
                            [0.0000, -0.0000,  0.7854],
                            [0.5554, -0.5554, -0.0000]])
    gS = None
    gV = torch.FloatTensor([[0.0000,  0.0000, -0.0000],
                            [0.5554,  0.5554, -0.0000],
                            [-0.5554,  0.5554, -0.0000]])

    # input (A)
    A = torch.FloatTensor([[-0.0000, -0.0000,  0.0000],
                           [1.0000,  2.0000,  0.0000],
                           [2.0000,  1.0000,  0.0000]])

    # some (default: True)
    some = True

    # raw_u (U)
    U = torch.FloatTensor([[0.0000,  0.0000,  1.0000],
                           [0.7071, -0.7071, -0.0000],
                           [0.7071,  0.7071,  0.0000]])

    # sigma (S)
    S = torch.FloatTensor([3.0000,  1.0000,  0.0000])

    # raw_v
    V = torch.FloatTensor([[0.7071,  0.7071,  0.0000],
                           [0.7071, -0.7071,  0.0000],
                           [0.0000,  0.0000,  1.0000]])

    jacA_X = torch.FloatTensor([
        [-0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000, -0.0000,  0.0000,  0.0000],
        [0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000, -0.0000,  0.0000],
        [0.0000,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000, -0.0000],
        [0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  1.0000,
         0.0000,  0.0000,  1.0000,  0.0000,  0.0000],
        [0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         1.0000,  0.0000,  0.0000,  1.0000,  0.0000],
        [0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  1.0000,  0.0000,  0.0000,  1.0000],
        [1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  1.0000,  0.0000,  0.0000],
        [0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  1.0000,  0.0000],
        [0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  1.0000]])

    return [gU, gS, gV], A, some, U, S, V, jacA_X.t()


def problemset():

    # For dataset:
    # [0.5, -0.5, 0.],
    # [-0.5, -0.5, 0.],
    # [-0.5, 0.5, 0.],
    # [0.5, 0.5, 0.]

    # loss(r1) w.r.t output (U)
    # gU = torch.FloatTensor([[0.0000,  0.0000,  0.0000],
    #                         [0.0000,  0.0000,  0.7854],
    #                         [0.0000,  0.7854,  0.0000]])

    # loss (r2) w.r.t output (U)
    # gU = torch.FloatTensor([[0.5000,  0.0000, -0.7854],
    #                         [0.0000,  0.5000,  0.0000],
    #                         [-0.7854,  0.0000, -0.5000]])

    gU = torch.FloatTensor([[0.0000, -0.7854,  0.0000],
                            [0.7854,  0.0000,  0.0000],
                            [0.0000,  0.0000,  0.0000]])

    # loss (r1) w.r.t output(V)
    # gV = torch.FloatTensor([[0.0000,  0.0000,  0.0000],
    #                         [0.7854,  0.0000,  0.0000],
    #                         [0.0000, -0.7854,  0.0000]])

    # loss (r2) w.r.t output(V)
    # gV = torch.FloatTensor([[-0.7854,  0.0000, -0.5000],
    #                         [0.0000,  0.5000,  0.0000],
    #                         [0.5000,  0.0000, -0.7854]])

    gV = torch.FloatTensor([[0.0000,  0.7854,  0.0000],
                            [0.0000,  0.0000,  0.7854],
                            [0.0000,  0.0000,  0.0000]])

    gS = None  # not involved in calculating rotation (R = UV^T)

    # input (A)
    A = torch.FloatTensor([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]])

    # some (default: True)
    some = True

    # raw_u (U)
    U = torch.FloatTensor([[0., 0., 1.],
                           [0., 1., 0.],
                           [1., 0., 0.]])

    # sigma (S)
    S = torch.FloatTensor([1.0+1e-1, 1.0-1e-1, 0.+1e-1])

    # raw_v
    V = torch.FloatTensor([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]])

    # jacobian of A w.r.t prefixed X
    jacA_X = torch.FloatTensor([
        [-0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
            0.0000, 0.0000,  0.0000, -0.0000,  0.0000,  0.0000],
        [0.0000, -0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000, 0.0000,  0.0000,  0.0000, -0.0000,  0.0000],
        [0.0000,  0.0000, -0.0000,  0.0000,  0.0000,  0.0000,
         0.0000, 0.0000,  0.0000,  0.0000,  0.0000, -0.0000],
        [-0.5000,  0.0000,  0.0000, -0.5000,  0.0000,  0.0000,
         0.5000, 0.0000,  0.0000,  0.5000,  0.0000,  0.0000],
        [0.0000, -0.5000,  0.0000,  0.0000, -0.5000,  0.0000,
         0.0000, 0.5000,  0.0000,  0.0000,  0.5000,  0.0000],
        [0.0000,  0.0000, -0.5000,  0.0000,  0.0000, -0.5000,
         0.0000, 0.0000,  0.5000,  0.0000,  0.0000,  0.5000],
        [0.5000,  0.0000,  0.0000, -0.5000,  0.0000,  0.0000, -
         0.5000, 0.0000,  0.0000,  0.5000,  0.0000,  0.0000],
        [0.0000,  0.5000,  0.0000,  0.0000, -0.5000,  0.0000,
         0.0000, -0.5000,  0.0000,  0.0000,  0.5000,  0.0000],
        [0.0000,  0.0000,  0.5000,  0.0000,  0.0000, -0.5000,  0.0000, 0.0000, -0.5000,  0.0000,  0.0000,  0.5000]])

    return [gU, gS, gV], A, some, U, S, V, jacA_X.t()


if __name__ == '__main__':

    # grad_output, A, some, U, S, V, jacA_X = workingset()
    grad_output, A, some, U, S, V, jacA_X = problemset()

    result = svd_backward(grad_output, A, some, U, S, V)
    print("result of delta Loss w.r.t A")
    print(result)

    print("Gradient Loss w.r.t prefixed X:")
    print(jacA_X.mm(result.view(9, -1)).view(-1, 3))
