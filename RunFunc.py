import numpy as np
from scipy import constants
import scipy

def pressFunc(x, b1, b2, h0, p0):
    b = np.ones(len(x))
    b[x<=h0] = b1
    b[x>h0] = b2
    return np.exp(-b * (x -h0)  + np.log(p0))

def temp_func(x,h0,h1,h2,h3,h4,h5,a0,a1,a2,a3,a4,b0):
    a = np.ones(x.shape)
    b = np.ones(x.shape)
    a[x < h0] = a0
    a[h0 <= x] = 0
    a[h1 <= x] = a1
    a[h2 <= x] = a2
    a[h3 <= x] = 0
    a[h4 <= x ] = a3
    a[h5 <= x ] = a4
    b[x < h0] = b0
    b[h0 <= x] = b0 + h0 * a0
    b[h1 <= x] = b0 + h0 * a0
    b[h2 <= x] = a1 * (h2-h1) + b0 + h0 * a0
    b[h3 <= x ] = a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    b[h4 <= x ] = a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    b[h5 <= x ] = a3 * (h5-h4) + a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    h = np.ones(x.shape)
    h[x < h0] = 0
    h[h0 <= x] = h0
    h[h1 <= x] = h1
    h[h2 <= x] = h2
    h[h3 <= x] = h3
    h[h4 <= x] = h4
    h[h5 <= x] = h5
    return a * (x - h) + b

def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = 1#(5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def composeAforO3(A_lin, temp, press, ind, wvnmbr, g_doub_prime, E, S):


    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]

    f_broad = 1
    scalingConst = 1e11
    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3 * 1e8
    C2 = constants.h * constants.c * 1e2 * v_0 / (constants.Boltzmann * temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1))

    # take linear
    num_mole = 1 / (constants.Boltzmann)  # * temp_values)

    AscalConstKmToCm = 1e3
    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    # 1e2 for pressure values from hPa to Pa
    A_scal = press.reshape((SpecNumLayers, 1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm / (temp)
    theta_scale = num_mole *  f_broad * 1e-4 * scalingConst * S[ind, 0]
    A = A_lin * A_scal.T
    #np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
    return A, theta_scale


def composeAforO3withTemp(A_lin, temp, press, ind, wvnmbr, g_doub_prime, E, S, newTemp):


    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]

    f_broad = 1
    scalingConst = 1e11
    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3 * 1e8
    C2 = constants.h * constants.c * 1e2 * v_0 / (constants.Boltzmann * temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1))

    # take linear
    num_mole = 1 / (constants.Boltzmann)  # * temp_values)

    AscalConstKmToCm = 1e3
    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    # 1e2 for pressure values from hPa to Pa
    A_scal = press.reshape((SpecNumLayers, 1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm / (newTemp)
    theta_scale = num_mole *  f_broad * 1e-4 * scalingConst * S[ind, 0]
    A = A_lin * A_scal.T
    #np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
    return A, theta_scale

def genDataFindandtestMap(currMap, tang_heights_lin, A_lin_dx,  height_values, gamma0, VMR_O3, Results, AscalConstKmToCm, A_lin, temp_values, pressure_values, ind, scalingConst, relMapErrDat, wvnmbr, S, E,g_doub_prime):
    '''Find map from linear to non-linear data'''

    SpecNumMeas, SpecNumLayers = A_lin.shape
    relMapErr = relMapErrDat
    while np.mean(relMapErr) >= relMapErrDat:
        testDat = SpecNumMeas
        A_O3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, wvnmbr, g_doub_prime, E, S)

        # Results = np.zeros((SampleRounds, SpecNumLayers))
        LinDataY = np.zeros((testDat, SpecNumMeas))
        NonLinDataY = np.zeros((testDat, SpecNumMeas))
        for test in range(testDat):
            #print(test)
            ProfRand = test#np.random.randint(low=0, high=SampleRounds)
            # Results = np.loadtxt(f'Res{testSet}.txt')

            O3_Prof = np.copy(Results[ProfRand])
            O3_Prof[O3_Prof < 0] = 0
            nonLinA = calcNonLin(tang_heights_lin, A_lin_dx,  height_values, pressure_values, ind, temp_values, VMR_O3, AscalConstKmToCm, wvnmbr, S, E,g_doub_prime)
            noise = np.random.normal(0, np.sqrt(1 / gamma0), SpecNumMeas)
            # noise = np.zeros(SpecNumMeas)

            LinDataY[test] = np.matmul( currMap @ (A_O3 * 2), O3_Prof.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
                SpecNumMeas) #+ noise
            NonLinDataY[test] = np.matmul(A_O3 * nonLinA,
                                          O3_Prof.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
                SpecNumMeas) #+ noise
            #currMap = np.eye(SpecNumMeas)


        RealMap = LinModelSolve(LinDataY, NonLinDataY, SpecNumMeas)


        #test map do find error
        testNum = 10# len(Results)#100
        testO3 = np.copy(Results)#np.random.multivariate_normal(VMR_O3.reshape(SpecNumLayers), 1e-18 * L_d, size=testNum)
        testO3[testO3 < 0] = 0

        testDataY = np.zeros((testNum, SpecNumMeas))
        testNonLinY = np.zeros((testNum, SpecNumMeas))

        for k in range(0, testNum):
            currO3 = testO3[k]
            noise = np.random.normal(0, np.sqrt(1 / gamma0), SpecNumMeas)
            nonLinA = calcNonLin(tang_heights_lin, A_lin_dx,  height_values, pressure_values, ind, temp_values, VMR_O3, AscalConstKmToCm, wvnmbr, S, E,g_doub_prime)

            testDataY[k] = np.matmul(currMap @ (A_O3 * 2), currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(SpecNumMeas)# + noise
            testNonLinY[k] = np.matmul(A_O3 * nonLinA, currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
                SpecNumMeas) #+ noise
            testNonLinY[k] = testDataY[k] #+ noise

        relMapErr = testSolvedMap(RealMap, gamma0, testNum, SpecNumMeas, testNonLinY, testDataY)
        print('mean realtive Error: ' + str(np.mean(relMapErr)))
        currMap = RealMap @ np.copy(currMap)
        relMapErr = 1
    return currMap, relMapErr, LinDataY, NonLinDataY, testO3

def LinModelSolve(LinDataY, NonLinDataY, SpecNumMeas):
    # basis
    RealMap = None
    while RealMap is None:
        try:

            RealMap = np.zeros((SpecNumMeas, SpecNumMeas))

            for i in range(0, SpecNumMeas):
                RealMap[i,:] = np.linalg.solve(LinDataY, NonLinDataY[:, i])

        except np.linalg.LinAlgError:
            RealMap = None
            print('pass')
            pass

    return RealMap

def testSolvedMap(RealMap, gamma0, testNum, SpecNumMeas, testNonLinY, testDataY):
    relMapErr = np.zeros(testNum)
    for k in range(0, testNum):

        noise = np.random.multivariate_normal(np.zeros(SpecNumMeas), np.sqrt(1 / gamma0) * np.eye(SpecNumMeas))

        mappedDat = (RealMap @ testDataY[k]) #+ noise
        relMapErr[k] = np.linalg.norm((testNonLinY[k]) - mappedDat) / np.linalg.norm((testNonLinY[k])) * 100


    return relMapErr

def gen_trap_rul(dxs):
    #val = np.zeros(len(dxs)+1)
    sumMat = np.eye(len(dxs)+1)
    Ones = np.ones((len(dxs)+1,len(dxs)+1))
    sumMat = sumMat + np.triu(Ones,1) - np.triu(Ones,2)
    return 0.5*(dxs @ np.copy(sumMat[:-1,:]))

def calcNonLin(tang_heights, dxs,  height_values, pressure_values, ind, temp_values, VMR_O3, AscalConstKmToCm, wvnmbr, S, E,g_doub_prime):
    '''careful that A_lin is just dx values
    maybe do A_lin_copy = np.copy(A_lin/2)
    A_lin_copy[:,-1] = A_lin_copy[:,-1] * 2
    if A_lin has been generated for linear data'''

    SpecNumLayers = len(temp_values)#
    SpecNumMeas = len(tang_heights)
    temp = temp_values.reshape((SpecNumLayers, 1))
    # wvnmbr = np.loadtxt('wvnmbr.txt').reshape((909,1))
    # S = np.loadtxt('S.txt').reshape((909,1))
    # E = np.loadtxt('E.txt').reshape((909,1))
    # g_doub_prime = np.loadtxt('g_doub_prime.txt').reshape((909,1))

    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]

    f_broad = 1
    #scalingConst = 1e11
    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))



    # take linear
    num_mole = 1 / (constants.Boltzmann)

    theta = num_mole * f_broad * 1e-4 * VMR_O3.reshape((SpecNumLayers,1)) * S[ind,0]
    ConcVal = - pressure_values.reshape((SpecNumLayers, 1)) * 1e2 * LineIntScal / temp_values * theta * AscalConstKmToCm

    afterTrans = np.zeros((SpecNumMeas, SpecNumLayers))
    preTrans = np.zeros((SpecNumMeas, SpecNumLayers))
    for i in range(0,SpecNumMeas):
        t = 0
        while height_values[t] <= tang_heights[i]:
            t += 1
        flipDxs = np.flip(dxs[i, t - 1:])
        flipVal = np.flip(ConcVal[t - 1:])
        currDxs = gen_trap_rul(np.append(flipDxs, dxs[i, t - 1]))
        ValPerLayAfter = np.sum(np.append(flipVal , ConcVal[t]) * currDxs)
        afterTrans[i, t - 1] = np.exp(ValPerLayAfter)
        for j in range(t-1, SpecNumLayers-1):
            currDxs = gen_trap_rul(dxs[i,j:])
            ValPerLayPre = np.sum(ConcVal[j:].T  * currDxs)
            preTrans[i,j] = np.exp(ValPerLayPre)

            if j >= t:
                currDxs = gen_trap_rul(np.append(flipDxs, dxs[i, t - 1:j]))
                ValPerLayAfter = np.sum(np.append(flipVal , ConcVal[t:j + 1]) * currDxs)
                afterTrans[i, j] = np.exp(ValPerLayAfter)

        currDxs = gen_trap_rul(np.append(flipDxs, dxs[i, t - 1:]))
        ValPerLayAfter = np.sum(np.append(np.flip(ConcVal[t - 1:]), ConcVal[t:]) * currDxs)
        afterTrans[i, -1] = np.exp(ValPerLayAfter)
        preTrans[i, -1] = 1

    return preTrans + afterTrans


def forward_substitution(L, b):
    # Get number of rows
    n = L.shape[0]

    # Allocating space for the solution vector
    y = np.zeros_like(b, dtype=np.double);

    # Here we perform the forward-substitution.
    # Initializing  with the first row.
    y[0] = b[0] / L[0, 0]

    # Looping over rows in reverse (from the bottom  up),
    # starting with the second to last row, because  the
    # last row solve was completed in the last step.
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y

def back_substitution(U, y):
    # Number of rows
    n = U.shape[0]

    # Allocating space for the solution vector
    x = np.zeros_like(y, dtype=np.double)

    # Here we perform the back-substitution.
    # Initializing with the last row.
    x[-1] = y[-1] / U[-1, -1]

    # Looping over rows in reverse (from the bottom up),
    # starting with the second to last row, because the
    # last row solve was completed in the last step.
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]

    return x


def lu_solve(L, U, b):

    y = forward_substitution(L, b)

    return back_substitution(U, y)


def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T, A) + l * L
    # Bu, Bs, Bvh = np.linalg.svd(B)
    upL = scipy.linalg.cholesky(B)
    # return np.sum(np.log(Bs))
    return 2 * np.sum(np.log(np.diag(upL)))

def f(ATy, y, B_inv_A_trans_y):
    return np.matmul(y[0::, 0].T, y[0::, 0]) - np.matmul(ATy[0::, 0].T, B_inv_A_trans_y)