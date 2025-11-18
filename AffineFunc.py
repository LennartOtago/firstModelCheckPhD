import numpy as np
from scipy import constants
import pandas as pd
import math
import scipy as scy

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

def MinLogMargPost(params, A, y, L, ATA, ATy, betas):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    lamb = params[1]
    betaG, betaD  = betas
    #print(lamb)
    if lamb < 0  or gam < 0:
        return np.nan

    m, n = A.shape

    #print(lamb)
    Bp = ATA + lamb * L

    LowTri = np.linalg.cholesky(Bp)
    UpTri = LowTri.T
    # check if L L.H = B
    B_inv_A_trans_y = scy.linalg.cho_solve((LowTri,True),  ATy[:,0])

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)


def CurrMarg(lamb, gam,G, F, n, m, betaG, betaD):

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)



def f(ATy, y, B_inv_A_trans_y):
    return np.matmul(y[:,0].T, y[:,0]) - np.matmul(ATy[:,0].T,B_inv_A_trans_y)

def gen_trap_rul(dxs):
    #val = np.zeros(len(dxs)+1)
    sumMat = np.eye(len(dxs)+1)
    Ones = np.ones((len(dxs)+1,len(dxs)+1))
    sumMat = sumMat + np.triu(Ones,1) - np.triu(Ones,2)
    return 0.5*(dxs @ np.copy(sumMat[:-1,:]))


def calcNonLin(tang_heights, dxs,  height_values, pressure_values, ind, temp_values, VMR_O3, wvnmbr, S, E,g_doub_prime,g_prime):
    '''careful that A_lin is just dx values
    maybe do A_lin_copy = np.copy(A_lin/2)
    A_lin_copy[:,-1] = A_lin_copy[:,-1] * 2
    if A_lin has been generated for linear data'''

    # from : https://hitran.org/docs/definitions-and-units/
    # all calc in CGS
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0] # in cm^-1

    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / temp_values) + g_prime[ind, 0] * np.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / temp_values)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296) + g_prime[ind, 0] * np.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / temp_values) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / temp_values)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))



    num_mole = 1 / constants.Boltzmann
    # 1e-4 cm^2/molec to m^2/molec
    theta = num_mole * VMR_O3 * S[ind,0] * 1e-4
    # 1e2 for pressure hPa to Pa and 1e5 for km to m
    ConcVal = - pressure_values * 1e2 * LineIntScal / temp_values * theta * 1e3

    SpecNumMeas = len(tang_heights)
    SpecNumLayers = len(VMR_O3)

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


def composeAforO3(A_lin, temp, press, ind, wvnmbr, g_doub_prime, g_prime, E, S):

    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0] # in cm^-1


    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / temp) + g_prime[ind, 0] * np.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296) + g_prime[ind, 0] * np.exp(
        - HitrConst2 * (E[ind, 0] + v_0) / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3
    C2 = constants.h * constants.c * 1e2 * v_0 / (constants.Boltzmann * temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1)) # in W m^2/cm^3/sr
    # for number density of air molec / m^3 and 1e2 for pressure values from hPa to Pa
    num_mole = press * 1e2 / (constants.Boltzmann * temp)
    kmTom = 1e3  # for dx integration
    # 1e4 for W cm/cm^2 to W cm/m^2 and S[ind, 0] in cm^2 / molec
    theta_scale = num_mole * 1e4 * S[ind, 0] * kmTom

    A_scal = LineIntScal * Source * theta_scale

    A = A_lin * A_scal.T

    return A, 1


def genDataFindandtestMap(tang_heights_lin, A_lin_dx,  height_values, gamma0, newCondMean, testO3Prof, A_lin, temp_values, pressure_values, ind, relMapErrDat, wvnmbr, S, E,g_doub_prime, g_prime):
    '''Find map from linear to non-linear data'''

    SpecNumMeas, SpecNumLayers = A_lin.shape
    relMapErr = np.copy(relMapErrDat)
    while np.max(relMapErr) >= relMapErrDat:
        Results = np.copy(testO3Prof)
        Results[0] = newCondMean
        testDat = SpecNumMeas
        A_O3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, wvnmbr, g_doub_prime, g_prime, E, S)

        # Results = np.zeros((SampleRounds, SpecNumLayers))
        LinDataY = np.zeros((testDat, SpecNumMeas))
        NonLinDataY = np.zeros((testDat, SpecNumMeas))
        for test in range(testDat):
            #print(test)
            ProfRand = test#np.random.randint(low=0, high=SampleRounds)
            # Results = np.loadtxt(f'Res{testSet}.txt')

            O3_Prof = np.copy(Results[ProfRand]).reshape((SpecNumLayers,1))
            #O3_Prof[O3_Prof < 0] = 0
            nonLinA = calcNonLin(tang_heights_lin, A_lin_dx,  height_values, pressure_values, ind, temp_values, O3_Prof, wvnmbr, S, E, g_doub_prime, g_prime)
            noise = np.random.normal(0, np.sqrt(1 / gamma0), SpecNumMeas)
            # noise = np.zeros(SpecNumMeas)

            #LinDataY[test] = np.matmul( currMap @ (A_O3 * 2), O3_Prof.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(SpecNumMeas) #+ noise
            LinDataY[test] = np.matmul((A_O3 * 2),
                                       O3_Prof.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
                SpecNumMeas)
            NonLinDataY[test] = np.matmul(A_O3 * nonLinA,
                                          O3_Prof.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
                SpecNumMeas) #+ noise
            #currMap = np.eye(SpecNumMeas)



        RealMap = LinModelSolve(LinDataY, NonLinDataY, SpecNumMeas)
        currMap = np.copy(RealMap) #@ np.copy(currMap)

        #test map do find error
        testNum = 10# len(Results)#100
        testO3 = np.copy(Results)#np.random.multivariate_normal(VMR_O3.reshape(SpecNumLayers), 1e-18 * L_d, size=testNum)
        #testO3[testO3 < 0] = 0

        testDataY = np.zeros((testNum, SpecNumMeas))
        testNonLinY = np.zeros((testNum, SpecNumMeas))

        for k in range(0, testNum):
            currO3 = testO3[k].reshape((SpecNumLayers,1))
            noise = np.random.normal(0, np.sqrt(1 / gamma0), SpecNumMeas)
            nonLinA = calcNonLin(tang_heights_lin, A_lin_dx,  height_values, pressure_values, ind, temp_values, currO3 , wvnmbr, S, E, g_doub_prime, g_prime)

            testDataY[k] = np.matmul(currMap @ (A_O3 * 2), currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(SpecNumMeas)# + noise
            testNonLinY[k] = np.matmul(A_O3 * nonLinA, currO3.reshape((SpecNumLayers, 1)) * theta_scale_O3).reshape(
                SpecNumMeas) #+ noise
            #testNonLinY[k] = testDataY[k] #+ noise


        relMapErr = testSolvedMap(testNum, testNonLinY, testDataY)
        print('max realtive Error: ' + str(np.max(relMapErr)))

        #relMapErr = 1
    return currMap, relMapErr, LinDataY, NonLinDataY, testO3

def LinModelSolve(LinDataY, NonLinDataY, SpecNumMeas):
    # basis
    RealMap = None
    while RealMap is None:
        try:

            RealMap = np.zeros((SpecNumMeas, SpecNumMeas))

            for i in range(0, SpecNumMeas):
                RealMap[i,:] = np.linalg.solve(LinDataY, NonLinDataY[:, i])
                #RealMap[i, :] = np.linalg.solve(NonLinDataY, LinDataY[:, i])

        except np.linalg.LinAlgError:
            RealMap = None
            print('pass')
            pass

    return RealMap

def testSolvedMap(testNum, testNonLinY, testDataY):
    relMapErr = np.zeros(testNum)
    for k in range(0, testNum):


        mappedDat = testDataY[k]
        relMapErr[k] = np.linalg.norm(testNonLinY[k] - mappedDat) / np.linalg.norm(testNonLinY[k]) * 100


    return relMapErr
