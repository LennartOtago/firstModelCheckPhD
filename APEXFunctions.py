from scipy.special import wofz
import numpy as np
from scipy import constants
import pandas as pd
import math
import scipy as scy

def f(ATy, y, B_inv_A_trans_y):
    return np.matmul(y[:,0].T, y[:,0]) - np.matmul(ATy[:,0].T,B_inv_A_trans_y)


def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))




def generate_L(neigbours):
    #Dirichlet Boundaries
    siz = int(np.size(neigbours, 0))
    neig = np.size(neigbours, 1)
    L = np.zeros((siz, siz))

    for i in range(0, siz):
        L[i, i] = neig
        for j in range(0, neig):
            if ~np.isnan(neigbours[i, j]):
                L[i, int(neigbours[i, j])] = -1
    #non periodic boundaries Neumann
    # L[0,0] = 1
    # L[-1,-1] = 1
    #periodic boundaires
    # L[0,-1] = -1
    # L[-1,0] = -1

    return L

def temp_func(x):
    h0 = 11
    h1 = 20.1
    h2 = 32.2
    h3 = 47.4
    h4 = 51.4
    h5 = 71.8
    h6 = 86
    a0 = -6.5
    a1 = 1
    a2 = 2.8
    a3 = -2.8
    a4 = -2
    b0 = 288.15

    a = np.ones(x.shape)
    b = np.ones(x.shape)

    a[x < h0] = a0
    a[h0 <= x] = 0
    a[h1 <= x] = a1
    a[h2 <= x] = a2
    a[h3 <= x] = 0
    a[h4 <= x ] = a3
    a[h5 <= x ] = a4
    a[h6 <= x ] = 0

    b[x < h0] = b0
    b[h0 <= x] = b0 + h0 * a0
    b[h1 <= x] = b0 + h0 * a0
    b[h2 <= x] = a1 * (h2-h1) + b0 + h0 * a0
    b[h3 <= x ] = a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    b[h4 <= x ] = a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    b[h5 <= x ] = a3 * (h5-h4) + a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0
    b[h6 <= x ] = a4 * (h6-h5) + a3 * (h5-h4) + a2 * (h3-h2) + a1 * (h2-h1) + b0 + h0 * a0

    h = np.ones(x.shape)
    h[x < h0] = 0
    h[h0 <= x] = h0
    h[h1 <= x] = h1
    h[h2 <= x] = h2
    h[h3 <= x] = h3
    h[h4 <= x] = h4
    h[h5 <= x] = h5
    h[h6 <= x] = h6
    return a * (x - h) + b


def composeAforO3(A_lin, temp, press, ind):

    files = '634f1dc4.par'  # /home/lennartgolks/Python /Users/lennart/PycharmProjects

    my_data = pd.read_csv(files, header=None)
    data_set = my_data.values

    size = data_set.shape
    wvnmbr = np.zeros((size[0], 1))
    S = np.zeros((size[0], 1))
    F = np.zeros((size[0], 1))
    g_air = np.zeros((size[0], 1))
    g_self = np.zeros((size[0], 1))
    E = np.zeros((size[0], 1))
    n_air = np.zeros((size[0], 1))
    g_doub_prime = np.zeros((size[0], 1))
    g_prime = np.zeros((size[0], 1))

    for i, lines in enumerate(data_set):
        wvnmbr[i] = float(lines[0][5:15])  # in 1/cm
        S[i] = float(lines[0][16:25])  # in cm/mol
        F[i] = float(lines[0][26:35])
        g_air[i] = float(lines[0][35:40])
        g_self[i] = float(lines[0][40:45])
        E[i] = float(lines[0][46:55])
        n_air[i] = float(lines[0][55:59])
        g_doub_prime[i] = float(lines[0][148:153])
        g_prime[i] = float(lines[0][155:160])

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
    theta_scale = num_mole * S[ind, 0] * 1e4  * kmTom

    A_scal = LineIntScal * Source * theta_scale

    A = A_lin * A_scal.T

    return A,  1
'''generate forward map accoring to trapezoidal rule'''
def gen_sing_map(dxs, tang_heights, heights):
    m,n = dxs.shape
    A_lin = np.zeros((m,n+1))
    for i in range(0,m):
        t = 0
        while heights[t] <= tang_heights[i]:
            t += 1
        A_lin[i, t - 1:] = gen_trap_rul(dxs[i, t - 1:])
        # A_lin[i, t-1] = 0.5 * dxs[i, t-1]
        # for j in range(t, n):
        #     A_lin[i,j] = 0.5 * (dxs[i,j-1] + dxs[i,j])
        # A_lin[i, -1] = 0.5 * dxs[i, -1]
    return A_lin


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

def gen_forward_map(meas_ang, heights, obs_height, R):
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)
    num_meas = len(tang_height)
    A_height = np.zeros((num_meas, len(heights)-1))

    for m in range(0, num_meas):
        t = 0
        #find t so that layers[t] is larger than tang height
        while heights[t] < tang_height[m]:
            t += 1

        for i in range(t, len(heights)):
            A_height[m, i-1] = np.sqrt((heights[i] + R) ** 2 - (tang_height[m] + R) ** 2) - np.sum( A_height[m, :i])

    return A_height, tang_height, heights[-1]


def add_noise_Blokk(Ax,SNR):
    stdNoise = max(Ax)/SNR
    return Ax + np.random.normal(0,stdNoise , (len(Ax), 1)) , 1/stdNoise**2


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

def g_tayl(delta_lam, g_0, trace_B_inv_L_1, trace_B_inv_L_2, trace_B_inv_L_3, trace_B_inv_L_4, trace_B_inv_L_5, trace_B_inv_L_6):

    # trace_B_inv_L_1 = np.mean(g_MC_log_det(B_inv_L, num_sam))
    # trace_B_inv_L_2 = np.mean(g_MC_log_det(np.matmul(B_inv_L, B_inv_L), num_sam))
    # trace_B_inv_L_3 = np.mean(g_MC_log_det(np.matmul(np.matmul(B_inv_L, B_inv_L), B_inv_L), num_sam))
    # trace_B_inv_L_4 = np.mean(g_MC_log_det(np.matmul(np.matmul(np.matmul(B_inv_L, B_inv_L), B_inv_L),B_inv_L) ,num_sam))
    #
    # return trace_B_inv_L_1 * delta_lam -  trace_B_inv_L_2 / 2 * delta_lam**2 + trace_B_inv_L_3 / 6 * delta_lam**3 - trace_B_inv_L_4 / 24 * delta_lam**4
    # trace_B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
    # trace_B_inv_L_3 = np.matmul(np.matmul(B_inv_L, B_inv_L), B_inv_L)
    # trace_B_inv_L_4 = np.matmul(np.matmul(np.matmul(B_inv_L, B_inv_L), B_inv_L), B_inv_L)
    # trace_B_inv_L_5 = np.matmul(np.matmul(np.matmul(np.matmul(B_inv_L, B_inv_L), B_inv_L), B_inv_L), B_inv_L)
    # trace_B_inv_L_2 = np.trace(B_inv_L_2)
    # trace_B_inv_L_3 = np.trace(B_inv_L_3)
    # trace_B_inv_L_4 = np.trace(B_inv_L_4)
    # trace_B_inv_L_5 = np.trace(B_inv_L_5)
    return g_0 + trace_B_inv_L_1 * delta_lam + trace_B_inv_L_2 * delta_lam**2 + trace_B_inv_L_3 * delta_lam**3 + trace_B_inv_L_4 * delta_lam**4 + trace_B_inv_L_5 * delta_lam**5 + trace_B_inv_L_6 * delta_lam**6


def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T,A) + l * L
    #Bu, Bs, Bvh = np.linalg.svd(B)
    upL = scy.linalg.cholesky(B)
    #return np.sum(np.log(Bs))
    return 2* np.sum(np.log(np.diag(upL)))

def f_tayl( delta_lam, f_0, f_1, f_2, f_3, f_4, f_5, f_6):
    """calculate taylor series for """

    return f_0 + f_1 * delta_lam + f_2 * delta_lam**2 + f_3 * delta_lam**3 + f_4 * delta_lam**4 + f_5 * delta_lam**5 + f_6 * delta_lam**6


def MHwG(number_samples, burnIn, lam0, gamma0, f_0, g_0, fTaylor, betas, A, L):
    wLam = lam0 * 0.8#8e3#7e1
    betaG, betaD  = betas
    alphaG = 1
    alphaD = 1
    f_0_1, f_0_2, f_0_3, f_0_4 = fTaylor

    NumMeas, NumLay = A.shape
    k = 0

    gammas = np.zeros(number_samples + burnIn)
    #deltas = np.zeros(number_samples + burnIn)
    lambdas = np.zeros(number_samples + burnIn)

    gammas[0] = gamma0
    lambdas[0] = lam0
    lamMax = 1.25 * lam0
    lamMin = 0.75 * lam0
    delG = (g(A, L, lamMax) - g(A, L, lamMin))/ (np.log(lamMax) - np.log(lamMin))


    shape = NumMeas / 2 + alphaD + alphaG
    rate = f_0 / 2 + betaG + betaD * lam0

    #f_new = np.copy(f_0)
    #rate_old = np.copy(rate)
    for t in range(number_samples + burnIn-1):
        #print(t)

        # # draw new lambda
        lam_p = np.random.normal(lambdas[t], wLam)

        while lam_p < 0:#or lam_p > univarGridO3[1][-1]:
               lam_p = np.random.normal(lambdas[t], wLam)

        delta_lam = lam_p - lambdas[t]
        delta_lam_t = lambdas[t] - lam0
        delta_lam_p = lam_p - lam0

        delta_f = f_0_1 * delta_lam + f_0_2 * (delta_lam_p**2 - delta_lam_t**2) + f_0_3 *(delta_lam_p**3 - delta_lam_t**3) #+ f_0_4 * (delta_lam_p**4 - delta_lam_t**4) #+ f_0_5 * delta_lam**5
        #delta_g = g_0_1 * delta_lam + g_0_2 * (delta_lam_p**2 - delta_lam_t**2) + g_0_3 * (delta_lam_p**3 - delta_lam_t**3) #+ g_0_4 * (delta_lam_p**4 - delta_lam_t**4) #+ g_0_5 * delta_lam**5
        #delta_g = g(A, L, lam_p) - g(A, L, lambdas[t])

        Glam_p  = (np.log(lam_p) - np.log(lam0)) * delG + g_0

        Gcurr = (np.log(lambdas[t]) - np.log(lam0)) * delG + g_0

        # taylorG = g_tayl(lamb - minimum[1], g_0, g_0_1, g_0_2, g_0_3, g_0_5, 0 ,0)
        # taylorG = g(A, L, lamb)
        #taylorG = np.exp(GApprox)
        delta_g = Glam_p - Gcurr
        #delta_g = g(A, L, lam_p) - g(A, L, lambdas[t])
        log_MH_ratio = (NumLay/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

        #accept or rejeict new lam_p
        u = np.random.uniform()

        if np.log(u) <= np.min(log_MH_ratio,0):
            #accept
            k = k + 1
            lambdas[t + 1] = lam_p
            #only calc when lambda is updated
            #f_old = np.copy(f_new)
            #rate_old = np.copy(rate)
            #f_new = f_0 + delta_f
            #B = (ATA + lam_p * L)
            #LowTri = np.linalg.cholesky(B)
            #UpTri = LowTri.T
            #B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

            #f_new = f(ATy, y, B_inv_A_trans_y)
            delta_lam_p = lam_p - lam0
            delta_f = f_0_1 * delta_lam_p + f_0_2 * delta_lam_p ** 2 + f_0_3 * delta_lam_p ** 3#+ f_0_4 * delta_lam_p ** 4
            f_new = f_0 + delta_f
            #f_new = np.copy( f_p)
            # g_old = np.copy(g_new)
            rate = f_new / 2 + betaG + betaD * lam_p  # lambdas[t+1]
            if rate <= 0:
                print('scale < 0')
        else:
            #rejcet
            lambdas[t + 1] = np.copy(lambdas[t])


        gammas[t+1] = np.random.gamma(shape = shape, scale = 1/rate)

        #deltas[t+1] = lambdas[t+1] * gammas[t+1]

    return lambdas, gammas,k

#find minimum for first guesses
'''params[1] = delta
params[0] = gamma'''
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
