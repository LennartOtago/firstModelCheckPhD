from scipy.special import wofz
import numpy as np
from scipy import constants
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math
from scipy.sparse.linalg import gmres
import scipy as scy



def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))


# voigt function as real part of Faddeeva function
def V(x, sigma, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    # sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j * gamma) / sigma / np.sqrt(2))) / (sigma * np.sqrt(2 * np.pi))


def Lorenz(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x ** 2 + gamma ** 2)


def G(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha \
           * np.exp(-(x / alpha) ** 2 * np.log(2))


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

def get_temp_values(height_values):
    """ used to be based on the ISA model see omnicalculator.com/physics/altitude-temperature
    now https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html """
    temp_values = np.zeros(len(height_values))
    #temp_values[0] = 288.15#15 - (height_values[0] - 0) * 6.49 + 273.15
    ###calculate temp values
    for i in range(0, len(height_values)):
        if height_values[i] <= 11:
            temp_values[i] = np.around(- height_values[i] * 6.49 + 288.15,2)
        if 11 < height_values[i] <= 25:
            temp_values[i] = np.around(216.76,2)
        if 20 < height_values[i] <= 32:
            temp_values[i] = np.around(216.76 + (height_values[i] - 20) * 1, 2)
        if 32 < height_values[i] <= 47:
            temp_values[i] = np.around(228.76 + (height_values[i] - 32) * 2.8, 2)
        if 47 < height_values[i] <= 51:
            temp_values[i] = np.around(270.76, 2)
        if 51 < height_values[i] <= 71:
            temp_values[i] = np.around(270.76 - (height_values[i] - 51) * 2.8, 2)
        if 71 < height_values[i] <= 85:
            temp_values[i] = np.around(214.76 - (height_values[i] - 71) * 2.0, 2)
        if 85 < height_values[i]:
            temp_values[i] = 186.8

    return temp_values.reshape((len(height_values),1))
def get_temp(height_value):
    """ used to be based on the ISA model see omnicalculator.com/physics/altitude-temperature
    now https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html """
    #temp_values[0] = 288.15#15 - (height_values[0] - 0) * 6.49 + 273.15
    #calculate temp values

    if height_value < 11:
        temp_value = np.around(- height_value * 6.49 + 288.15,2)
    if 11 <= height_value < 20:
        temp_value = np.around(216.76 ,2)
    if 20 <= height_value < 32:
        temp_value = np.around(216.76 + (height_value - 20) * 1,2)
    if 32 <= height_value < 47:
        temp_value = np.around(228.76 + (height_value - 32)* 2.8, 2)
    if 47 <= height_value < 51:
        temp_value = np.around(270.76, 2)
    if 51 <= height_value < 71:
        temp_value = np.around(270.76 - (height_value - 51) * 2.8 ,2)
    if 71 <= height_value < 84.852:
        temp_value = np.around(214.76 - (height_value -71) * 2.0 ,2)
    if  84.852 <= height_value :
        temp_value = 186.8

    return temp_value



def gen_measurement(meas_ang, layers, w_cross, VMR_O3, P, T, Source, obs_height=300):
    '''generates Measurement given the input measurement angels and depending on the model layers in km
    obs_height is given in km
    '''

    # exclude first layer at h = 0  and
    # last layer at h = Observer
    min_ind = 1
    max_ind = -1
    layers = layers[min_ind: max_ind]

    w_cross = w_cross[min_ind: max_ind - 1]
    VMR_O3 = VMR_O3[min_ind: max_ind - 1]
    Source = Source[min_ind: max_ind - 1]
    P = P[min_ind: max_ind - 1]
    T = T[min_ind: max_ind - 1]

    R = 6371
    # get tangent height for each measurement layers[0:-1] #
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)
    num_meas = len(tang_height)
    # get dr's for measurements of different layers
    A_height = np.zeros((num_meas, len(layers) - 1))
    t = 0
    for m in range(0, num_meas):

        while layers[t] <= tang_height[m]:
            t += 1
        print(t)
        # first dr
        A_height[m, t - 1] = np.sqrt((layers[t] + R) ** 2 - (tang_height[m] + R) ** 2)
        dr = A_height[m, t - 1]
        for i in range(t, len(layers) - 1):
            # A_height[j,i] =  (height_values[j+i+1] + R)/np.sqrt((height_values[j+i+1]+ R)**2 - (height_values[j]+ R)**2 ) * d_height[j+i]
            A_height[m, i] = np.sqrt((layers[i + 1] + R) ** 2 - (tang_height[m] + R) ** 2) - dr
            dr = dr + A_height[m, i]
    # calc mearuements

    R_gas = constants.Avogadro * constants.Boltzmann * 1e7  # in ..cm^3
    # caculate number of molecules in one cm^3
    num_mole = (P / (constants.Boltzmann * 1e7 * T))

    THETA = (num_mole * w_cross * VMR_O3 * Source)
    # 2 * A_height * 1e5....2 * np.matmul(A_height*1e5, THETA[1::]) A_height in km
    # * 1e5 converts to cm
    return 2 * np.matmul(A_height, THETA), 2 * A_height, THETA, tang_height

def composeAforO3(A_lin, temp, press, ind, set_temp):

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

    for i, lines in enumerate(data_set):
        wvnmbr[i] = float(lines[0][5:15])  # in 1/cm
        S[i] = float(lines[0][16:25])  # in cm/mol
        F[i] = float(lines[0][26:35])
        g_air[i] = float(lines[0][35:40])
        g_self[i] = float(lines[0][40:45])
        E[i] = float(lines[0][46:55])
        n_air[i] = float(lines[0][55:59])
        g_doub_prime[i] = float(lines[0][155:160])

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
    A_scal = press.reshape((SpecNumLayers, 1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm / (set_temp)
    theta_scale = num_mole *  f_broad * 1e-4 * scalingConst * S[ind, 0]
    A = A_lin * A_scal.T
    #np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
    return A, theta_scale

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
    # #sumMat[-2,-1] = 0
    #
    # val[0] = dxs[0] * 0.5
    # for i in range(1,len(dxs)):
    #     val[i] = (dxs[i-1] + dxs[i])* 0.5
    #
    # val[-1] = dxs[-1] * 0.5
    # return val

# def gen_trap_rul(dxs):
#     val = np.zeros(len(dxs)+1)
#     val[0] = dxs[0] * 0.5
#     for i in range(1,len(dxs)):
#         val[i] = (dxs[i-1] + dxs[i])* 0.5
#
#     val[-1] = dxs[-1] * 0.5
#     return val

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
def add_noise(signal, snr):
    """
    Add noise to a signal based on the specified SNR (in percent).

    Parameters:
        signal: numpy array
            The original signal.
        snr_percent: float
            The desired signal-to-noise ratio in percent.

    Returns:
        numpy array
            The signal with added noise.
    """
    # Calculate root mean square of signal
    signal_power = np.sqrt(np.mean(np.abs(signal) ** 2))

    # Calculate noise power based on SNR (in percent)
    noise_power = signal_power / snr

    # Generate noise
    noise = np.random.normal(0, noise_power, signal.shape)

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal, 1/noise_power**2

def corr_add_noise(Ax, percent):
    return Ax + np.random.normal(0, percent * np.sqrt(np.max(Ax)**2/2), (len(Ax), 1))

# def old_add_noise(Ax, percent):
#     return Ax + np.random.normal(0, percent * np.max(Ax), (len(Ax), 1))

def plot_svd(ATA, height_values):
    '''
    we plot left singular vectors wighted with the singular value
    for symmetric sqaure matrix
    '''
    ATAu, ATAs, ATAvh = np.linalg.svd(ATA)

    # Create figure
    fig = go.Figure()
    # k_values = int(np.linspace(0, len(As)-1, len(As)))

    # Add traces, one for each slider step
    for k in range(0, len(ATAs)):
        x = height_values  # np.linspace(0, len(Au[:, k]) - 1, len(Au[:, k]))
        y = ATAu[:, k]  # *As[k]
        df = pd.DataFrame(dict(x=x, y=y))

        fig.add_trace(
            go.Scatter(
                x=df['x'],
                y=df['y'],
                visible=False,
                line=dict(color="#00CED1", width=6),
                name=f"index = {k}"
            )
        )

    # Make 10th trace visible
    fig.data[10].visible = True
    k = np.linspace(0, len(ATAs) - 1, len(ATAs))

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider at tangent model layer: " + str(height_values[i]) + " in m"}],
            label=str(height_values[i]),  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "index= ", "suffix": ""},
        pad={"b": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title="Left Singlar Vectors weighted with Singular Values",
        xaxis_title="height values"
    )
    fig.update_yaxes(range=[np.min(ATAu), np.max(ATAu)])

    fig.show()

    fig.write_html('SVD.html')
    return ATAu, ATAs, ATAvh


''' generate dx'''
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


def f(ATy, y, B_inv_A_trans_y):

    return np.matmul(y[0::,0].T, y[0::,0]) - np.matmul(ATy[0::,0].T,B_inv_A_trans_y)


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

def g_MC_log_det(B_inv_L, num_sam):
    # calc trace of B_inv_L with monte carlo estiamtion
    # do 4 times as colin
    trace_Bs = np.zeros(num_sam)
    for k in range(num_sam):
        z = np.random.randint(2, size=len(B_inv_L))
        z[z == 0] = -1
        trace_Bs[k] = np.matmul(z.T, np.matmul(B_inv_L, z))

    return trace_Bs

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

def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',**kw))
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',va='bottom',rotation=90,**kw))
                     for text,color in zip(list_of_strings[::-1],list_of_colors) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.13, 0.2),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)

def neg_log_likehood(gamma,y, Ax):
    n = len(Ax)
    return - n/2 * np.log(gamma) + gamma/2 * np.matmul((y-Ax).T,(y-Ax))


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


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


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


def composeAforPress(A_lin, temp, O3, ind):
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

    for i, lines in enumerate(data_set):
        wvnmbr[i] = float(lines[0][5:15])  # in 1/cm
        S[i] = float(lines[0][16:25])  # in cm/mol
        F[i] = float(lines[0][26:35])
        g_air[i] = float(lines[0][35:40])
        g_self[i] = float(lines[0][40:45])
        E[i] = float(lines[0][46:55])
        n_air[i] = float(lines[0][55:59])
        g_doub_prime[i] = float(lines[0][155:160])

    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]

    f_broad = 1
    w_cross = f_broad * 1e-4 * O3
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
    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    # take linear
    num_mole = 1 / (constants.Boltzmann)  # * temp_values)

    AscalConstKmToCm = 1e3
    # 1e2 for pressure values from hPa to Pa
    A_scal = 1e2 * LineIntScal * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers, 1)) * scalingConst * S[ind, 0] * num_mole / temp

    A = A_lin * A_scal.T
    #np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
    return A, 1

def composeAforTemp(A_lin, press, O3, ind, old_temp):
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

    for i, lines in enumerate(data_set):
        wvnmbr[i] = float(lines[0][5:15])  # in 1/cm
        S[i] = float(lines[0][16:25])  # in cm/mol
        F[i] = float(lines[0][26:35])
        g_air[i] = float(lines[0][35:40])
        g_self[i] = float(lines[0][40:45])
        E[i] = float(lines[0][46:55])
        n_air[i] = float(lines[0][55:59])
        g_doub_prime[i] = float(lines[0][155:160])

    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]

    f_broad = 1
    w_cross = f_broad * 1e-4 * O3
    scalingConst = 1e11
    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / old_temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / old_temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / old_temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3 * 1e8
    C2 = constants.h * constants.c * 1e2 * v_0 / (constants.Boltzmann * old_temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1))
    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    # take linear
    num_mole = 1 / (constants.Boltzmann)  # * temp_values)

    AscalConstKmToCm = 1e3
    # 1e2 for pressure values from hPa to Pa
    A_scal = 1e2 * LineIntScal * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers, 1)) * scalingConst * S[ind, 0] * num_mole * press.reshape((SpecNumLayers, 1))

    A = A_lin * A_scal.T
    #np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
    return A, 1

def composeAforTempPress(A_lin, O3, ind, old_temp):
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

    for i, lines in enumerate(data_set):
        wvnmbr[i] = float(lines[0][5:15])  # in 1/cm
        S[i] = float(lines[0][16:25])  # in cm/mol
        F[i] = float(lines[0][26:35])
        g_air[i] = float(lines[0][35:40])
        g_self[i] = float(lines[0][40:45])
        E[i] = float(lines[0][46:55])
        n_air[i] = float(lines[0][55:59])
        g_doub_prime[i] = float(lines[0][155:160])

    # from : https://hitran.org/docs/definitions-and-units/
    HitrConst2 = 1.4387769  # in cm K
    v_0 = wvnmbr[ind][0]

    f_broad = 1
    w_cross = f_broad * 1e-4 * O3
    scalingConst = 1e11
    Q = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / old_temp)
    Q_ref = g_doub_prime[ind, 0] * np.exp(- HitrConst2 * E[ind, 0] / 296)
    LineIntScal = Q_ref / Q * np.exp(- HitrConst2 * E[ind, 0] / old_temp) / np.exp(- HitrConst2 * E[ind, 0] / 296) * (
                1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / old_temp)) / (
                              1 - np.exp(- HitrConst2 * wvnmbr[ind, 0] / 296))

    C1 = 2 * constants.h * constants.c ** 2 * v_0 ** 3 * 1e8
    C2 = constants.h * constants.c * 1e2 * v_0 / (constants.Boltzmann * old_temp)
    # plancks function
    Source = np.array(C1 / (np.exp(C2) - 1))
    SpecNumMeas, SpecNumLayers = np.shape(A_lin)
    # take linear
    num_mole = 1 / (constants.Boltzmann)  # * temp_values)

    AscalConstKmToCm = 1e3
    # 1e2 for pressure values from hPa to Pa
    A_scal = 1e2 * LineIntScal * Source * AscalConstKmToCm * w_cross.reshape((SpecNumLayers, 1)) * scalingConst * S[ind, 0] * num_mole

    A = A_lin * A_scal.T
    #np.savetxt('AMat.txt', A, fmt='%.15f', delimiter='\t')
    return A, 1

