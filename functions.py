from scipy.special import wofz
import numpy as np
from scipy import constants
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math
from scipy.sparse.linalg import gmres


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


def get_temp_values(height_values):
    """ used to be based on the ISA model see omnicalculator.com/physics/altitude-temperature
    now https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html """
    temp_values = np.zeros(len(height_values))
    #temp_values[0] = 288.15#15 - (height_values[0] - 0) * 6.49 + 273.15
    ###calculate temp values
    for i in range(0, len(height_values)):
        if height_values[i] < 11:
            temp_values[i] = np.around(15.04 - height_values[i] * 6.49 + 273.15,2)
        if 11 <= height_values[i] < 25:
            temp_values[i] = np.around(-55.46 + 273.15,2)
        if 25 <= height_values[i] :
            temp_values[i] = np.around(-131.21 + height_values[i] * 2.99 + 273.15,2)

    return temp_values.reshape((len(height_values),1))


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


# def add_noise(Ax, percent):
#     return Ax + np.random.normal(0, percent * np.max(Ax), (len(Ax), 1))


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
    # Calculate signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    # Calculate noise power based on SNR (in percent)
    #noise_power = signal_power / (100 / snr_percent)
    noise_power = signal_power / snr

    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal, 1/noise_power

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

'''Generate Forward Map A
where each collum is one measurment defined by a tangent height
every entry is length in km each measurement goes through
first non-zero entry of each row is the lowest layer (which should have a Ozone amount of 0)
last entry of each row is highest layer (also 0 Ozone)'''
def gen_forward_map(meas_ang, height, obs_height, R):
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)
    num_meas = len(tang_height)
    # add one extra layer so that the difference in height can be calculated
    layers = np.zeros(len(height)+1)
    layers[0:-1] = height
    layers[-1] = height[-1] + (height[-1] - height[-2])/2


    A_height = np.zeros((num_meas, len(layers)-1))
    t = 0
    for m in range(0, num_meas):

        while layers[t] <= tang_height[m]:

            t += 1

        # first dr
        A_height[m, t - 1] = 0.5 * np.sqrt((layers[t] + R) ** 2 - (tang_height[m] + R) ** 2)
        dr = 2 * A_height[m, t - 1]
        for i in range(t, len(layers) - 1):
            A_height[m, i] = np.sqrt((layers[i + 1] + R) ** 2 - (tang_height[m] + R) ** 2) - dr
            dr = dr + A_height[m, i]
        A_height[m, i] = 0.5 * A_height[m, i]
    return 2 * A_height, tang_height, layers[-1]


def f(ATy, y, B_inv_A_trans_y):

    return np.matmul(y[0::,0].T, y[0::,0]) - np.matmul(ATy[0::,0].T,B_inv_A_trans_y)


def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T,A) + l * L
    Bu, Bs, Bvh = np.linalg.svd(B)
    # np.log(np.prod(Bs))
    return np.sum(np.log(Bs))

def f_tayl( delta_lam, f_0, f_1, f_2, f_3, f_4):
    """calculate taylor series for """



    return f_0 + f_1 * delta_lam + f_2 * delta_lam**2 + f_3 * delta_lam**3 + f_4 * delta_lam**4# + f_5 * delta_lam**5 #- f_6 * delta_lam**6

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