import numpy as np
import matplotlib as mpl

from importetFunctions import *
import time
import pickle as pl
#import matlab.engine
from functions import *
from errors import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt


import pandas as pd
from numpy.random import uniform, normal, gamma
import scipy as scy
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


""" for plotting figures,
PgWidth in points, either collumn width page with of Latex"""
def scientific(x, pos):
    # x:  tick value
    # pos: tick position
    return '%.e' % x
scientific_formatter = FuncFormatter(scientific)

fraction = 1.5


dpi = 300

PgWidthPt = 245
PgWidthPt =  fraction * 421/2 #phd
n_bins = 20
burnIn = 50
betaG = 1e-4
betaD = 1e-10  # 1e-4
#Colors
#pyTCol = [230/255,159/255, 0/255]
pyTCol = [213/255,94/255, 0/255]
#pyTCol = [240/255, 228/255, 66/255]
MTCCol = 'k'
dataCol = [225/255, 190/255, 106/255]
#dataCol =[230/255,159/255, 0/255]
regCol = [212/255, 17/255, 89/255]
#MargCol = [86/255, 180/255, 233/255]
MargCol = [255/255, 194/255, 10/255]
defBack = mpl.get_backend()
mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': fraction * 12,
                     'text.usetex': True,
                     'font.family' : 'serif',
                     'font.serif'  : 'cm',
                     'text.latex.preamble': r'\usepackage{bm, amsmath}'})


ResCol = "#1E88E5"#"#0072B2"
MeanCol = 'k'#"#FFC107"#"#d62728"
RegCol = "#D81B60"#"#D55E00"
TrueCol = 'green' # "#004D40" #'k'
DatCol =  'gray' # 'k'"#332288"#"#009E73"


tol = 1e-6

df = pd.read_excel('ExampleOzoneProfiles.xlsx')

#print the column names
print(df.columns)

#get the values for a given column
press = df['Pressure (hPa)'].values #in hectpascal or millibars
O3 = df['Ozone (VMR)'].values
minInd = 7
maxInd = 44
pressure_values = press[minInd:maxInd]
VMR_O3 = O3[minInd:maxInd]
scalingConstkm = 1e-3
# https://en.wikipedia.org/wiki/Pressure_altitude
# https://www.weather.gov/epz/wxcalc_pressurealtitude
heights = 145366.45 * (1 - ( press /1013.25)**0.190284 ) * 0.3048 * scalingConstkm

height_values = heights[minInd:maxInd]

""" analayse forward map without any real data values"""

MinH = height_values[0]
MaxH = height_values[-1]
R = 6371 # earth radiusin km
ObsHeight = 500 # in km

''' do svd for one specific set up for linear case and then exp case'''

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#exp case first
SpecNumMeas = 33#105
SpecNumLayers = len(height_values)

# find minimum and max angle in radians
# min and max angle are defined by the height values of the retrived profile
MaxAng = np.arcsin((height_values[-1]+ R) / (R + ObsHeight))
MinAng = np.arcsin((height_values[0] + R) / (R + ObsHeight))

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#coeff = 1/(SpecNumMeas)
#meas_ang = (MinAng) + (MaxAng - MinAng) * np.exp(- coeff * 1* np.linspace(0, int(SpecNumMeas) -1 , SpecNumMeas ))
# coeff = 1/np.log(SpecNumMeas)
# meas_ang = (MinAng) + (MaxAng - MinAng) * coeff * 0.9 * np.log( np.linspace(1, int(SpecNumMeas) , SpecNumMeas ))

# fig, axs = plt.subplots(tight_layout=True)
# plt.scatter(range(len(meas_ang )),meas_ang )
# plt.show()
meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas)
A_lin, tang_heights_lin, extraHeight = gen_forward_map(meas_ang,height_values,ObsHeight,R)


# fig, axs = plt.subplots(tight_layout=True)
# plt.scatter(range(len(tang_heights_lin)),tang_heights_lin)
# #plt.show()

ATA_lin = np.matmul(A_lin.T,A_lin)
#condition number for A
A_lin = A_lin
A_linu, A_lins, A_linvh = np.linalg.svd(A_lin)
cond_A_lin =  np.max(A_lins)/np.min(A_lins)
print("normal: " + str(orderOfMagnitude(cond_A_lin)))



#to test that we have the same dr distances
tot_r = np.zeros(SpecNumMeas)
#calculate total length
for j in range(0, SpecNumMeas):
    tot_r[j] = 2*(np.sqrt( ( extraHeight + R)**2 - (tang_heights_lin[j] + R )**2) )
print('Distance through layers check: ' + str(np.allclose( sum(A_lin.T), tot_r)))









##


# graph Laplacian
# direchlet boundary condition
NOfNeigh = 2#4
neigbours = np.zeros((len(height_values),NOfNeigh))
# neigbours[0] = np.nan, np.nan, 1, 2
# neigbours[-1] = len(height_values)-2, len(height_values)-3, np.nan, np.nan
# neigbours[0] = np.nan, 1
# neigbours[-1] = len(height_values)-2, np.nan
for i in range(0,len(height_values)):
    neigbours[i] = i-1, i+1
    #neigbours[i] = i-2, i-1, i+1, i+2#, i+3 i-3,


neigbours[neigbours >= len(height_values)] = np.nan
neigbours[neigbours < 0] = np.nan

L = generate_L(neigbours)
startInd = 23
L[startInd::, startInd::] = L[startInd::, startInd::] * 5
L[startInd, startInd] = -L[startInd, startInd-1] - L[startInd, startInd+1] #-L[startInd, startInd-2] - L[startInd, startInd+2]

#L[startInd+1, startInd+1] = -L[startInd+1, startInd+1-1] - L[startInd+1,startInd+1+1] -L[startInd+1, startInd+1-2] - L[startInd+1, startInd+1+2]
# L[16, 16] = 13

np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')



#taylor exapnsion for f to do so we need y (data)

''' load data and pick wavenumber/frequency'''
#
##check absoprtion coeff in different heights and different freqencies
filename = 'tropical.O3.xml'

#VMR_O3, height_values, pressure_values = testReal.get_data(filename, ObsHeight * 1e3)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
#height_values = np.around(height_values * 1e-3,2)#in km 1e2 # in cm
#d_height = (height_values[1::] - height_values[0:-1] )
#d_height = layers[1::] - layers[0:-1]
N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3

# https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
temperature = get_temp_values(heights)
temp_values = temperature[minInd:maxInd]
#x = VMR_O3 * N_A * pressure_values /(R_gas * temp_values)#* 1e-13
#https://hitran.org/docs/definitions-and-units/
#files = '/home/lennartgolks/Python/firstModelCheck/634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects
files = '634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects

my_data = pd.read_csv(files, header=None)
data_set = my_data.values

size = data_set.shape
wvnmbr = np.zeros((size[0],1))
S = np.zeros((size[0],1))
F = np.zeros((size[0],1))
g_air = np.zeros((size[0],1))
g_self = np.zeros((size[0],1))
E = np.zeros((size[0],1))
n_air = np.zeros((size[0],1))
g_doub_prime= np.zeros((size[0],1))


for i, lines in enumerate(data_set):
    wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
    S[i] = float(lines[0][16:25]) # in cm/mol
    F[i] = float(lines[0][26:35])
    g_air[i] = float(lines[0][35:40])
    g_self[i] = float(lines[0][40:45])
    E[i] = float(lines[0][46:55])
    n_air[i] = float(lines[0][55:59])
    g_doub_prime[i] = float(lines[0][155:160])


#load constants in si annd convert to cgs units by multiplying
h = scy.constants.h #* 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b_cgs = constants.Boltzmann #* 1e7#in J K^-1
#T = temp_values[0:-1] #in K
N_A = constants.Avogadro # in mol^-1
R = constants.gas_constant


mol_M = 48 #g/mol for Ozone
#ind = 293
ind = 623
#pick wavenumber in cm^-1
v_0 = wvnmbr[ind][0]#*1e2
#wavelength
lamba = 1/v_0
f_0 = c_cgs*v_0
print("Frequency " + str(np.around(v_0*c_cgs/1e9,2)) + " in GHz")

C1 =2 * scy.constants.h * scy.constants.c**2 * v_0**3 * 1e8
C2 = scy.constants.h * scy.constants.c * 1e2 * v_0  / (scy.constants.Boltzmann * temp_values )
#plancks function
Source = np.array(C1 /(np.exp(C2) - 1) ).reshape((SpecNumLayers,1))

#differs from HITRAN, implemented as in Urban et al
T_ref = 296 #K usually
p_ref = pressure_values[0]




'''weighted absorption cross section according to Hitran and MIPAS instrument description
S is: The spectral line intensity (cm^−1/(molecule cm^−2))
f_broad in (1/cm^-1) is the broadening due to pressure and doppler effect,
 usually one can describe this as the convolution of Lorentz profile and Gaussian profile
 VMR_O3 is the ozone profile in units of molecule (unitless)
 has to be extended if multiple gases are to be monitored
 I multiply with 1e-4 to go from cm^2 to m^2
 '''
f_broad = 1
w_cross =  VMR_O3 * f_broad * 1e-4
#w_cross[0], w_cross[-1] = 0, 0

#from : https://hitran.org/docs/definitions-and-units/
HitrConst2 = 1.4387769 # in cm K

# internal partition sum
Q = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ temp_values)
Q_ref = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ 296)
LineInt = S[ind,0] * Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))
LineIntScal =  Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))
#
# fig, axs = plt.subplots(tight_layout=True)
# plt.plot(LineInt,height_values)
# plt.show()

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''

#take linear
num_mole = 1 / ( scy.constants.Boltzmann )#* temp_values)

AscalConstKmToCm = 1e3
#1e2 for pressure values from hPa to Pa
A_scal = pressure_values.reshape((SpecNumLayers,1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm/ ( temp_values)
scalingConst = 1e11
#theta =(num_mole * w_cross.reshape((SpecNumLayers,1)) * Source * scalingConst )
theta = num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]

# A_scal = pressure_values.reshape((SpecNumLayers,1)) / ( temp_values)
# scalingConst_old = 1e16
# theta =(num_mole * w_cross.reshape((SpecNumLayers,1)) * Source * scalingConst_old )
#
#num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst

""" plot forward model values """
numDensO3 =  N_A * press * 1e2 * O3 / (R * temp_values[0,:]) * 1e-6
# fig, axs = plt.subplots(tight_layout=True)
# plt.plot(press ,heights,color = [0, 205/255, 127/255])
# #plt.plot((1/ temp_values) ,heights,color ='k')
# axs.set_ylabel('Height in km')
# axs.set_xlabel('Number density of Ozone in cm$^{-3}$')
# plt.savefig('theta.png')
# plt.show()



fig, axs = plt.subplots(tight_layout=True, figsize=set_size(PgWidthPt, fraction=fraction))
#plt.plot(press/1013.25,heights, label = 'pressure in hPa/' + str(np.around(max(press),3)) )
#plt.plot(Source/max(Source),height_values, label = r'Source in $\frac{W}{m^2 sr}\frac{1}{\frac{1}{cm}}$/' + str(np.around(max(Source[0]),5)) )
plt.plot(temperature,heights, color = 'darkred')# label = r'Source in K/' + str(np.around(max(temperature[0]),3)) )
#plt.plot(LineInt,heights[minInd:maxInd], color = 'darkred')# label = r'Source in K/' + str(np.around(max(temperature[0]),3)) )
#axs.legend()
axs.tick_params(axis = 'x', labelcolor="darkred")
ax2 = axs.twiny() # ax1 and ax2 share y-axis
line3 = ax2.plot(press[minInd:maxInd],heights[minInd:maxInd], color = 'blue') #, label = 'pressure in hPa/' + str(np.around(max(press),3)) )
ax2.spines['top'].set_color('blue')
ax2.tick_params(labelcolor="blue")
ax2.set_xlabel('Pressure in hPa')
axs.set_ylabel('Height in km')
axs.set_xlabel('Temperature in K')
#axs.set_xlabel('Line intensity in cm / molecule')
#axs.set_title()
plt.savefig('PandQ.png')
plt.show()


A = A_lin * A_scal.T
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))
#theta[0] = 0
#theta[-1] = 0
Ax = np.matmul(A, theta)

#convolve measurements and add noise
#y = add_noise(Ax, 0.01)
#y[y<=0] = 0
SNR = 60
y, gamma = add_noise(Ax, SNR)
#y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))


ATy = np.matmul(A.T, y)


np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')
np.savetxt('ForWardMatrix.txt', A, header = 'Forward Matrix A', fmt = '%.15f', delimiter= '\t')
np.savetxt('height_values.txt', height_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('tan_height_values.txt', tang_heights_lin, fmt = '%.15f', delimiter= '\t')

np.savetxt('pressure_values.txt', pressure_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('VMR_O3.txt', VMR_O3, fmt = '%.15f', delimiter= '\t')




"""start the mtc algo with first guesses of noise and lumping const delta"""


vari = np.zeros((len(theta)-2,1))

for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])

#find minimum for first guesses
'''params[1] = delta
params[0] = gamma'''
def MinLogMargPost(params):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gamma = params[0]
    lamb = params[1]
    if lamb < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers
    m = SpecNumMeas

    Bp = ATA + lamb * L


    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD *  lamb * gamma + betaG *gamma)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MinLogMargPost, [1/(np.max(Ax) * 0.01)**2,1/(np.mean(vari))*(np.max(Ax) * 0.01)**2])

lam0 = minimum[1]
print(minimum)



##
""" finally calc f and g with a linear solver adn certain lambdas
 using the gmres"""

lam= np.logspace(-5,15,500)
f_func = np.zeros(len(lam))
g_func = np.zeros(len(lam))



for j in range(len(lam)):

    B = (ATA + lam[j] * L)

    B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    #print(exitCode)

    CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
    if np.linalg.norm(ATy[0::, 0]- CheckB_inv_ATy)/np.linalg.norm(ATy[0::, 0])<=tol:
        f_func[j] = f(ATy, y, B_inv_A_trans_y)
    else:
        f_func[j] = np.nan

    g_func[j] = g(A, L, lam[j])


np.savetxt('f_func.txt', f_func, fmt = '%.15f')
np.savetxt('g_func.txt', g_func, fmt = '%.15f')
np.savetxt('lam.txt', lam, fmt = '%.15f')
##
#
# '''check error in g(lambda)'''
#
#
# # B = (ATA + minimum[1]/minimum[0] * L)
# B = (ATA + minimum[1] * L)
#
# B_inv = np.zeros(np.shape(B))
# for i in range(len(B)):
#     e = np.zeros(len(B))
#     e[i] = 1
#     B_inv[:, i], exitCode = gmres(B, e, tol=tol, restart=25)
#     if exitCode!= 0 :
#         print('B_inv ' + str(exitCode))
#
# B_inv_L = np.matmul(B_inv, L)
# num_sam = 10
# trace_B_inv_L_1 = g_MC_log_det(B_inv_L, num_sam)
# trace_B_inv_L_2 = g_MC_log_det(np.matmul(B_inv_L, B_inv_L), num_sam)
# stdL1 = np.sqrt(np.var(trace_B_inv_L_1))
# stdL2 = np.sqrt(np.var(trace_B_inv_L_2))
#
# MCErrL1 = stdL1/ np.sqrt(num_sam)
# MCErrL2 = stdL2/ np.sqrt(num_sam)
#
##
''' check taylor series in f(lambda) and g(lambda)
around lam0 from gmres = '''

# lam0 =minimum[1]# / minimum[0]
# lam_try = np.linspace(lam0-lam0/2,lam0+lam0/2,101)
# f_try_func = np.zeros(len(lam_try))
# g_try_func = np.zeros(len(lam_try))
#
# g_func_tay = np.ones(len(lam_try)) * g(A, L, lam0)
#
# B = (ATA + lam0* L)
# B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
# f_func_tay = np.ones(len(lam_try)) * f(ATy, y, B_inv_A_trans_y)
#
# for j in range(len(lam_try)):
#
#     B = (ATA + lam_try[j] * L)
#
#     B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
#     #print(exitCode)
#
#     CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
#
#     if np.linalg.norm(ATy[0::, 0]- CheckB_inv_ATy)/np.linalg.norm(ATy[0::, 0])<=tol :
#         f_try_func[j] = f(ATy, y, B_inv_A_trans_y)
#     else:
#         f_try_func[j] = np.nan
#     delta_lam = lam_try[j] - lam0
#
#     g_try_func[j] = g(A, L, lam_try[j])
#
#     B_inv_L = np.zeros(np.shape(B))
#     for i in range(len(B)):
#         B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
#         if exitCode != 0:
#             print('B_inv_L ' + str(exitCode))
#     relative_tol_L = tol
#     #CheckB_inv_L = np.matmul(B, B_inv_L)
#     #print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)
#     B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
#     B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
#     B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
#     B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)
#
#     f_func_tay[j] = f_func_tay[j] + f_tayl(delta_lam, B_inv_A_trans_y, ATy[0::, 0], B_inv_L, B_inv_L_2, B_inv_L_3, B_inv_L_4, B_inv_L_5)
#     g_func_tay[j] = g_func_tay[j] + g_tayl(delta_lam, B_inv_L, B_inv_L_2, B_inv_L_3, B_inv_L_4, B_inv_L_5)

#taylor series arounf lam_0

B = (ATA + lam0* L)

B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
#print(exitCode)

CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)



B_inv_L = np.zeros(np.shape(B))

for i in range(len(B)):
    B_inv_L[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print('B_inv_L ' + str(exitCode))

#relative_tol_L = tol
#CheckB_inv_L = np.matmul(B, B_inv_L)
#print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)

B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)
B_inv_L_6 = np.matmul(B_inv_L_4, B_inv_L_2)


f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L), B_inv_A_trans_y)
f_0_2 = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y)
f_0_3 = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_3) ,B_inv_A_trans_y)
f_0_4 = -1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)
#f_0_5 = 120 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)


g_0_1 = np.trace(B_inv_L)
g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
g_0_3 = 1 /6 * np.trace(B_inv_L_3)
g_0_4 = -1 /24 * np.trace(B_inv_L_4)
g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
g_0_6 = 0#1 /720 * np.trace(B_inv_L_6)




##

'''do the sampling'''
number_samples = 10000


#inintialize sample
gamma0 = minimum[0] #3.7e-5#1/np.var(y) * 1e1 #(0.01* np.max(Ax))1e-5#
#0.275#1/(2*np.mean(vari))0.1#
lambda0 = minimum[1]#deltas[0]/gammas[0]
#deltas[0] =  minimum[1] * minimum[0]
ATy = np.matmul(A.T, y)
B = (ATA + lambda0 * L)

B_inv_A_trans_y0, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

Bu, Bs, Bvh = np.linalg.svd(B)
cond_B =  np.max(Bs)/np.min(Bs)
print("Condition number B: " + str(orderOfMagnitude(cond_B)))


#wLam = 2e2#5.5e2
#wgam = 1e-5
#wdelt = 1e-1

alphaG = 1
alphaD = 1
rate = f(ATy, y, B_inv_A_trans_y0) / 2 + betaG + betaD * lambda0
# draw gamma with a gibs step
shape = SpecNumMeas/2 + alphaD + alphaG

f_new = f(ATy, y,  B_inv_A_trans_y0)
#g_old = g(A, L,  lambdas[0])

def MHwG(number_samples, burnIn, lambda0, gamma0):
    wLam = 6e3#1e3#7e1

    alphaG = 1
    alphaD = 1
    k = 0

    gammas = np.zeros(number_samples + burnIn)
    #deltas = np.zeros(number_samples + burnIn)
    lambdas = np.zeros(number_samples + burnIn)

    gammas[0] = gamma0
    lambdas[0] = lambda0

    B = (ATA + lambda0 * L)
    B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0=B_inv_A_trans_y0, tol=tol)

    #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    shape = SpecNumMeas / 2 + alphaD + alphaG
    rate = f(ATy, y, B_inv_A_trans_y) / 2 + betaG + betaD * lambda0


    for t in range(number_samples + burnIn-1):
        #print(t)

        # # draw new lambda
        lam_p = normal(lambdas[t], wLam)

        while lam_p < 0:
                lam_p = normal(lambdas[t], wLam)

        delta_lam = lam_p - lambdas[t]
        # B = (ATA + lam_p * L)
        # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
        # if exitCode != 0:
        #     print(exitCode)


        # f_new = f(ATy, y,  B_inv_A_trans_y)
        # g_new = g(A, L,  lam_p)
        #
        # delta_f = f_new - f_old
        # delta_g = g_new - g_old

        delta_f = f_0_1 * delta_lam + f_0_2 * delta_lam**2 + f_0_3 * delta_lam**3
        delta_g = g_0_1 * delta_lam + g_0_2 * delta_lam**2 + g_0_3 * delta_lam**3

        log_MH_ratio = ((SpecNumLayers)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

        #accept or rejeict new lam_p
        u = uniform()
        if np.log(u) <= log_MH_ratio:
        #accept
            k = k + 1
            lambdas[t + 1] = lam_p
            #only calc when lambda is updated

            B = (ATA + lam_p * L)
            B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0= B_inv_A_trans_y0,tol=tol, restart=25)
            #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)

            # if exitCode != 0:
            #         print(exitCode)

            f_new = f(ATy, y,  B_inv_A_trans_y)
            #g_old = np.copy(g_new)
            rate = f_new/2 + betaG + betaD * lam_p#lambdas[t+1]

        else:
            #rejcet
            lambdas[t + 1] = np.copy(lambdas[t])




        gammas[t+1] = np.random.gamma(shape = shape, scale = 1/rate)

        #deltas[t+1] = lambdas[t+1] * gammas[t+1]

    return lambdas, gammas,k



startTime = time.time()
lambdas ,gammas, k = MHwG(number_samples, burnIn, lambda0, gamma0)
elapsed = time.time() - startTime
print('MTC Done in ' + str(elapsed) + ' s')



print('acceptance ratio: ' + str(k/(number_samples+burnIn)))
deltas = lambdas * gammas
np.savetxt('samples.txt', np.vstack((gammas[burnIn::], deltas[burnIn::], lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')


AutoCorrData = np.loadtxt("auto_corr_dat.txt", skiprows=3, dtype='float')

with open("auto_corr_dat.txt") as fID:
    for n, line in enumerate(fID):
       if n == 1:
            IntAutoDelt, IntAutoGam, IntAutoLam = [float(IAuto) for IAuto in line.split()]
            break



#refine according to autocorrelation time
new_lamb = lambdas[burnIn::math.ceil(IntAutoLam)]
#SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
new_gam = gammas[burnIn::math.ceil(IntAutoGam)]
#SetGamma = new_gam[np.random.randint(low = 0,high =len(new_gam),size =1)]
new_delt = deltas[burnIn::math.ceil(IntAutoDelt)]
#SetDelta = new_delt[np.random.randint(low = 0,high =len(new_delt),size =1)]

MTCMargSamp = np.vstack((gammas[burnIn::], lambdas[burnIn::])).T
TrMTC = np.zeros(number_samples)
for i,sam in enumerate(MTCMargSamp):
    TrMTC[i] = MinLogMargPost(sam)

##
#draw paramter samples
paraSamp = 200#n_bins
Results = np.zeros((paraSamp,len(theta)))
NormRes = np.zeros(paraSamp)
xTLxRes = np.zeros(paraSamp)
SetGammas = new_gam[np.random.randint(low=0, high=len(new_gam), size=paraSamp)]
SetDeltas  = new_delt[np.random.randint(low=0, high=len(new_delt), size=paraSamp)]

startTimeX = time.time()
for p in range(paraSamp):
    # SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
    SetGamma = SetGammas[p] #minimum[0]
    SetDelta  = SetDeltas[p] #minimum[1]
    W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
    v_1 = np.sqrt(SetGamma) *  A.T @ W
    W2 = np.random.multivariate_normal(np.zeros(len(L)), L)
    v_2 = np.sqrt(SetDelta) * W2

    SetB = SetGamma * ATA + SetDelta * L
    RandX = (SetGamma * ATy[0::, 0] + v_1 + v_2)

    # SetB_inv = np.zeros(np.shape(SetB))
    # for i in range(len(SetB)):
    #     e = np.zeros(len(SetB))
    #     e[i] = 1
    #     SetB_inv[:, i], exitCode = gmres(SetB, e, tol=tol, restart=25)
    #     if exitCode != 0:
    #         print(exitCode)

    B_inv_A_trans_y, exitCode = gmres(SetB, RandX, x0=B_inv_A_trans_y0, tol=tol)

    # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    #CheckB_inv = np.matmul(SetB, SetB_inv)
    #print(np.linalg.norm(np.eye(len(SetB)) - CheckB_inv) / np.linalg.norm(np.eye(len(SetB))) < tol)

    Results[p, :] = B_inv_A_trans_y

    NormRes[p] = np.linalg.norm( np.matmul(A,B_inv_A_trans_y) - y[0::,0])
    xTLxRes[p] = np.sqrt(np.matmul(np.matmul(B_inv_A_trans_y.T, L), B_inv_A_trans_y))

elapsedX = time.time() - startTimeX
print('Time to solve for x ' + str(elapsedX/paraSamp))


NormLTest = np.linalg.norm( np.matmul(A,np.mean(Results,0 )) - y[0::,0])
xTLxCurveTest = np.sqrt(np.matmul(np.matmul(np.mean(Results,0 ).T, L), np.mean(Results,0 )))


np.savetxt('O3Res.txt', Results/(num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst), fmt = '%.15f', delimiter= '\t')




startTime = time.time()
BinHistStart = 3

lambHist, lambBinEdges = np.histogram(new_lamb, bins=BinHistStart, density=True)

MargResults = np.zeros((BinHistStart, len(theta)))
MargVarResults = np.zeros((BinHistStart, len(theta)))
B_inv_Res = np.zeros((BinHistStart, len(theta)))
# MargResults = np.zeros((BinHist,BinHist,len(theta)))
# LamMean = 0

for p in range(BinHistStart):
    # DLambda = ( lambBinEdges[p+1] - lambBinEdges[p])/2
    SetLambda = lambBinEdges[p]
    # LamMean = LamMean + SetLambda * lambHist[p]/sum(lambHist)
    SetB = ATA + SetLambda * L

    B_inv_A_trans_y, exitCode = gmres(SetB, ATy[0::, 0], x0=B_inv_A_trans_y0, tol=tol)

    # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    MargResults[p, :] = B_inv_A_trans_y * lambHist[p] / np.sum(lambHist)
    MargVarResults[p, :] = B_inv_A_trans_y ** 2 * lambHist[p] / np.sum(lambHist)
    B_inv_Res[p, :] = B_inv_A_trans_y

trapezMat = 2 * np.ones(MargResults.shape)
trapezMat[:, 0] = 1
trapezMat[:, -1] = 1
oldMargInteg = 0.5 * np.sum(MargResults * trapezMat, 0)  # * (lambBinEdges[1]- lambBinEdges[0] )
MargIntegSq = 0.5 * np.sum(MargVarResults * trapezMat , 0)
MargInteg = np.copy(oldMargInteg)
MargX =  MargInteg/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
MargXErr = np.sqrt( (MargIntegSq - MargInteg**2 )/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)**2 )
MargTime = time.time() - startTime
print('Post Mean in ' + str(MargTime) + ' s')

oldRelErr = 0
print(BinHistStart)
for BinHist in range(BinHistStart+1,100):

    lambHist, lambBinEdges = np.histogram(new_lamb, bins= BinHist, density =True)

    MargResults = np.zeros((BinHist,len(theta)))
    MargVarResults = np.zeros((BinHist,len(theta)))
    B_inv_Res = np.zeros((BinHist,len(theta)))
    #MargResults = np.zeros((BinHist,BinHist,len(theta)))
    #LamMean = 0
    startTime  = time.time()
    for p in range(BinHist):
        #DLambda = ( lambBinEdges[p+1] - lambBinEdges[p])/2
        SetLambda =  lambBinEdges[p]
        #LamMean = LamMean + SetLambda * lambHist[p]/sum(lambHist)
        SetB = ATA + SetLambda * L

        B_inv_A_trans_y, exitCode = gmres(SetB, ATy[0::, 0], x0=B_inv_A_trans_y0, tol=tol)

        # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
        if exitCode != 0:
            print(exitCode)

        MargResults[p, :] = B_inv_A_trans_y * lambHist[p]/ np.sum(lambHist)
        MargVarResults[p, :] = B_inv_A_trans_y**2 * lambHist[p]/ np.sum(lambHist)
        B_inv_Res[p, :] = B_inv_A_trans_y

    trapezMat = 2 * np.ones(MargResults.shape)
    trapezMat[:, 0] = 1
    trapezMat[:, -1] = 1
    newMargInteg = 0.5 * np.sum(MargResults * trapezMat , 0) #* (lambBinEdges[1]- lambBinEdges[0] )

    MargIntegSq = 0.5 * np.sum(MargVarResults * trapezMat , 0)

    NormMargRes = np.linalg.norm( np.matmul(A,newMargInteg) - y[0::,0])
    xTLxMargRes = np.sqrt(np.matmul(np.matmul(newMargInteg.T, L),newMargInteg))
    newRelErr = np.linalg.norm(oldMargInteg - newMargInteg) / np.linalg.norm(newMargInteg) * 100
    print(newRelErr)
    print(BinHist)
    oldMargInteg = np.copy(newMargInteg)
    if  0.1 > newRelErr:
        print(f'break at {BinHist}')
        break
    oldRelErr = np.copy(newRelErr)


MargInteg = np.copy(oldMargInteg)
fristVar = np.zeros((BinHist,len(theta)))
for p in range(BinHist):

     fristVar = (MargInteg - B_inv_Res[p, :])**2  * lambHist[p]/ np.sum(lambHist)

otherVar = 0.5 * np.sum( fristVar * trapezMat , 0)/(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)

NormMargRes = np.linalg.norm(np.matmul(A, MargInteg) - y[0::, 0])
xTLxMargRes = np.sqrt(np.matmul(np.matmul(MargInteg.T, L), MargInteg))


MargX =  MargInteg/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
MargXErr = np.sqrt( (MargIntegSq - MargInteg**2 )/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)**2 )
MargTime = time.time() - startTime
print('Post Mean in ' + str(MargTime) + ' s')
#MargX  = np.sum(np.sum(MargResults, axis = 0), axis = 0) /(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)

# mpl.use(defBack)
# mpl.rcParams.update(mpl.rcParamsDefault)
# plt.rcParams.update({'font.size': 12})
fig2, ax = plt.subplots()
ax.plot(MargX,height_values)
ax.errorbar(MargX,height_values,yerr = 3*np.sqrt(otherVar)/2 , markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.', label = 'posterior mean ', markersize =3, linewidth =1)#, markerfacecolor = 'none'

plt.plot(VMR_O3,height_values, color = [0, 158/255, 115/255], linewidth = 11, label = 'VMR O$_3$', zorder=0)

fig, axs = plt.subplots()#, dpi = dpi)

axs.bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(lambBinEdges)[0])#10)

plt.savefig('LamHistBreakBins.png')
plt.show()

print('MTC Done in ' + str(elapsed) + ' s')

##
"Fitting prob distr to hyperparameter histogram"

def skew_norm_pdf(x,mean=0,w=1,skewP=0, scale = 0.1):
    # adapated from:
    # http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy
    t = (x-mean) / w
    return 2.0 * w * scy.stats.norm.pdf(t) * scy.stats.norm.cdf(skewP*t) * scale


##

import pytwalk

def MinLogMargPost(params):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gamma = params[0]
    lamb = params[1]
    if lamb < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers
    m = SpecNumMeas

    Bp = ATA + lamb * L


    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], x0 =B_inv_A_trans_y0,  tol=tol)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD * lamb * gamma + betaG * gamma)

#minimum = optimiz

def MargPostSupp(Params):
	return all(0 < Params)

MargPost = pytwalk.pytwalk( n=2, U=MinLogMargPost, Supp=MargPostSupp)
startTime = time.time()
tWalkSampNum= 10000
MargPost.Run( T=tWalkSampNum+ burnIn, x0=minimum, xp0=np.array([normal(minimum[0], minimum[0]/4), normal((minimum[1]),(minimum[1])/4)]) )
elapsedtWalkTime = time.time() - startTime
print('Elapsed Time for t-walk: ' + str(elapsedtWalkTime))
#MargPost.Ana()
#MargPost.TS()

#MargPost.Hist( par=0 )
#MargPost.Hist( par=1 )

MargPost.SavetwalkOutput("MargPostDat.txt")

#load data and make histogram
SampParas = np.loadtxt("MargPostDat.txt")


# eng = matlab.engine.start_matlab()
# eng.Run_Autocorr_PyTWalk(nargout=0)
# eng.quit()
#
# AutoCorrDataPyTWalk= np.loadtxt("autoCorrPyTWalk.txt", skiprows=3, dtype='float')
# #IntAutoLam, IntAutoGam , IntAutoDelt = np.loadtxt("auto_corr_dat.txt",userow = 1, skiprows=1, dtype='float'
#
with open("autoCorrPyTWalk.txt") as fID:
    for n, line in enumerate(fID):
       if n == 1:
            IntAutoDeltaPyT, IntAutoGamPyT, IntAutoLamPyT = [float(IAuto) for IAuto in line.split()]

            break

#for f and g image
LPYT = SampParas[burnIn::math.ceil(IntAutoLamPyT),1]
GPYT = SampParas[burnIn::math.ceil(IntAutoGamPyT),0]
deltasPyT = SampParas[:,1]*SampParas[:,0]
##
#plot para traces for MTC
fig, axs = plt.subplots( 3,1,  tight_layout=True, figsize=set_size(PgWidthPt, fraction=fraction) )
#fig.suptitle(str(number_samples)+' mtc samples in ' + str(math.ceil(elapsed)) + 's')
axs[0].plot(range(len(gammas)), gammas)
axs[0].set_xlabel(r'samples with $\tau_{int}=$ ' + str(math.ceil(IntAutoGam)))
axs[0].set_ylabel('$\gamma$')
axs[1].plot(range(len(deltas)), deltas)
axs[1].set_xlabel(r'samples with $\tau_{int}$= ' + str(math.ceil(IntAutoDelt)))
axs[1].set_ylabel('$\delta$')
axs[2].plot(range(len(lambdas)), lambdas)
axs[2].set_xlabel(r'samples with $\tau_{int}$= ' + str(math.ceil(IntAutoLam)))
axs[2].set_ylabel('$\lambda$')
axs[1].ticklabel_format(axis='y', style='sci',scilimits=(0,0))
axs[2].ticklabel_format(axis='y', style='sci',scilimits=(0,0))
with open('TraceMTCPara.pickle', 'wb') as filID: # should be 'wb' rather than 'w'
    pl.dump(fig, filID)
plt.savefig('TraceMTCPara.svg')
plt.show()

# #to open figure
# fig_handle = pl.load(open('TraceMTCPara.pickle','rb'))
# fig_handle.show()
##
#plot para traces for t-walk
fig, axs = plt.subplots( 2,1, tight_layout=True)
fig.suptitle(str(tWalkSampNum)+' t-walk samples in ' + str(math.ceil(elapsedtWalkTime)) + 's')
axs[0].plot(range(len(SampParas[:,0])), SampParas[:,0])
axs[0].set_xlabel(r'samples with $\tau_{int}=$ ' + str(math.ceil(IntAutoGamPyT)))
axs[0].set_ylabel('$\gamma$')
axs[1].plot(range(len(SampParas[:,1])), SampParas[:,1])
axs[1].set_xlabel(r'samples with $\tau_{int}$= ' + str(math.ceil(IntAutoDeltaPyT)))
#axs[1].set_ylabel('$\delta$')
axs[1].set_ylabel('$\lambda$')
with open('TracetWalkPara.pickle', 'wb') as filID: # should be 'wb' rather than 'w'
    pl.dump(fig, filID)
#plt.savefig('TracetWalkPara.png')
plt.show()



print('t-walk Done')

plt.close('all')


##

BinSetLamb = np.arange(min(new_lamb),max(new_lamb),(max(new_lamb)-min(new_lamb))/n_bins)
BinSetGam = np.arange(min(new_gam),max(new_gam),(max(new_gam)-min(new_gam))/n_bins)
BinSetDelt = np.arange(min(new_delt),max(new_delt),(max(new_delt)-min(new_delt))/n_bins)

fig, axs = plt.subplots(3, 1, tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction))

axs[0].hist(new_gam,bins=BinSetGam, color = MTCCol, zorder = 0, label = 'MTC')

axs[1].hist(new_lamb,bins=BinSetLamb, color = MTCCol, zorder = 0)#10)

axs[2].hist(new_delt,bins=BinSetDelt, color = MTCCol, zorder = 0)

axs[0].set_title(r'$\gamma$, the noise precision', fontsize = 12)
axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)

axs[2].set_title(r'$\delta $, the smoothness parameter', fontsize = 12)
plt.savefig('MTCHistoResPraesi.png')
plt.show()


##
# mpl.use(defBack)
# mpl.rcParams.update(mpl.rcParamsDefault)
# plt.rcParams.update({'font.size': 12})
# fig, axs = plt.subplots(3, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction))#, dpi = dpi)
# n_bins = n_bins
# BinSetLamb = np.arange(min(new_lamb),max(new_lamb),(max(new_lamb)-min(new_lamb))/n_bins)
# BinSetGam = np.arange(min(new_gam),max(new_gam),(max(new_gam)-min(new_gam))/n_bins)
# BinSetDelt = np.arange(min(new_delt),max(new_delt),(max(new_delt)-min(new_delt))/n_bins)
#
#
# axs[0].hist(new_gam,bins=BinSetGam, color = MTCCol, zorder = 0, label = 'MTC')
# #axs[0].set_ylim([0,400])
# axs0 = axs[0].twinx()
# axs0.hist(GPYT,bins=BinSetGam,color = pyTCol, zorder = 1, label = 't-walk')
# axs0.set_ylim([0,100])
# axs0.tick_params(axis = 'y', colors=pyTCol, which = 'both')
#
# hist0, lab0 = axs[0].get_legend_handles_labels()
# hist00, lab00 = axs0.get_legend_handles_labels()
# axs[0].legend(labels = lab0 + lab00, handles = hist0+hist00 , labelcolor = [MTCCol, pyTCol] ,loc='upper right',frameon=True, fontsize = 12)#,bbox_to_anchor=(1.05, 1.15))
# axs[0].spines[:].set_visible(False)
# axs0.spines['right'].set_color(pyTCol)
# axs[1].hist(new_lamb,bins=BinSetLamb, color = MTCCol, zorder = 0)#10)
# #axs[2].set_ylim([0,200])
# axs1 = axs[1].twinx()
# axs1.hist(LPYT,bins=BinSetLamb,color = pyTCol, zorder = 1)
# axs1.set_ylim([0,100])
# axs1.tick_params(axis = 'y', colors=pyTCol, which = 'both')
# axs[1].spines[:].set_visible(False)
# axs1.spines['right'].set_color(pyTCol)
#
# axs[0].set_title(r'$\gamma$, the noise precision', fontsize = 12)
# axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)
# axs[2].plot(range(len(SampParas[burnIn::,2])), TrMTC ,color = MTCCol, linewidth = 1.2)
# axs[2].set_xlim([0,number_samples])
# axs[2].set_title('trace of the neg-log of the marginal posterior $-\log \, \pi( \lambda, \gamma | \mathbf{y})$', fontsize = 12)
# #axs[2].set_ylabel(r'$-log \pi( \gamma, \lambda | \mathbf{y})$')
# axs[2].set_xlabel('iterations')
# axs[2].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.f'))
# #axs[2].tick_params(axis = 'y', which = 'both', labelleft=False, left=False)
# axs2 = axs[2].twinx()
# axs2.plot(range(len(SampParas[burnIn::,2])), SampParas[burnIn::,2],color = pyTCol, linewidth = 0.6)
# axs2.tick_params(axis = 'y', which = 'both', labelright=False, right=False, labelleft=False, left=False)
# axs[2].spines[:].set_visible(False)
#
# plt.savefig('AllHistoResultsPlus.png')
# plt.show()
#
# ##
# mpl.use('pgf')
# mpl.rcParams.update(pgf_params)
# fig.savefig('AllHistoResultsPlus.pgf', bbox_inches='tight', dpi= dpi)



##
'''make figure for f and g including the best lambdas and taylor series'''
#
#
#
# fig,axs = plt.subplots(1,2, figsize=(14, 5))
# axs[0].plot(lam,f_func)
# axs[0].scatter(lam0,f_try_func[50], color = 'green', s= 70, zorder=4)
# axs[0].annotate('mode $\lambda_0$ of marginal posterior',(lam0+2e4,f_try_func[50]), color = 'green', fontsize = 14.7)
# axs[0].scatter(np.mean(lambdas),f_MTC, color = 'red', zorder=5)
# axs[0].annotate('MTC $\lambda$ sample mean',(np.mean(lambdas)+1e4,f_MTC), color = 'red')
# axs[0].scatter(lamPyT,f_tW, color = 'k', s = 35, zorder=5)
# axs[0].annotate('T-Walk $\lambda$ sample mean',(lamPyT+1e5,f_tW+2e6), color = 'k')
# axs[0].set_xscale('log')
# axs[0].set_yscale('log')
# axs[0].set_ylabel('f($\lambda$)')
# axs[0].set_xlabel('$\lambda$')
# inset_ax = axs[0].inset_axes([0.05,0.41,0.55,0.55])
# inset_ax.scatter(lam0,f_try_func[50], color = 'green', s=60, zorder=3)
# inset_ax.annotate('$\lambda_0$',(lam0+1e3,f_try_func[50]-3e5), color = 'green', fontsize = 20 )
# inset_ax.plot(lam_try,f_func_tay, color = 'red',linewidth = 5, label = '$5^{th}$ Taylor series')
# inset_ax.plot(lam_try,f_try_func, label = 'f($\lambda$)')
# inset_ax.set_xscale('log')
# inset_ax.set_yscale('log')
# inset_ax.legend(loc = 'upper left', facecolor = 'none')
# inset_ax.tick_params(
#     axis='y',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     left=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelleft=False)
#
# #axs.set_yscale('log')
# axs[1].plot(lam,g_func)
# axs[1].scatter(lam0,g_try_func[50], color = 'green', s=70, zorder=4)
# axs[1].annotate('mode $\lambda_0$ of marginal posterior',(lam0+3e5,g_try_func[50]), color = 'green')
# #axs[1].scatter(np.mean(lambdas),g_func[239], color = 'red', zorder=5)
# axs[1].errorbar(np.mean(lambdas),g(A, L, np.mean(lambdas) ), color = 'red', zorder=5, xerr=np.sqrt(np.var(lambdas))/2, fmt='o')
# axs[1].annotate('MTC $\lambda$ sample mean',(np.mean(lambdas)+1e4,g(A, L, np.mean(lambdas) )-45), color = 'red')
# axs[1].scatter(lamPyT,g(A, L, lamPyT) , color = 'k', s=35, zorder=5)
# axs[1].annotate('T-Walk $\lambda$ sample mean',(lamPyT+1e6,g(A_lin, L, lamPyT) +50), color = 'k')
# axs[1].set_xscale('log')
# axs[1].set_xlabel('$\lambda$')
# axs[1].set_ylabel('g($\lambda$)')
# inset_ax = axs[1].inset_axes([0.05,0.41,0.55,0.55])
# inset_ax.plot(lam_try,g_func_tay, color = 'red',linewidth = 5,label = '$5^{th}$ Taylor series')
# inset_ax.plot(lam_try,g_try_func, label = 'g($\lambda$)')
# inset_ax.scatter(lam0,g_try_func[50], color = 'green', s=60, zorder=3)
# inset_ax.annotate('$\lambda_0$',(lam0+1e3,g_try_func[50]-2), color = 'green', fontsize = 20 )
# inset_ax.set_xscale('log')
# inset_ax.set_yscale('log')
# inset_ax.legend(loc = 'upper left', facecolor = 'none')
# inset_ax.tick_params(
#     axis='y',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     left=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelleft=False)
# with open('f_and_g.pickle', 'wb') as filID: # should be 'wb' rather than 'w'
#     pl.dump(fig, filID)
# #plt.savefig('f_and_g.png')
# #plt.show()
##
"""f und g for  paper"""
B_mode = ATA + minimum[1] * L
B_mode_inv_A_trans_y, exitCode = gmres(B_mode, ATy[0::, 0], tol=tol)
if exitCode != 0:
    print(exitCode)
f_mode = f(ATy, y, B_mode_inv_A_trans_y)



B_MTC = ATA + np.mean(new_lamb) * L
B_MTC_inv_A_trans_y, exitCode = gmres(B_MTC, ATy[0::, 0], tol=tol)
if exitCode != 0:
    print(exitCode)
f_MTC = f(ATy, y, B_MTC_inv_A_trans_y)


lamPyT = np.mean(LPYT)
varPyT = np.var(LPYT)
B_tW = ATA + lamPyT * L
B_tW_inv_A_trans_y, exitCode = gmres(B_tW, ATy[0::, 0], tol=tol)
if exitCode != 0:
    print(exitCode)
f_tW = f(ATy, y, B_tW_inv_A_trans_y)




B_MTC_min = ATA + (np.mean(lambdas) - np.sqrt(np.var(lambdas))/2) * L
B_MTC_min_inv_A_trans_y, exitCode = gmres(B_MTC_min, ATy[0::, 0], tol=tol)
if exitCode != 0:
    print(exitCode)
f_MTC_min = f(ATy, y, B_MTC_min_inv_A_trans_y)

B_MTC_max = ATA + (np.mean(lambdas) + np.sqrt(np.var(lambdas))/2) * L
B_MTC_max_inv_A_trans_y, exitCode = gmres(B_MTC_max, ATy[0::, 0], tol=tol)
if exitCode != 0:
    print(exitCode)
f_MTC_max = f(ATy, y, B_MTC_max_inv_A_trans_y)

xMTC = np.mean(lambdas) - np.sqrt(np.var(lambdas))/2

B_pyT_min = ATA + (lamPyT - np.sqrt(varPyT)/2) * L
B_pyT_min_inv_A_trans_y, exitCode = gmres(B_pyT_min, ATy[0::, 0], tol=tol)
if exitCode != 0:
    print(exitCode)
f_pyT_min = f(ATy, y, B_pyT_min_inv_A_trans_y)

B_pyT_max = ATA + (lamPyT + np.sqrt(varPyT)/2) * L
B_pyT_max_inv_A_trans_y, exitCode = gmres(B_pyT_max, ATy[0::, 0], tol=tol)
if exitCode != 0:
    print(exitCode)
f_pyT_max = f(ATy, y, B_pyT_max_inv_A_trans_y)

xpyT = lamPyT - np.sqrt(varPyT)/2

B_min = ATA + (np.mean(lambdas) - np.sqrt(np.var(lambdas)) ) * L
B_min_inv_A_trans_y, exitCode = gmres(B_min, ATy[0::, 0], tol=tol)
if exitCode != 0:
    print(exitCode)
f_min = f(ATy, y, B_min_inv_A_trans_y)

B_max = ATA + (np.mean(lambdas) + np.sqrt(np.var(lambdas)) ) * L
B_max_inv_A_trans_y, exitCode = gmres(B_max, ATy[0::, 0], tol=tol)
if exitCode != 0:
    print(exitCode)
f_max = f(ATy, y, B_max_inv_A_trans_y)



##

# BinHist = 30#n_bins
# lambHist, lambBinEdges = np.histogram(new_lamb, bins= BinHist, density= True)

fCol = [0, 144/255, 178/255]
gCol = [230/255, 159/255, 0]
#gCol = [240/255, 228/255, 66/255]
#gCol = [86/255, 180/255, 233/255]
gmresCol = [204/255, 121/255, 167/255]


delta_lam = lambBinEdges - minimum[1]
taylorG = g_tayl(delta_lam,g(A, L, minimum[1]) ,g_0_1, g_0_2, g_0_3, g_0_4,g_0_5, g_0_6)
taylorF = f_tayl(delta_lam, f_mode, f_0_1, f_0_2, f_0_3, f_0_4)


fig,axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))#, dpi = dpi)

axs.plot(lam,f_func, color = fCol, zorder = 2, linestyle=  'dotted')
#axs.scatter(minimum[1],f_mode, color = gmresCol, zorder=0, marker = 's')#

#axs.annotate('$\lambda_0$ mode of marginal posterior',(5.05e4,0.25), color = 'green', fontsize = 14.7)
#axs.scatter(np.mean(lambdas),f_MTC, color = MTCCol, zorder=6)#, s = 10)
#axs.annotate('MTC $\lambda$ sample mean',(5.05e4,0.375), color = 'red')
#axs.scatter(lamPyT,f_tW, color = pyTCol, zorder=5, marker = 'D')#s = 35
#axs.annotate('T-Walk $\lambda$ sample mean',(5.05e4,0.6), color = 'k')

axs.set_yscale('log')
axs.set_xlabel('$\lambda$')
axs.set_ylabel('$f(\lambda)$')#, color = fCol)
axs.tick_params(axis = 'y',  colors=fCol, which = 'both')

ax2 = axs.twinx() # ax1 and ax2 share y-axis
ax2.plot(lam,g_func, color = gCol, zorder = 2, linestyle=  'dashed')
#ax2.scatter(minimum[1],g(A, L, minimum[1]), color = gmresCol, zorder=0, marker = 's')

#ax2.scatter(np.mean(lambdas),g(A, L, np.mean(lambdas) ), color = MTCCol, zorder=5)
#ax2.scatter(lamPyT,g(A, L, lamPyT) , color = pyTCol, zorder=6, marker = 'D')
#ax2.annotate('T-Walk $\lambda$ sample mean',(lamPyT+1e6,g(A_lin, L, lamPyT) +50), color = 'k')
ax2.set_ylabel('$g(\lambda)$')#,color = gCol)
ax2.tick_params(axis = 'y', colors= gCol)
axs.set_xscale('log')
axins = axs.inset_axes([0.05,0.5,0.4,0.45])
axins.plot(lam,f_func, color = fCol, zorder=3, linestyle=  'dotted', linewidth = 3, label = '$f(\lambda)$')

axins.axvline( minimum[1], color = gmresCol, label = r'$\pi(\lambda_0|\bm{y}, \gamma)$')

axins.plot(lambBinEdges,taylorF , color = 'k', linewidth = 1, zorder = 1, label = 'Taylor series' )
axs.plot(lambBinEdges,taylorF , color = 'k', linewidth = 1, zorder = 2, label = 'Taylor series' )

#axins.scatter(,f_tayl(delta_lam, f_mode, f_0_1, f_0_2, f_0_3), color = 'k')
#axins.errorbar(np.mean(lambdas),f_MTC, color = MTCCol, zorder=3,xerr=np.sqrt(np.var(lambdas))/2,markersize = 15, fmt='o', label = r'\textbf{MwG}') #markersize = 15
#axins.errorbar(lamPyT,f_tW, xerr=np.sqrt(varPyT)/2, color = pyTCol, markersize = 10,zorder=5,fmt='D', label = 't-walk') #markersize = 10
#axins.add_patch(mpl.patches.Rectangle( (xpyT, f_pyT_min), np.sqrt(varPyT), f_pyT_max - f_pyT_min,edgecolor=pyTCol,facecolor='none', alpha = 1, zorder = 0, linewidth = 5))
#axins.add_patch(mpl.patches.Rectangle((xMTC, f_MTC_min), np.sqrt(np.var(lambdas)), f_MTC_max - f_MTC_min,edgecolor=MTCCol, facecolor='none',alpha =1,zorder = 0, linewidth = 5))

#axins.scatter(minimum[1],f_mode, color = gmresCol, s= 95, zorder=0, marker = 's', label = r'mode of $\pi(\lambda|\bm{y})$')#r'\texttt{optimize.fmin()}'

axins.set_ylim(0.95 * taylorF[0],1.5 * taylorF[-1])
axins.set_xlabel('$\lambda$')
#axins.set_xlim([np.mean(lambdas) -np.sqrt(np.var(lambdas)), 1.5*np.mean(lambdas) + np.sqrt(np.var(lambdas))])# apply the x-limits
axins.set_yscale('log')
axins.set_xscale('log')

# for label in axins.xaxis.get_ticklabels():
#     label.set_visible(False)
#axins.xaxis.get_ticklabels()[:] = []

axins.tick_params(axis='y', which='both',  left=False, labelleft=False)
#axins.tick_params(axis='x', which='both', labelbottom=False)
# for label in axins.tick_params(axis='x').get_ticklabels()[::3]:
#     label.set_visible(True)
#plt.setp(axins.get_yticklabels(), visible=False)
axins.tick_params(axis='y', which='both', length=0)

#
# firstXLabel = axins.get_xticklabels()
# firstXticks = axins.get_xticks()
# xTicksLab =[firstXLabel[0],str(minimum[1]),firstXLabel[1]]
# xTicks=np.array([firstXticks[0],minimum[1],firstXticks[1]])
# axins.set_xticks(ticks =xTicks,labels = xTicksLab)
#axins.tick_params(axis='x', which='both', labelbottom = False)

# axins.set_xticks(ticks = [minimum[1]],labels = [str(minimum[1])])
# axins.tick_params(axis='x', which='both', labelbottom = True)
axin2 = axins.twinx()
axin2.spines['top'].set_visible(False)
axin2.spines['right'].set_visible(False)
axin2.spines['bottom'].set_visible(False)
axin2.spines['left'].set_visible(False)

#plt.setp(axin2.get_yticklabels(), visible=False)


axin2.tick_params(axis = 'y', which = 'both',labelright=False, right=False)
axin2.tick_params(axis='y', which='both', length=0)


# #axin2.set_xticks([np.mean(lambdas) -np.sqrt(np.var(lambdas)) , np.mean(lambdas), np.mean(lambdas) + np.sqrt(np.var(lambdas)) ] )
axin2.plot(lam,g_func, color = gCol, zorder=3, linestyle=  'dashed', linewidth = 3,label = '$g(\lambda)$')

axin2.plot(lambBinEdges, taylorG, color = 'k', linewidth = 1, zorder = 2 )

ax2.plot(lambBinEdges, taylorG , color = 'k', linewidth = 1, zorder = 1)
ax2.axvline( minimum[1], color = gmresCol)
#axin2.scatter(minimum[1],g(A, L, minimum[1]), color = gmresCol, s=95, zorder=0, marker = 's')


# #axin2.add_patch(mpl.patches.Rectangle( (xpyT,g(A, L, lamPyT - np.sqrt(varPyT)/2)), np.sqrt(varPyT), g(A, L, lamPyT + np.sqrt(varPyT)/2) - g(A, L, lamPyT - np.sqrt(varPyT)/2),color="black", alpha = 0.5,  zorder = 0))

#axin2.errorbar(lamPyT,g(A, L, lamPyT) , xerr=np.sqrt(varPyT)/2, color = pyTCol, zorder=1, fmt='D', markersize=10, capsize =0)#,markeredgewidth = 3)
#axin2.errorbar(np.mean(lambdas),g(A, L, np.mean(lambdas) ), xerr=np.sqrt(np.var(lambdas))/2, color = MTCCol, zorder=1, fmt='o',markersize=15, capsize =0)#,markeredgewidth = 3)
#axin2.errorbar(lamPyT,g(A, L, lamPyT) , xerr=0, color = pyTCol, zorder=1, fmt='D', markersize=10, capsize =0)#,markeredgewidth = 3)
#axin2.errorbar(np.mean(lambdas),g(A, L, np.mean(lambdas) ), xerr=0, color = MTCCol, zorder=1, fmt='o',markersize=15, capsize =0)#,markeredgewidth = 3)


#axs.indicate_inset_zoom(axins, edgecolor="none", linewidth= 0.1)
axin2.set_ylim(0.8 * taylorG[0],1.05 * taylorG[-1])
axin2.set_xlim(min(lambBinEdges),max(lambBinEdges))
axin2.set_xscale('log')

#mark_inset(axs, axins, loc1=3, loc2=4, fc="none", ec="0.5")
lines2, lab2 = axin2.get_legend_handles_labels()
lines, lab0 = axins.get_legend_handles_labels()
#axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
#axs.spines['left'].set_color(fCol)
axs.spines['left'].set_color('k')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color('k')
#ax2.spines['right'].set_color(gCol)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)


#handlesZoom, labels = axins.get_legend_handles_labels()

#axs.legend(handles = [handles,],loc = 'lower right')

axs.legend(np.append(lines2,lines),np.append(lab2,lab0), loc = 'lower right')

# firstXLabel = axins.get_xticklabels()
# firstXLabel2 = axin2.get_xticklabels()
# firstXticks = axins.get_xticks()
# firstXticks2 = axin2.get_xticks()
# axins.set_xticks(ticks = firstXticks,labels = [])
#axin2.set_xticks(ticks = [minimum[1]],labels = [str(minimum[1])])
# axin2.set_xticks(ticks = firstXticks2,labels = [])
axins.set_xlim(min(lambBinEdges),max(lambBinEdges))

fig.savefig('f_and_g_paper.svg', bbox_inches='tight')
#plt.savefig('f_and_g_paper.png',bbox_inches='tight')
plt.show()
#for legend
# tikzplotlib_fix_ncols(fig)
# tikzplotlib.save("f_and_g_paper.pgf")
##
# import tikzplotlib
# tikzplotlib_fix_ncols(fig)
# tikzplotlib.save("f_and_g_paper.tex")
# ##
#
# plt.close()
# mpl.use('pgf')
# mpl.rcParams.update(pgf_params)
# fig.savefig('f_and_g_paper.pgf', bbox_inches='tight', dpi = 300)

print('bla')

##
'''L-curve refularoization
'''

lamLCurve = np.logspace(-1,10,200)
#lamLCurve = np.linspace(1e-15,1e3,200)

NormLCurve = np.zeros(len(lamLCurve))
xTLxCurve = np.zeros(len(lamLCurve))
xTLxCurve2 = np.zeros(len(lamLCurve))
for i in range(len(lamLCurve)):
    B = (ATA + lamLCurve[i] * L)

    x, exitCode = gmres(B, ATy[0::, 0], tol=tol)
    if exitCode != 0:
        print(exitCode)
        NormLCurve[i] = np.nan
        xTLxCurve[i] = np.nan

    else:
        NormLCurve[i] = np.linalg.norm( np.matmul(A,x) - y[0::,0])#, ord = 2)
        #NormLCurve[i] = np.sqrt( (np.matmul(A, x) - y[0::, 0]).T @ (np.matmul(A, x) - y[0::, 0]) )
        #NormLCurve[i] =np.linalg.norm( np.matmul(A_lin,x))
        #NormLCurve[i] = np.sqrt(np.sum((np.matmul(A_lin, x) - y)**2))

        xTLxCurve[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))
        #xTLxCurve[i] = np.linalg.norm(np.matmul(L,x) , ord = 2)
        #xTLxCurve[i] = np.linalg.norm(np.matmul(scy.linalg.sqrtm(L),x) , ord = 2)
        #xTLxCurve[i] = np.linalg.norm(x)#, ord = 2)
        #xTLxCurve[i] = np.sqrt(x.T @ x)

startTime  = time.time()
lamLCurveZoom = np.logspace(1,8.5,200)
NormLCurveZoom = np.zeros(len(lamLCurve))
xTLxCurveZoom = np.zeros(len(lamLCurve))
for i in range(len(lamLCurveZoom)):
    B = (ATA + lamLCurveZoom[i] * L)

    x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    NormLCurveZoom[i] = np.linalg.norm( np.matmul(A,x) - y[0::,0])
    xTLxCurveZoom[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))


#
# # NormLCurveZoom = lamLCurveZoom**(-2+1)
# # xTLxCurveZoom = lamLCurveZoom**2
# mpl.use(defBack)
# mpl.rcParams.update(mpl.rcParamsDefault)
# mpl.rcParams.update({'font.size': 12})#,
# # fig, axs = plt.subplots( tight_layout=True,figsize=set_size(245, fraction=fraction))
# # axs.scatter( xTLxCurveZoom,NormLCurveZoom)
# # axs.set_xscale('log')
# # axs.set_yscale('log')
# # plt.show()


# fig, axs = plt.subplots( tight_layout=True,figsize=set_size(245, fraction=fraction))
# axs.plot( NormLCurveZoom, xTLxCurveZoom)
# #axs.plot( lamLCurveZoom, xTLxCurveZoom)
# axs.set_xscale('log')
# axs.set_yscale('log')
# plt.show()


# #
# diff = lamLCurveZoom[1::] - lamLCurveZoom[0:-1]
# diffMat = np.append((np.zeros(198) - np.eye(198)),np.zeros((198,2)), axis = 1) + np.append(np.zeros((198,2)), np.zeros(198) + np.eye(198), axis = 1)
# def diff_central(x, y):
#     x0 = x[:-2]
#     x1 = x[1:-1]
#     x2 = x[2:]
#     y0 = y[:-2]
#     y1 = y[1:-1]
#     y2 = y[2:]
#     f = (x2 - x1)/(x2 - x0)
#     return (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0)
#
# st = int(0)
# dy_dt = np.gradient(NormLCurveZoom[st::], lamLCurveZoom[st::] )
# dx_dt = np.gradient(xTLxCurveZoom[st::], lamLCurveZoom[st::])
#
#
# # dx_dt = diff_central(lamLCurveZoom, NormLCurveZoom)
# # dy_dt = diff_central(lamLCurveZoom, xTLxCurveZoom)
#
# d2x_dt2 = np.gradient(dx_dt, lamLCurveZoom[st::] )
# d2y_dt2 = np.gradient(dy_dt, lamLCurveZoom [st::])
#
# curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
#
#
# dcurv_dt = np.gradient(curvature , lamLCurveZoom[st::] )
# fig, axs = plt.subplots( tight_layout=True,figsize=set_size(245, fraction=fraction))
# #axs.scatter( NormLCurveZoom, xTLxCurveZoom)
# #axs.scatter( lamLCurveZoom[st::], curvature)
# axs.plot( lamLCurveZoom[st::], dy_dt)
# #axs.plot( lamLCurveZoom[st::], dx_dt)
# #axs.set_xscale('log')
# #axs.set_yscale('log')
# plt.show()


#np.savetxt('LCurve.txt', np.vstack((NormLCurveZoom, xTLxCurveZoom, lamLCurveZoom)).T, header = 'Norm ||Ax - y|| sqrt(x.T L x) lambdas', fmt = '%.15f \t %.15f \t %.15f')

#
# eng = matlab.engine.start_matlab()
# eng.run_l_corner(nargout=0)
# eng.quit()
#
# opt_norm , opt_regNorm, opt_ind  = np.loadtxt("l_curve_output.txt", skiprows=4, dtype='float')
#

#IntAutoLam, IntAutoGam , IntAutoDelt = np.loadtxt("auto_corr_dat.txt",userow = 1, skiprows=1, dtype='float'


# with open("l_curve_output.txt") as fID:
#     for n, line in enumerate(fID):
#         if n == 0:
#             opt_norm = float(line)
#         if n == 1:
#             opt_regNorm = float(line)
#         if n == 2:
#             opt_ind = float(line)
#
#
#             break

#lam_opt = opt_ind#lamLCurve[int(opt_ind - 1)]
#lam_opt = LamMean#sum(lambBinEdges[:-1]* lambHist[p]/sum(lambHist))


#import tikzplotlib

import kneed

# calculate and show knee/elbow
kneedle = kneed.KneeLocator(NormLCurveZoom, xTLxCurveZoom, curve='convex', direction='decreasing', online = True, S = 0, interp_method="interp1d")
knee_point = kneedle.knee

elapsedtRegTime = time.time() - startTime
print('Elapsed Time to find oprimal Reg Para: ' + str(elapsedtRegTime))
#knee_point = kneedle.knee_y #

lam_opt = lamLCurveZoom[ np.where(NormLCurveZoom == knee_point)[0][0]]
print('Knee: ', lam_opt) #print('Elbow: ', elbow_point)

elbow_point = kneedle.elbow

lam_opt_elbow = lamLCurveZoom[ np.where(NormLCurveZoom == knee_point)[0][0]]

print('Elbow: ', lam_opt_elbow)

B = (ATA + lam_opt * L)
x_opt, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
LNormOpt = np.linalg.norm( np.matmul(A,x_opt) - y[0::,0])#, ord = 2)
xTLxOpt = np.sqrt(np.matmul(np.matmul(x_opt.T, L), x_opt))

# xTLxMargRes = np.sqrt(np.matmul(np.matmul(np.sum(MargInteg,0 ).T, L),np.sum(MargInteg,0 )))
#

#xTLxOpt = np.linalg.norm(x_opt,ord =2)
#xTLxOpt = np.linalg.norm(np.matmul(L,x_opt), ord = 2)

# mpl.use('QT5Agg')
# #mpl.use("png") bbox_inches='tight'
# mpl.rcParams.update(mpl.rcParamsDefault)
#
# fig, axs = plt.subplots( tight_layout=True,figsize=set_size(245, fraction=fraction))
# axs.scatter(NormLCurve,xTLxCurve, zorder = 0, color = 'green')
# #axs.scatter(NormLCurve,xTLxCurve2, zorder = 0, color = 'k')
# axs.scatter(l_corner_output[0],l_corner_output[1], zorder = 0, color = 'blue')
# axs.scatter(LNormOpt,xTLxOpt , zorder = 0, color = 'red')
# axs.set_xscale('log')
# axs.set_yscale('log')
# plt.show()


#B = (ATA + minimum[1]/minimum[0] * L)
# B = (ATA + minimum[1] * L)
#
# x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
# if exitCode != 0:
#     print(exitCode)
#
# NormLTest = np.linalg.norm( np.matmul(A,x) - y[0::,0])
# xTLxCurveTest = np.sqrt(np.matmul(np.matmul(x.T, L), x))




# B = (ATA + minimum[1]/minimum[0] * L)
# x, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
# if exitCode != 0:
#     print(exitCode)
SampleNorm = np.linalg.norm( np.matmul(A,np.mean(Results,0 )) - y[0::,0])
SamplexTLx = np.sqrt(np.matmul(np.matmul(np.mean(Results,0 ).T, L), np.mean(Results,0 )))

##


# {'text.usetex': True})
# mpl.rcParams['mathtext.fontset'] = 'custom'
# mpl.rcParams['mathtext.it'] = 'STIXGeneral:italic'
# mpl.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'[0, 114/255, 178/255]



fig, axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
axs.scatter(NormLCurve,xTLxCurve, zorder = 0, color =  DatCol, s = 1, marker ='s')
#axs.scatter(LNormOpt ,xTLxOpt, zorder = 10, color = 'red', label = 'Opt. Tikh. regularization ')
#axs.scatter(opt_norm ,opt_regNorm, zorder = 10, color = 'red')
axs.scatter(NormRes, xTLxRes, color = ResCol, s = 1.5, marker = "+")# ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')
#axs.scatter(NewNormRes, NewxTLxRes, color = 'red', label = 'MTC RTO method')#, marker = "." ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')

#axs.scatter(SampleNorm, SamplexTLx, color = 'green', marker = 's', s= 100)
axs.scatter(NormMargRes, xTLxMargRes, color = MeanCol, marker = '.', s= 25, label = 'posterior mean',zorder=2)
#E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[\mathbf{x}_{\lambda}]$
#axs.axvline(x = knee_point)
axs.scatter(knee_point, kneedle.knee_y, color = regCol, marker = 'v',label = 'max. curvature', s= 25,zorder=1)
#zoom in
x1, x2, y1, y2 = NormLCurveZoom[0], NormLCurveZoom[-31], xTLxCurveZoom[0], xTLxCurveZoom[-1] # specify the limits
axins = axs.inset_axes([0.1,0.05,0.55,0.45])
#axins.scatter(LNormOpt ,xTLxOpt, zorder = 10, color = regCol)

axins.scatter(NormRes, xTLxRes, color = ResCol, label = r'posterior samples ',marker = '+')#,$\mathbf{x} \sim \pi (\mathbf{x}| \mathbf{y}, \mathbf{\theta})$ s = 15)
axins.scatter(NormLCurve,xTLxCurve, color =  DatCol,marker = 's', s= 10,zorder=0)
axins.scatter(NormMargRes, xTLxMargRes, color = MeanCol, marker = '.', s= 100,zorder=2)
# axins.scatter(LNormOpt, xTLxOpt, color = 'crimson', marker = "s", s =80)[240/255,228/255,66/255]
#axins.annotate(r'E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[\lambda]$ = ' + str('{:.2f}'.format(lam_opt)), (LNormOpt+0.05,xTLxOpt))
#axins.scatter(NewNormRes, NewxTLxRes, color = 'red', label = 'MTC RTO method', s = 10)#, marker = "." ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')
axins.scatter(knee_point, kneedle.knee_y, color = RegCol, marker = 'v', s = 120,zorder=1)
axins.set_xlim(x1-0.01, x2-1) # apply the x-limits
#axins.set_ylim(y2,y1)
axins.set_ylim(y2,max(xTLxRes)+0.001) # apply the y-limits (negative gradient)
axins.tick_params(axis = 'x', which = 'both', labelbottom=False, bottom = False)
axins.tick_params(axis = 'y', which = 'both', labelleft=False, left = False)
axins.set_xscale('log')
axins.set_yscale('log')
handles2, labels2 = axins.get_legend_handles_labels()
axs.indicate_inset_zoom(axins, edgecolor="none")


axs.set_xscale('log')
axs.set_yscale('log')
axs.set_ylabel(r'$ \sqrt{\bm{x}^T \bm{L}\bm{x}}$', style='italic')
axs.set_xlabel(r'$|| \bm{Ax} - \bm{y}||$')
#axs.set_title('L-curve for m=' + str(SpecNumMeas))
mark_inset(axs, axins, loc1=1, loc2=3, fc="none", ec="0.5")

handles, labels = axs.get_legend_handles_labels()

axs.legend(handles = [handles[0],handles[1],handles2[0]],loc = 'upper right',  frameon =True)
plt.savefig('LCurve.svg')
#plt.savefig('LCurve.png')
#tikzplotlib.save("LCurve.tex")
plt.show()


#tikzplotlib_fix_ncols(fig)
#tikzplotlib.save("LCurve.pgf")
print('bla')

np.savetxt('RegSol.txt',x_opt /(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst), fmt = '%.15f', delimiter= '\t')
##

#
# pgf_params = { "pgf.texsystem": "pdflatex",
#    'text.usetex': True,
#     'pgf.rcfonts': False,
#     'mathtext.fontset':  'stix',
#                'mathtext.fontset' :'custom',
# 'mathtext.it': 'STIXGeneral:italic',
# 'mathtext.bf': 'STIXGeneral:italic:bold' }
# mpl.use('pgf')
# mpl.rcParams.update(pgf_params)
#
# fig.savefig('LCurve.pgf', bbox_inches='tight')
#

## make scatter plot for results
#
# BinHist = 30#n_bins
# lambHist, lambBinEdges = np.histogram(new_lamb, bins= BinHist, density= True)

paramsSkew, covs = scy.optimize.curve_fit(skew_norm_pdf,lambBinEdges[1::], lambHist/ np.sum(lambHist), p0 = [np.mean(lambBinEdges[1::]),np.sqrt(np.var(lambdas)),0.01, 1] )#np.mean(new_lamb)+1e3



fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction), gridspec_kw={'height_ratios': [3, 1]} )#, dpi = dpi)

axs[0].scatter(gammas[burnIn::math.ceil(IntAutoLam)+5],deltas[burnIn::math.ceil(IntAutoLam)+5], marker = '.', color = MTCCol)
axs[0].set_xlabel(r'the noise precision $\gamma$')
axs[0].set_ylabel(r'the smoothnes parameter $\delta$')
#axs[1].hist(new_lamb,bins=BinHist, color = MTCCol, zorder = 0, density = True)#10)
axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(lambBinEdges)[0])#10)

axs[1].plot(lambBinEdges[1::],  skew_norm_pdf(lambBinEdges[1::], *paramsSkew )/np.sum(skew_norm_pdf(lambBinEdges[1::], *paramsSkew )), zorder = 1, color =  gmresCol)#"#009E73")
axs[1].axvline( lam_opt, color = RegCol,linewidth=2)

axs[1].set_xlabel(r'the regularization parameter $\lambda =\delta / \gamma$')
axs[0].ticklabel_format(axis='y', style='sci',scilimits=(0,0))
plt.savefig('ScatterplusHisto.svg')
plt.show()




##


fig, axs = plt.subplots(3, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction))#, dpi = dpi)
n_bins = n_bins
BinSetLamb = np.arange(min(new_lamb),max(new_lamb)+ lam_opt/3,(max(new_lamb)+ lam_opt/3-min(new_lamb))/n_bins)
BinSetGam = np.arange(min(new_gam),max(new_gam),(max(new_gam)-min(new_gam))/n_bins)
BinSetDelt = np.arange(min(new_delt),max(new_delt),(max(new_delt)-min(new_delt))/n_bins)

axs[0].hist(new_gam,bins=BinSetGam, color = MTCCol, zorder = 0, label = r'\textbf{MwG}')
#axs[0].set_ylim([0,400])
axs0 = axs[0].twinx()
axs0.hist(SampParas[burnIn::math.ceil(IntAutoGamPyT),0],bins=BinSetGam,color = pyTCol, zorder = 1, label = 't-walk')
axs0.set_ylim([0,100])
axs0.tick_params(axis = 'y', colors=pyTCol, which = 'both')
axs[0].spines[:].set_visible(False)
axs0.spines['right'].set_color(pyTCol)
hist0, lab0 = axs[0].get_legend_handles_labels()
hist00, lab00 = axs0.get_legend_handles_labels()
axs[0].legend(labels = lab0 + lab00, handles = hist0+hist00  ,loc='upper right',frameon=True, fontsize = 12)#,bbox_to_anchor=(1.05, 1.15))
#axs0.legend(labels = ['t-walk'], labelcolor = [ MTCCol] ,loc='upper right', bbox_to_anchor=(1.01, 0.7),frameon=False)
#axs0.legend(labels = ['MTC','t-walk'], labelcolor = ['k', 'cyan'] ,loc='upper right', bbox_to_anchor=(1.01, 1.11),frameon=False)
axs[1].hist(new_delt,bins=BinSetDelt, color = 'k', zorder = 0)
axs[1].xaxis.set_major_formatter(scientific_formatter)
# for label in axs[1].xaxis.get_ticklabels()[::2]:
#     label.set_visible(False)
#axs[1].set_ylim([0,750])
axs1 = axs[1].twinx()
axs1.hist(deltasPyT[burnIn::math.ceil(IntAutoDeltaPyT)],bins=BinSetDelt,color = pyTCol, zorder = 1)
axs1.set_ylim([0,100])
axs1.tick_params(axis = 'y', colors=pyTCol, which = 'both')
axs[1].spines[:].set_visible(False)
axs1.spines['right'].set_color(pyTCol)
axs[2].hist(new_lamb,bins=BinSetLamb, color = MTCCol, zorder = 0)#10)
#axs[2].set_ylim([0,200])
axs2 = axs[2].twinx()
LPYT = SampParas[burnIn::math.ceil(IntAutoLamPyT),1]
axs2.hist(SampParas[burnIn::math.ceil(IntAutoLamPyT),1] ,bins=BinSetLamb,color = pyTCol, zorder = 1)
axs[2].axvline( lam_opt, color = "#D55E00",linewidth=7.0)
#axs[2].set_xlim([0, lam_opt+50])
axs2.set_ylim([0,100])
axs2.tick_params(axis = 'y', colors=pyTCol, which = 'both')
axs[2].spines[:].set_visible(False)
axs2.spines['right'].set_color(pyTCol)
axs[0].set_title(r'$\gamma$, the noise precision', fontsize = 12)
#axs[0].set_xlabel(r'$\gamma$, the noise precision', fontsize = 12)
axs[1].set_title(r'$\delta$, the prior precision', fontsize = 12)
axs[2].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)
#fig.savefig('AllHistoResults.pgf', bbox_inches='tight')
plt.savefig('AllHistoResults.png')
plt.show()

##




###
plt.close('all')

TrueCol = [50/255,220/255, 0/255]#'#02ab2e'
Sol= Results[2,:]/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
x = np.mean(Results,0 )/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
#xerr = np.sqrt(np.var(Results / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst), 0)) / 2
xerr = np.sqrt(np.var(Results,0)/(num_mole *S[ind,0]  * f_broad * 1e-4 * scalingConst)**2)/2
XOPT = x_opt /(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
MargX = MargInteg/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)

fig3, ax2 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
 # ax1 and ax2 share y-axis
line3 = ax2.scatter(y, tang_heights_lin, label = r'data $\bm{y}$', zorder = 0, marker = '*', color =DatCol )#,linewidth = 5

ax1 = ax2.twiny()
#ax1.scatter(VMR_O3,height_values,marker = 'o', facecolor = 'None', color = "#009E73", label = 'true profile', zorder=1, s =12)#,linewidth = 5)
ax1.plot(VMR_O3,height_values,marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 1.5, markersize =7)

# edgecolor = [0, 158/255, 115/255]
#line1 = ax1.plot(VMR_O3,height_values, color = [0, 158/255, 115/255], linewidth = 10, zorder=0)
for n in range(0,paraSamp,35):
    Sol = Results[n, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

    ax1.plot(Sol,height_values,marker= '+',color = ResCol,label = r'$\bm{x} \sim \pi(\bm{x}|\bm{y}, \bm{\theta})$', zorder = 1, linewidth = 0.5, markersize = 5)
    with open('Samp' + str(n) +'.txt', 'w') as f:
        for k in range(0, len(Sol)):
            f.write('(' + str(Sol[k]) + ' , ' + str(height_values[k]) + ')')
            f.write('\n')

# ax1.plot(Sol, height_values, marker='+', color=ResCol, label='posterior samples ', zorder=4, linewidth=0.5,
# markersize=2, linestyle = 'none')
#$\mathbf{x} \sim \pi(\mathbf{x} |\mathbf{y}, \mathbf{\theta} ) $' , markerfacecolor = 'none'
ax1.plot(XOPT, height_values, markerfacecolor = 'none', markeredgecolor = RegCol, color = RegCol ,marker='v', zorder=1, label=r'$\bm{x}_{\lambda}$', markersize =8, linewidth = 2 )# color="#D55E00"
#line2 = ax1.errorbar(x,height_values,capsize=5, yerr = np.zeros(len(height_values)) ,color = MTCCol,zorder=5,markersize = 5, fmt = 'o',label = r'$\mathbf{x} \sim \pi(\mathbf{x} |\mathbf{y}, \mathbf{\theta} ) $')#, label = 'MC estimate')

line3 = ax1.plot(MargX,height_values, markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.',  label = r'$\text{E}_{\bm{x},\bm{\theta}|\bm{y}} [\bm{x}]$', markersize =3, linewidth =1)#, markerfacecolor = 'none'
line3 = ax1.errorbar(MargX,height_values,yerr = np.sqrt(otherVar)/2 , markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.', markersize =3, linewidth =1)#, markerfacecolor = 'none'

#E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[h(\mathbf{x})]$
# markersize = 6
#line4 = ax1.errorbar(x, height_values,capsize=5, xerr = xerr,color = MTCCol, fmt = 'o', markersize = 5,zorder=5)#, label = 'MC estimate')
#line5 = ax1.errorbar(MargX,height_values, color = MargCol, markeredgecolor= MargCol, capsize=5,  markersize = 6, zorder=3, fmt = 's')
#xerr =MargXErr/2,yerr = np.zeros(len(height_values))



#line5 = ax1.plot(x_opt/(num_mole * S[ind,0] * f_broad * 1e-4 * scalingConst),height_values, color = 'crimson', linewidth = 7, label = 'reg. sol.', zorder=1)

ax1.set_xlabel(r'Ozone volume mixing ratio ')
#multicolor_ylabel(ax1,('(Tangent)','Height in km'),('k', dataCol),axis='y')
ax2.set_ylabel('(Tangent) Height in km')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Handles = [handles[0], handles[1], handles[2]]
# Labels =  [labels[0], labels[1], labels[2]]
# LegendVertical(ax1, Handles, Labels, 90, XPad=-45, YPad=12)

legend = ax1.legend(handles = [handles[-3], handles2[0], handles[0],handles[-2],handles[-1]])# loc='lower right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

#plt.ylabel('Height in km')
ax1.set_ylim([heights[minInd-1], heights[maxInd+1]])
#ax2.set_xlim([min(y),max(y)])
#ax1.set_xlim([min(x)-max(xerr)/2,max(x)+max(xerr)/2]) Ozone


ax2.set_xlabel(r'Spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
ax2.tick_params(colors = DatCol, axis = 'x')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)
#ax2.spines['top'].set_color(pyTCol)
fig3.savefig('FirstRecRes.svg')
plt.show()






Samp = Results[::15,:] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

np.savetxt('14Samples.txt', Samp, fmt = '%.15f', delimiter= '\t')
#np.savetxt('GroundTruth.txt',VMR_O3, fmt = '%.15f', delimiter= '\t')
#np.savetxt('RegSolution.txt', XOPT, fmt = '%.15f', delimiter= '\t')
#np.savetxt('PosteriorMean.txt', MargX, fmt = '%.15f', delimiter= '\t')
#np.savetxt('HeigthValues.txt', height_values, fmt = '%.15f', delimiter= '\t')

##
with open('RegSolution.txt', 'w') as f:
    for n in range(0,len(XOPT)):
        f.write('(' + str(XOPT[n]) + ' , ' + str(height_values[n]) + ')')
        f.write('\n')

with open('GroundTruth.txt', 'w') as f:
    for n in range(0,len(VMR_O3)):
        f.write('(' + str(VMR_O3[n]) + ' , ' + str(height_values[n]) + ')')
        f.write('\n')

with open('PosteriorMean.txt', 'w') as f:
    for n in range(0,len(MargX)):
        f.write('(' + str(MargX[n]) + ' , ' + str(height_values[n]) + ')')
        f.write('\n')

with open('SimData.txt', 'w') as f:
    for n in range(0,len(y)):
        f.write('(' + str(y[n,0]) + ' , ' + str(tang_heights_lin[n]) + ')')
        f.write('\n')


##




OptRes = x_opt/(num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)

#plt.rcParams["font.serif"] = "cmr"
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
line1 = ax1.plot(VMR_O3,height_values, color = [0, 158/255, 115/255], linewidth = 7, label = 'True VMR of O$_3$', zorder=0)

#ax1.plot(Sol,height_values)
for n in range(0,paraSamp,4):
    Sol = Results[n, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)
    ax1.plot(Sol,height_values, linewidth = 0.2, color = MTCCol )
line2 = ax1.errorbar(x,height_values,capsize=4, yerr = np.zeros(len(height_values)) ,color = MTCCol, fmt = '-o',label = 'MTC RTO method ')#, label = 'MC estimate')
#line3 = ax1.errorbar(MargX,height_values, color = MTCCol, capsize=4, yerr = np.zeros(len(height_values)), fmt = '-x', label = r'MTC E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[h(\mathbf{x})]$')
line4 = ax1.errorbar(x, height_values,capsize=4, xerr = xerr,color = MTCCol, fmt = '-o', mec='cyan', ecolor ='cyan')#, label = 'MC estimate')
#line5 = ax1.errorbar(MargX,height_values, color = MTCCol, capsize=4, xerr =MargXErr/2, zorder=5, fmt = '-x')

#line6 = ax1.plot(OptRes,height_values, color = 'red', linewidth = 2, label = 'Regularized Solution', marker = 'o')


ax2 = ax1.twiny() # ax1 and ax2 share y-axis
line3 = ax2.plot(y, tang_heights_lin, color = dataCol, label = r'Data',linewidth = 5, zorder = 0)

#ax2.set_xlabel(r'Spectral Ozone radiance in $\frac{W}{m^2 sr} \times \frac{1}{\frac{1}{cm}}$',labelpad=10 )# color =dataCol,
#ax2.tick_params(colors = dataCol)
ax1.set_xlabel(r'Ozone volume mixing ratio ')
#multicolor_ylabel(ax1,('(Tangent)','Height in km'),('k', dataCol),axis='y')
#ax1.set_ylabel('Tangent Height in km')
ax1.set_ylabel('Height in km')
#handles, labels = ax1.get_legend_handles_labels()
#handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend()
#legend = ax1.legend(handles = [handles[0], handles2[0]], loc='upper right')#, bbox_to_anchor=(1.01, 1.01), frameon =True)

ax1.set_ylim([heights[minInd-1], heights[maxInd+1]])


#fig3.savefig('OptRecov.png')#, dpi = dpi)
#fig3.savefig('TrueRecocovMean.png')#, dpi = dpi)
fig3.savefig('TrueRecocovRTOData.png')#, dpi = dpi)
#fig3.savefig('Data.png')#, dpi = dpi)
plt.show()



print('bla')