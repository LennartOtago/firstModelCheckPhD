import numpy as np
import matplotlib as mpl

#from importetFunctions import *
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
betaG = 1e-10
betaD = 1e-10
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


tol = 1e-8
dat = np.loadtxt('/Users/lennart/PycharmProjects/openData/testProf.txt')
#dat = np.loadtxt('/home/lennartgolks/PycharmProjects/openData/testProf.txt')
press = dat[0,:]
O3 = dat[1,:]
O3[O3 < 0] = 0
# df = pd.read_excel('ExampleOzoneProfiles.xlsx')
#
# #print the column names
# print(df.columns)
#
# #get the values for a given column
# press = df['Pressure (hPa)'].values #in hectpascal or millibars
# O3 = df['Ozone (VMR)'].values


minInd = 35#2
minInd = 2
maxInd = 54
maxInd = 45
skipInd = 1
pressure_values = press[minInd:maxInd][::skipInd]#press[minInd:maxInd]
VMR_O3 = O3[minInd:maxInd][::skipInd]#O3[minInd:maxInd]
scalingConstkm = 1e-3

fig3, axs = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
axs.plot(VMR_O3, pressure_values)
axs.invert_yaxis()
axs.set_yscale('log')
plt.show()

# def height_to_pressure(p0, x, dx):
#     R = constants.gas_constant
#     R_Earth = 6356#6371  # earth radiusin km
#     grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
#     temp = get_temp(x)
#     return p0 * np.exp(-28.97 * grav / R * dx/temp )

def pressure_to_height(p0, pplus, x):
    R = constants.gas_constant
    R_Earth =6356# 6371  # earth radiusin km6356#
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    temp = get_temp(x)
    dP = pplus - p0
    #return np.log(pplus/p0) /(-28.97 * grav / R /temp )

    if x > 80:
       M = 28.97 * 0.995
    else:
        M = 28.97
    return (dP/p0) /(-M * grav / R /temp )

calc_press = np.zeros((len(press)+1,1))
calc_press[0] = 1013.25
calc_press[1:] = press.reshape((len(press),1)) #hPa
actual_heights = np.zeros(len(press)+1)

for i in range(1,len(calc_press)):
    dx = pressure_to_height(calc_press[i-1], calc_press[i], actual_heights[i-1])
    actual_heights[i] = actual_heights[i - 1] + dx


""" analayse forward map without any real data values"""
heights = actual_heights[1:]
SpecNumLayers = len(VMR_O3)
#height_values = heights[minInd:maxInd].reshape((SpecNumLayers,1))
height_values = np.around(heights[minInd:maxInd][::skipInd].reshape((SpecNumLayers,1)),2)
#height_values = np.around(heights[minInd:][::skipInd].reshape((SpecNumLayers,1)),2)
MinH = height_values[0]
MaxH = height_values[-1]
R_Earth =6356# 6371 # earth radiusin km
ObsHeight = 500 # in km



fig3, axs = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
axs.plot(O3, heights)
plt.show()


''' do svd for one specific set up for linear case and then exp case'''

# find minimum and max angle in radians
# min and max angle are defined by the height values of the retrived profile
MaxAng = np.arcsin((height_values[-1]+ R_Earth) / (R_Earth + ObsHeight))
#MaxAng = [np.arcsin((55+ R_Earth) / (R_Earth + ObsHeight))]

MinAng = np.arcsin((height_values[0] + R_Earth) / (R_Earth + ObsHeight))




pointAcc = 0.00045#0.00075#
meas_ang = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))

SpecNumMeas = len(meas_ang)
m = SpecNumMeas

A_lin_dx, tang_heights_lin, extraHeight = gen_forward_map(meas_ang,height_values,ObsHeight,R_Earth)

A_lin = gen_sing_map(A_lin_dx, tang_heights_lin, height_values)


#condition number for A
A_lin = A_lin
A_linu, A_lins, A_linvh = np.linalg.svd(A_lin)
cond_A_lin =  np.max(A_lins)/np.min(A_lins)
print("normal: " + str(orderOfMagnitude(cond_A_lin)))



#to test that we have the same dr distances

tot_r = np.zeros((SpecNumMeas,1))
#calculate total length
for j in range(0, SpecNumMeas):
    tot_r[j] =  np.sqrt( ( height_values[-1] + R_Earth)**2 - (tang_heights_lin[j] +R_Earth )**2)
print('Distance through layers check: ' + str(np.allclose( sum(A_lin_dx.T,0), tot_r[:,0])))



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
startInd = 38
L[startInd::, startInd::] = L[startInd::, startInd::] * 10
L[startInd, startInd] = -L[startInd, startInd-1] - L[startInd, startInd+1] #-L[startInd, startInd-2] - L[startInd, startInd+2]


#cholesky decomposition of L for W1 and v1
lowC_L = scy.linalg.cholesky(L, lower = True)


#taylor exapnsion for f to do so we need y (data)

''' load data and pick wavenumber/frequency'''
# check absoprtion coeff in different heights and different freqencies
filename = 'tropical.O3.xml'


#VMR_O3, height_values, pressure_values = testReal.get_data(filename, ObsHeight * 1e3)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
#height_values = np.around(height_values * 1e-3,2)#in km 1e2 # in cm
#d_height = (height_values[1::] - height_values[0:-1] )
#d_height = layers[1::] - layers[0:-1]
N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann #* 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3

# https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
temperature = get_temp_values(heights)
#temp_values = temperature[minInd:maxInd]
temp_values = temperature[minInd:maxInd][::skipInd]

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



fig, axs = plt.subplots(tight_layout=True, figsize=set_size(PgWidthPt, fraction=fraction))
#plt.plot(press/1013.25,heights, label = 'pressure in hPa/' + str(np.around(max(press),3)) )
#plt.plot(Source/max(Source),height_values, label = r'Source in $\frac{W}{m^2 sr}\frac{1}{\frac{1}{cm}}$/' + str(np.around(max(Source[0]),5)) )
plt.plot(temperature,heights, color = 'darkred')# label = r'Source in K/' + str(np.around(max(temperature[0]),3)) )

plt.plot(temp_values,height_values, color = 'red')# label = r'Source in K/' + str(np.around(max(temperature[0]),3)) )

#plt.plot(LineInt,heights[minInd:maxInd], color = 'darkred')# label = r'Source in K/' + str(np.around(max(temperature[0]),3)) )
#axs.legend()
axs.tick_params(axis = 'x', labelcolor="darkred")
ax2 = axs.twiny() # ax1 and ax2 share y-axis
line3 = ax2.plot(press,heights, color = 'blue') #, label = 'pressure in hPa/' + str(np.around(max(press),3)) )
ax2.spines['top'].set_color('blue')
ax2.tick_params(labelcolor="blue")
ax2.set_xlabel('Pressure in hPa')
axs.set_ylabel('Height in km')
axs.set_xlabel('Temperature in K')
#axs.set_xlabel('Line intensity in cm / molecule')
#axs.set_title()
plt.savefig('PandQ.png')
plt.show()


AO3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, temp_values)
A = 2*AO3

ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(As)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))
#theta[0] = 0
#theta[-1] = 0
#Ax = np.matmul(A, theta)
Ax =np.matmul(A, VMR_O3 * theta_scale_O3)
#convolve measurements and add noise
#y = add_noise(Ax, 0.01)
#y[y<=0] = 0
SNR = 60

y, gam0 = add_noise(Ax.reshape((SpecNumMeas,1)), SNR)


fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)
ax1.scatter(y, tang_heights_lin, color = 'r')
plt.show()


signal_power = np.sqrt(np.mean(np.abs(Ax) ** 2))
noise = np.random.normal(0, np.sqrt(1 / gam0), size =y.shape)
noise_power = np.sqrt(np.mean(np.abs(noise) ** 2))
snr = signal_power / noise_power

nonLinA = calcNonLin(tang_heights_lin, A_lin_dx, height_values, pressure_values, ind, temp_values, VMR_O3, AscalConstKmToCm, wvnmbr, S, E,g_doub_prime)
OrgData = np.matmul(AO3 * nonLinA,VMR_O3 * theta_scale_O3)
DatErr = np.linalg.norm( OrgData -  Ax) / np.linalg.norm(OrgData) * 100
print('DataErr '+ str(DatErr))

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin, label = 'linear Data')
ax1.plot(OrgData , tang_heights_lin, label = 'nonlinear Data')
ax1.scatter(y, tang_heights_lin, color = 'r')
ax1.legend()
plt.show()

noise = np.random.normal(0, np.sqrt(1 / gam0), size = OrgData.shape)
nonLinY = (OrgData + noise).reshape((SpecNumMeas,1))
y = (Ax + noise).reshape((SpecNumMeas,1))


## some modifiaction
# to find correaltion bewteen tmep and others

y = np.copy(nonLinY)

new_temp_values = np.mean(temp_values) * np.ones((SpecNumLayers,1))
AO3, theta_scale_O3 = composeAforO3(A_lin, new_temp_values, pressure_values, ind, new_temp_values)
A = 2*AO3
ATA = np.matmul(A.T,A)
Ax =np.matmul(A, VMR_O3 * theta_scale_O3)


fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin, color = 'g')
ax1.scatter(y, tang_heights_lin, color = 'r')
ax1.plot(y, tang_heights_lin, color = 'r')
ax1.scatter(nonLinY, tang_heights_lin, color = 'k')
ax1.plot(nonLinY, tang_heights_lin, color = 'k')

plt.show()

#y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
ATy = np.matmul(A.T, y)

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)
ax1.scatter(y, tang_heights_lin)
ax1.plot(y, tang_heights_lin)
plt.show()
#print(1/np.var(y))


fig3, axs = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
axs.plot(VMR_O3, height_values)
#axs.invert_yaxis()
#axs.set_yscale('log')
plt.show()

fig3, axs = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
axs.plot(pressure_values, height_values)
#axs.invert_yaxis()
axs.set_xscale('log')
plt.show()


##
"""start the mtc algo with first guesses of noise and lumping const delta"""

theta = VMR_O3 * theta_scale_O3
vari = np.zeros((len(theta)-2,1))

for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])
    #vari[j - 1] = abs(-theta[j + 1] + 2*theta[j] - theta[j - 1])**2

##
#find minimum for first guesses
'''params[1] = delta
params[0] = gamma'''
def MinLogMargPost(params):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    lamb = params[1]
    if lamb < 0  or gam < 0:
        return np.nan

    n = SpecNumLayers
    m = SpecNumMeas

    Bp = ATA + lamb * L

    LowTri = np.linalg.cholesky(Bp)
    UpTri = LowTri.T
    # check if L L.H = B
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MinLogMargPost, [gam0,1/gam0* 1/ np.mean(vari)/15], maxiter = 25)
gamma0 = minimum[0]
lam0 = minimum[1]
print(minimum)


##taylor series arounf lam_0

B = (ATA + lam0 * L)

LowTri = np.linalg.cholesky(B)
UpTri = LowTri.T
# check if L L.H = B
B_inv_A_trans_y0 = lu_solve(LowTri, UpTri,  ATy[0::, 0])


B_inv_L = np.zeros(np.shape(B))

for i in range(len(B)):
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    B_inv_L[:, i] = lu_solve(LowTri, UpTri,  L[:, i])

B_inv_L_2 = np.matmul(B_inv_L, B_inv_L)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L)
B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L)
B_inv_L_6 = np.matmul(B_inv_L_4, B_inv_L_2)


f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L), B_inv_A_trans_y0)
f_0_2 = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y0)
f_0_3 = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_3) ,B_inv_A_trans_y0)
f_0_4 = 0#-1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y0)
f_0_5 = 0#1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_5) ,B_inv_A_trans_y0)
f_0_6 = 0#-1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_6) ,B_inv_A_trans_y0)



g_0_1 = np.trace(B_inv_L)
g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
g_0_3 = 1 /6 * np.trace(B_inv_L_3)
g_0_4 = 0#-1 /24 * np.trace(B_inv_L_4)
g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
g_0_6 = 0#-1 /720 * np.trace(B_inv_L_6)


f_0 = f(ATy, y, B_inv_A_trans_y0)

'''do the sampling'''
number_samples = 10000
burnIn = 100


alphaG = 1
alphaD = 1
rate = f_0 / 2 + betaG + betaD * lam0
# draw gamma with a gibs step
shape = SpecNumMeas/2 + alphaD + alphaG

#f_new = f_0
#g_old = g(A, L,  lambdas[0])

def MHwG(number_samples, burnIn, lam0, gamma0, f_0):
    wLam = lam0 * 0.9#8e3#7e1

    alphaG = 1
    alphaD = 1
    k = 0

    gammas = np.zeros(number_samples + burnIn)
    #deltas = np.zeros(number_samples + burnIn)
    lambdas = np.zeros(number_samples + burnIn)

    gammas[0] = gamma0
    lambdas[0] = lam0



    shape = SpecNumMeas / 2 + alphaD + alphaG
    rate = f_0 / 2 + betaG + betaD * lam0

    f_new = np.copy(f_0)
    #rate_old = np.copy(rate)
    for t in range(number_samples + burnIn-1):
        #print(t)

        # # draw new lambda
        lam_p = normal(lambdas[t], wLam)

        while lam_p < 0:
                lam_p = normal(lambdas[t], wLam)

        delta_lam = lam_p - lambdas[t]

        delta_f = f_0_1 * delta_lam + f_0_2 * delta_lam**2 + f_0_3 * delta_lam**3 #+ f_0_4 * delta_lam**4 + f_0_5 * delta_lam**5
        delta_g = g_0_1 * delta_lam + g_0_2 * delta_lam**2 + g_0_3 * delta_lam**3 #+ g_0_4 * delta_lam**4 + g_0_5 * delta_lam**5

        log_MH_ratio = ((SpecNumLayers)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

        #accept or rejeict new lam_p
        u = uniform()
        if np.log(u) <= log_MH_ratio:
            #accept
            k = k + 1
            lambdas[t + 1] = lam_p
            delta_lam = lam_p - lam0
            delta_f = f_0_1 * delta_lam + f_0_2 * delta_lam ** 2 + f_0_3 * delta_lam ** 3
            f_new = f_0 + delta_f
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



startTime = time.time()
lambdas, gammas, k = MHwG(number_samples, burnIn, lam0, gamma0, f_0)
elapsed = time.time() - startTime
print('MTC Done in ' + str(elapsed) + ' s')
print('acceptance ratio: ' + str(k/(number_samples+burnIn)))

##
startTime = time.time()
BinHistStart = 5

lambHist, lambBinEdges = np.histogram(lambdas, bins=BinHistStart, density=True)
gamHist, gamBinEdges = np.histogram(gammas, bins=BinHistStart, density=True)
MargResults = np.zeros((BinHistStart, len(theta)))
B_inv_Res = np.zeros((BinHistStart, len(theta)))
VarB = np.zeros((BinHistStart, len(L), len(L)))
gamInt = np.zeros(BinHistStart)
meanGamInt = np.zeros(BinHistStart)
IDiag = np.eye(len(L))
startTime = time.time()
for p in range(0,BinHistStart):

    SetLambda = lambBinEdges[p] + (lambBinEdges[p] + lambBinEdges[p + 1]) / 2
    SetGamma = gamBinEdges[p] + (gamBinEdges[p] + gamBinEdges[p + 1]) / 2
    SetB = ATA + SetLambda * L

    LowTri = np.linalg.cholesky(SetB)
    UpTri = LowTri.T
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

    MargResults[p, :] = B_inv_A_trans_y * lambHist[p] / np.sum(lambHist)
    B_inv_Res[p, :] = B_inv_A_trans_y

    B_inv = np.zeros(SetB.shape)
    # startTime = time.time()
    LowTri = np.linalg.cholesky(SetB)
    UpTri = LowTri.T
    for j in range(len(B)):
        B_inv[:, j] = lu_solve(LowTri, UpTri, IDiag[:, j])

    VarB[p] = B_inv * lambHist[p] / np.sum(lambHist)
    gamInt[p] = 1 / SetGamma * gamHist[p] / np.sum(gamHist)
    meanGamInt[p] = SetGamma * gamHist[p] / np.sum(gamHist)

oldMargInteg =np.sum(MargResults,0)  / theta_scale_O3
MargInteg= np.copy(oldMargInteg)
CondVar =np.sum(gamInt) * np.sum(VarB,0)  / (theta_scale_O3) ** 2

BinHist = BinHistStart
MargTime = time.time() - startTime
print('Post Mean in ' + str(MargTime) + ' s')
oldRelErr = 0
print(BinHistStart)


lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density= True)
gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)
fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction) )#, dpi = dpi)

axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], color = 'k', zorder = 0,width = np.diff(gamBinEdges)[0])#10)


axs[0].set_xlabel(r'the noise precision $\gamma$')


axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = 'k', zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_xlabel(r'$\lambda =\delta / \gamma$, the regularization parameter')
plt.savefig('AnnikaTestHistoPlot.png')
plt.show()



##
'''L-curve refularoization
'''

lamLCurve = np.logspace(1,7,200)

NormLCurve = np.zeros(len(lamLCurve))
xTLxCurve = np.zeros(len(lamLCurve))
xTLxCurve2 = np.zeros(len(lamLCurve))
for i in range(len(lamLCurve)):
    B = (ATA + lamLCurve[i] * L)

    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    x = lu_solve(LowTri, UpTri, ATy[0::, 0])

    NormLCurve[i] = np.linalg.norm( np.matmul(A,x) - y[0::,0])
    xTLxCurve[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))


startTime  = time.time()
lamLCurveZoom = np.logspace(0.2,6,200)
NormLCurveZoom = np.zeros(len(lamLCurve))
xTLxCurveZoom = np.zeros(len(lamLCurve))
for i in range(len(lamLCurveZoom)):
    B = (ATA + lamLCurveZoom[i] * L)

    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    x = lu_solve(LowTri, UpTri, ATy[0::, 0])

    NormLCurveZoom[i] = np.linalg.norm( np.matmul(A,x) - y[0::,0])
    xTLxCurveZoom[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))



import kneed

# calculate and show knee/elbow
kneedle = kneed.KneeLocator(NormLCurveZoom, xTLxCurveZoom, curve='convex', direction='decreasing', online = True, S = 1, interp_method="interp1d")
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
#x_opt, exitCode = gmres(B, ATy[0::, 0], rtol=tol, restart=25)
LowTri = np.linalg.cholesky(B)
UpTri = LowTri.T
x_opt = lu_solve(LowTri, UpTri, ATy[0::, 0])
LNormOpt = np.linalg.norm( np.matmul(A,x_opt) - y[0::,0])#, ord = 2)
xTLxOpt = np.sqrt(np.matmul(np.matmul(x_opt.T, L), x_opt))




fig, axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
axs.scatter(NormLCurve,xTLxCurve, zorder = 0, color =  DatCol, s = 5, marker ='s')


axs.scatter(knee_point, kneedle.knee_y, color = regCol, marker = 'v',label = 'max. curvature', s= 50,zorder=1)


axs.set_xscale('log')
axs.set_yscale('log')
axs.set_ylabel(r'$ \sqrt{\mathbf{x}^T \mathbf{L}\mathbf{x}}$', style='italic')
axs.set_xlabel(r'$|| \mathbf{Ax} - \mathbf{y}||$')

handles, labels = axs.get_legend_handles_labels()

#axs.legend(handles = [handles[0],handles[1],handles[2]],loc = 'upper right',  frameon =True)
plt.savefig('AnnikaTestLCurve.svg')

plt.show()


##
TrueCol = [50/255,220/255, 0/255]#'#02ab2e'

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 1.5, markersize =7)

line3 = ax1.errorbar(MargInteg,height_values[:,0],xerr = np.sqrt(np.diag(CondVar)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =3, linewidth =1, capsize = 3)#, markerfacecolor = 'none'
ax1.errorbar(MargInteg,height_values[:,0],  yerr = np.zeros(len(height_values)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', label = r'posterior $\mu \pm \sigma$ ', markersize =3, linewidth =1, capsize = 3)

ax1.plot(x_opt/theta_scale_O3,height_values[:,0],marker = 's',markerfacecolor = regCol, color = regCol , label = r'true $\bm{x}$', zorder=2 ,linewidth = 1.5, markersize =5)

ax1.set_xlabel(r'ozone volume mixing ratio ')

ax1.set_ylabel('height in km')
handles, labels = ax1.get_legend_handles_labels()

ax1.set_ylim([height_values[0], height_values[-1]])
ax1.legend()

fig3.savefig('AnnikaTestRes.svg')
plt.show()
