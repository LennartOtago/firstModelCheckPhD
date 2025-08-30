import numpy as np
import matplotlib as mpl
from puwr import tauint, correlated_data
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
betaG = 1e-35
betaD = 1e-35
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

df = pd.read_excel('ExampleOzoneProfiles.xlsx')

#print the column names
print(df.columns)

#get the values for a given column
press = df['Pressure (hPa)'].values #in hectpascal or millibars
O3 = df['Ozone (VMR)'].values
O3[O3<0] = 0
dir = '/home/lennartgolks/PycharmProjects/'
dir = '/Users/lennart/PycharmProjects/'
dat = np.loadtxt(dir + 'openData/testProf.txt')
#dat = np.loadtxt('/home/lennartgolks/PycharmProjects/openData/testProf.txt')
press = dat[0,:]
O3 = dat[1,:]
O3[O3 < 0] = 0


minInd = 5
maxInd = 50#47#51#54#47
skipInd = 1
pressure_values = press[minInd:maxInd][::skipInd]#press[minInd:maxInd]
VMR_O3 = O3[minInd:maxInd][::skipInd]#O3[minInd:maxInd]
scalingConstkm = 1e-3

# def height_to_pressure(p0, x, dx):
#     R = constants.gas_constant
#     R_Earth = 6356#6371  # earth radiusin km
#     grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
#     temp = get_temp(x)
#     return p0 * np.exp(-28.97 * grav / R * dx/temp )

def height_pressure(x, dx, p0):
    R = constants.gas_constant
    R_Earth = 6356#6371  # earth radiusin km6356#
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    temp = temp_func(x)
    #dP = pplus - p0
    M = 28.97
    return dx * (-M * grav / R /temp ) * p0, temp

def pressure_to_height(p0, pplus, x):
    R = constants.gas_constant
    R_Earth = 6356#6371  # earth radiusin km6356#
    grav = 9.81 * ((R_Earth)/(R_Earth + x))**2
    #temp = get_temp(x)
    temp = temp_func(x)
    dP = pplus - p0
    #return - np.log(p0/pplus) /(-28.97 * grav / R /temp )
    #return ( np.log(pplus) -np.log(p0))/ (-28.97 * grav / R / temp)
    M = 28.97
    return (dP/p0) /(-M * grav / R /temp ), temp

# til max 84.8520 geopotential height
#geopotential to geometric
#H = Z * R_Earth/(Z + R_Earth)
#H = Z * R_Earth/(Z + R_Earth)
R_Earth = 6356
H = 11
print(H * R_Earth/(R_Earth -H ))
Z = 91
R_Earth = 6356
print( Z * R_Earth/(Z + R_Earth))
calc_press = np.zeros((len(press)+1,1))
calc_press[0] = 1013.25
calc_temp = np.zeros((len(press),1))
calc_temp[0] = 288.15
calc_press[1:] = press.reshape((len(press),1)) #hPa
calc_press = np.copy(press) #hPa
calc_press[0] = 1013.25
actual_heights = np.zeros(len(press)+1)
actual_heights = np.zeros(len(press))
for i in range(1,len(calc_press)):
    dx, calc_temp[i-1] = pressure_to_height(calc_press[i-1], calc_press[i], actual_heights[i-1])
    actual_heights[i] = actual_heights[i - 1] + dx

calc_temp[-1] = temp_func( actual_heights[-1])

""" analayse forward map without any real data values"""
heights = actual_heights#[1:]
SpecNumLayers = len(VMR_O3)
#height_values = heights[minInd:maxInd].reshape((SpecNumLayers,1))
#height_values = np.around(heights[minInd:maxInd][::skipInd].reshape((SpecNumLayers,1)),2)
#temp_values =  np.around(calc_temp[minInd:maxInd][::skipInd],2)

MinH = min(heights[minInd:maxInd])
MaxH = max(heights[minInd:maxInd])
##
height_values = np.linspace(MinH, MaxH, SpecNumLayers).reshape(SpecNumLayers)
VMR_O3 = np.interp(height_values,  heights, O3,).reshape((SpecNumLayers,1))
new_calc_press  = np.zeros( SpecNumLayers)
new_calc_press[0]  = calc_press[minInd]
new_calc_temp = np.zeros( SpecNumLayers)
for i in range(1, SpecNumLayers):
    dx = height_values[i - 1] - height_values[i]
    dp, new_calc_temp[i-1] =  height_pressure(height_values[i - 1], dx, new_calc_press[i - 1] )
    new_calc_press[i] = new_calc_press[i - 1] - dp

new_calc_temp[-1] = temp_func( height_values[-1])
height_values = np.around(height_values,2).reshape((SpecNumLayers,1))
temp_values = np.around(new_calc_temp,2).reshape((SpecNumLayers,1))
pressure_values = new_calc_press.reshape(SpecNumLayers)

startInd = 23
EndInd = len(height_values[startInd::2]) + startInd
height_values[startInd:EndInd] = height_values[startInd::2]
temp_values[startInd:EndInd] = temp_values[startInd::2]
pressure_values[startInd:EndInd] = pressure_values[startInd::2]
VMR_O3[startInd:EndInd] = VMR_O3[startInd::2]

height_values = height_values[:EndInd]
temp_values = temp_values[:EndInd]
pressure_values = pressure_values[:EndInd]
VMR_O3 = VMR_O3[:EndInd]
SpecNumLayers = len(height_values)
##
#temp_values = np.interp(np.linspace(MinH, MaxH, SpecNumLayers).reshape(SpecNumLayers), heights, calc_temp.reshape(len(O3))).reshape((SpecNumLayers,1))
#pressure_values  = np.interp(np.linspace(MinH, MaxH, SpecNumLayers).reshape(SpecNumLayers), heights, calc_press).reshape(SpecNumLayers)

R_Earth = 6356#6371 # earth radiusin km
ObsHeight = 500 # in km

# def pressFunc(x, b, h0, p0):
#     return np.exp(-b * (x -h0)  + np.log(p0))
#
# def pressFunc(x, b, p0):
#     return np.exp(-b * x  + np.log(p0))
#
# def pressFuncFullFit(x, b1, b2, h0, p0):
#     b = np.ones(len(x))
#     b[x<=h0] = b1
#     b[x>h0] = b2
#     return np.exp(-b * (x -h0)  + np.log(p0))
#
#
# popt, pcov = scy.optimize.curve_fit(pressFunc, height_values[:,0], pressure_values, p0=[1.5e-1, pressure_values[0]])
# #popt, pcov = scy.optimize.curve_fit(pressFunc,  actual_heights,calc_press, p0=[1.5e-1, pressure_values[0]])
#
# print(popt)
# poptFull, pcov = scy.optimize.curve_fit(pressFuncFullFit, actual_heights,calc_press, p0=[1.5e-1,1.5e-1, 8, pressure_values[0]])
# poptFull, pcov = scy.optimize.curve_fit(pressFuncFullFit, height_values[:,0],pressure_values, p0=[1.5e-1,1.5e-1, 8, pressure_values[0]])
#
# print(poptFull)
#
fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)


#axs.plot(pressFunc(height_values[:,0], *popt),height_values[:,0], label = 'not full')
#axs.plot(pressFuncFullFit(height_values[:,0], *poptFull),height_values[:,0], label = 'full')
axs.plot(pressure_values,height_values[:,0], label = 'true')

axs.axvline(0.02)
axs.set_xlabel(r'pressure in hPa ')
axs.set_yscale('log')
axs.legend()
plt.show()

#
# tests = 1000
#
# means = np.zeros(2)
# sigmas = np.zeros(2)
# means[0] = popt[0]
# means[1] = popt[1]
# #means[2] = popt[2]
# ##means[3] = popt[3]
#
# sigmaP =  4# * 2
# sigmaH = 0.2*3
# sigmaGrad2 = 0.0001*5#0.01 #* 5
# sigmas[0] = sigmaGrad2
# #sigmas[1] = sigmaGrad2
# sigmas[1] = sigmaP
# #sigmas[2] = sigmaP
#
# PriorSamp = np.random.multivariate_normal(means, np.eye(2) * sigmas, tests)
#
#
# fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)
# ZeroP = np.zeros(tests)
# for i in range(0,tests):
#     ZeroP[i] = pressFunc(0, *PriorSamp[i,:])
#
# axs.hist(ZeroP, bins=n_bins)
#
# axs.set_xlabel(r'pressure in hPa ')
#
# plt.show()
# ##
# fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)
# axs.plot( pressure_values,height_values,marker = 'o',markerfacecolor = 'C1', color = 'C1' , label = 'true profile', zorder=0,linewidth = 3, markersize =15)
# axs.plot( calc_press,actual_heights,marker = 'o',markerfacecolor = 'C2', color = 'C2' , label = 'true profile', zorder=1,linewidth = 3, markersize =10)
#
# Sol = pressFunc(height_values[:, 0], *popt)
# axs.plot(Sol, height_values, markeredgecolor='C3', color='C3', zorder=2, marker='.', markersize=2, linewidth=1 )
#
# Sol = pressFuncFullFit(actual_heights, *poptFull)
# axs.plot(Sol, actual_heights, markeredgecolor='C4', color='C4', zorder=2, marker='.', markersize=2, linewidth=1 )
#
# axs.set_xlabel(r'pressure in hPa')
#
# axs.set_ylabel(r'height in km')
#
# plt.show()
#
# fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)
#
# axs.plot( calc_temp,actual_heights,marker = 'o',markerfacecolor = 'C2', color = 'C2' , label = 'true profile', zorder=0,linewidth = 3, markersize =10)
# axs.plot( temp_func(actual_heights),actual_heights,marker = 'o',markerfacecolor = 'C1', color = 'C1' , label = 'true profile', zorder=1,linewidth = 2, markersize =5)
# axs.plot( temp_values,height_values,marker = 'o',markerfacecolor = 'C3', color = 'C3' , label = 'true profile', zorder=1,linewidth = 2, markersize =5)
#
# axs.set_xlabel(r'Temperature in K')
#
# axs.set_ylabel(r'height in km')
#
# plt.show()
##
#np.savetxt('height_values.txt', height_values, fmt = '%.30f', delimiter= '\t')
#np.savetxt('pressure_values.txt', pressure_values, fmt = '%.30f', delimiter= '\t')

''' do svd for one specific set up for linear case and then exp case'''

# find minimum and max angle in radians
# min and max angle are defined by the height values of the retrived profile
MaxAng = np.arcsin((height_values[-1]+ R_Earth) / (R_Earth + ObsHeight))
#MaxAng = [np.arcsin((55+ R_Earth) / (R_Earth + ObsHeight))]

MinAng = np.arcsin((height_values[0] + R_Earth) / (R_Earth + ObsHeight))


##
pointAcc = 0.00075#0.00045
meas_ang = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))
b = 0.3
meas_ang1 = np.array(np.exp(b * np.linspace(0,len(meas_ang)-1 ,len(meas_ang))))
meas_ang1 = np.flip(meas_ang[-1] - (meas_ang[-1]-meas_ang[0]) * meas_ang1/np.max(meas_ang1))
A_lin_dx1, tang_heights_lin1, extraHeight = gen_forward_map(meas_ang1,height_values,ObsHeight,R_Earth)
b = -0.3
meas_ang2 = np.array(np.exp(b * np.linspace(0,len(meas_ang)-1 ,len(meas_ang))))
meas_ang2 = np.flip(meas_ang[0] + (meas_ang[-1]-meas_ang[0]) * meas_ang2/np.max(meas_ang2))
A_lin_dx2, tang_heights_lin2, extraHeight = gen_forward_map(meas_ang2,height_values,ObsHeight,R_Earth)

pointAcc = 0.00075 / 2
meas_ang3 = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))
A_lin_dx3, tang_heights_lin3, extraHeight = gen_forward_map(meas_ang3,height_values,ObsHeight,R_Earth)

pointAcc = 0.00055
meas_ang4 = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))
A_lin_dx4, tang_heights_lin4, extraHeight = gen_forward_map(meas_ang4,height_values,ObsHeight,R_Earth)



np.savetxt('ExpIncmeasAng.txt', meas_ang1, fmt = '%.30f', delimiter = '\t')
np.savetxt('ExpDecmeasAng.txt', meas_ang2, fmt = '%.30f', delimiter = '\t')
np.savetxt('measAngNormal.txt', meas_ang, fmt = '%.30f', delimiter = '\t')
##
pointAcc = 0.00085
meas_angChosen = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))[:25]
A_lin_dx, tang_heights_linChosen, extraHeight = gen_forward_map(meas_angChosen,height_values,ObsHeight,R_Earth)

pointAcc = 0.00075
meas_angNormal = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))
A_lin_dx, tang_heights_linNormal, extraHeight = gen_forward_map(meas_angNormal,height_values,ObsHeight,R_Earth)



#pointAcc = 0.002#0.00085#0.00075#0.00045
#meas_ang3 = np.array(np.arange(MinAng[0], MaxAng[0]-(MaxAng[0]- MinAng[0])/2, pointAcc))
#meas_ang2 = np.array(np.arange(MaxAng[0]-(MaxAng[0]- MinAng[0])/5,MaxAng[0], pointAcc))
#meas_ang =np.append(meas_ang1,meas_ang2)
#meas_ang = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))#[:25]
SpecNumMeas = len(meas_angChosen)
m = SpecNumMeas

A_lin_dx, tang_heights_lin, extraHeight = gen_forward_map(meas_angChosen,height_values,ObsHeight,R_Earth)
np.savetxt('tang_heights_lin.txt',tang_heights_lin, fmt = '%.15f', delimiter= '\t')
np.savetxt('A_lin_dx.txt',A_lin_dx, fmt = '%.15f', delimiter= '\t')

A_lin = gen_sing_map(A_lin_dx, tang_heights_lin, height_values)

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
#ax1.scatter(range(len(meas_ang)),meas_ang1, label = 'case 1')
ax1.scatter(range(len(tang_heights_lin)),tang_heights_lin, label = 'case 2')
#ax1.scatter(range(len(meas_ang)),meas_ang, s = 10, label = 'case 3')
plt.show(block = True)

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

# U, SingS, Vh = np.linalg.svd(A_lin , full_matrices=True)
#
# fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
#
# ax1.scatter(range(len(SingS)), SingS )
#
# ax1.set_yscale('log')
#
# plt.show(block = True)
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
    #neigbours[i] =  i + 1
    #neigbours[i] = i-2, i-1, i+1, i+2#, i+3 i-3,


neigbours[neigbours >= len(height_values)] = np.nan
neigbours[neigbours < 0] = np.nan

L = generate_L(neigbours)
coL = np.tril(np.triu(np.ones((len(VMR_O3)-1,len(VMR_O3))),0),+1)

for i in range(0,len(VMR_O3)-1):
        coL[i,i] = -1

# coL[0,0] = -np.sqrt(2)
# coL[0,1]= np.sqrt(2)
# coL[-1,-1] = -np.sqrt(2)
# coL[-2,-1] = np.sqrt(2)
coL[startInd::, startInd::] = coL[startInd::, startInd::] * 0.5
TIKH = coL.T@coL
TIKH[0,0] = 2 * np.copy(TIKH[0,0])
TIKH[-1,-1] = 2 * np.copy(TIKH[-1,-1])
#L = np.copy(TIKH)

LaplU, SingL, LalpVH =np.linalg.svd(L, full_matrices = True)
LaplValL, EigVecL = np.linalg.eig(L)
SortLaplValL = np.flip(np.array(sorted(LaplValL)))
##
# 35
#L[startInd::, startInd::] = L[startInd::, startInd::] * 0.5
#L[startInd, startInd-1] = - 4/6
#L[startInd, startInd+1] = -2/6
#L[startInd, startInd] = -L[startInd, startInd-1] - L[startInd, startInd+1] #-L[startInd, startInd-2] - L[startInd, startInd+2]
#np.sum(L,axis=1)

# delHeights = height_values[1:] - height_values[0:-1]
#
#
# from scipy.interpolate import UnivariateSpline
# calcdev = UnivariateSpline( height_values, VMR_O3, k=2, s=0).derivative(n=2)
# calcdev = np.gradient(VMR_O3[:,0],height_values[:,0], edge_order = 2)
# #calcdev = scy.misc.derivative(VMR_O3[:,0],n=2)
# fig, axs = plt.subplots( figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)
#
# axs.plot( calcdev,height_values[:,0],marker = 'o' )
# axs.plot( L@VMR_O3 /delHeights[0],height_values[:,0])
#
# plt.show(block = True)

##
delHeights = height_values[1:] - height_values[0:-1]
# newL = generate_L(neigbours)
# for i in range(1,len(newL)-1):
#     newL[i, i]= 1/delHeights[i-1]**2 + 1/delHeights[i]**2
#     newL[i, i-1] = -1/delHeights[i-1]**2
#     newL[i,i+1] = -1/delHeights[i]**2
#
# newL[0, 0] = 2 / delHeights[0] ** 2
# newL[1, 0] = -1 / delHeights[0] ** 2
# newL[0, 1] = -1 / delHeights[0] ** 2
#
# newL[-1, -1] = 2 / delHeights[-1] ** 2
# newL[-2,-1] = -1 / delHeights[-1] ** 2
# newL[-1, -2] = -1 / delHeights[-1] ** 2


#L = np.copy(newL)
##
#L[1:].shape

#L[startInd+1, startInd+1] = -L[startInd+1, startInd+1-1] - L[startInd+1,startInd+1+1] -L[startInd+1, startInd+1-2] - L[startInd+1, startInd+1+2]
# L[16, 16] = 13

np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')
#cholesky decomposition of L for W1 and v1
lowC_L = scy.linalg.cholesky(L, lower = True)

##
DMat =-(np.triu(np.ones((len(L), len(L))), k=0) - np.triu(np.ones((len(L), len(L))), k=1)) + (np.triu(np.ones((len(L), len(L))), k=1) - np.triu(np.ones((len(L), len(L))), k=2))
#DMat[1,0] = 1
LaplTry = DMat.T @ DMat
print(LaplTry )

##
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

# # https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
# temperature = get_temp_values(heights)
# #temp_values = temperature[minInd:maxInd]
# temp_values = temperature[minInd:maxInd][::skipInd]

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


np.savetxt('S.txt',S, delimiter= '\t')
#load constants in si annd convert to cgs units by multiplying
h = scy.constants.h #* 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b_cgs = constants.Boltzmann #* 1e7#in J K^-1
#T = temp_values[0:-1] #in K
N_A = constants.Avogadro # in mol^-1
R = constants.gas_constant


#mol_M = 48 #g/mol for Ozone
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
scalingConst = 1#e11
#theta =(num_mole * w_cross.reshape((SpecNumLayers,1)) * Source * scalingConst )
theta = num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]

np.savetxt('num_mole.txt', [num_mole], fmt = '%.15f', delimiter= '\t')
np.savetxt('wvnmbr.txt', wvnmbr, fmt = '%.15f', delimiter= '\t')
np.savetxt('g_doub_prime.txt', g_doub_prime, fmt = '%.15f', delimiter= '\t')
np.savetxt('E.txt', E, fmt = '%.15f', delimiter= '\t')
np.savetxt('LineIntScal.txt', LineIntScal, fmt = '%.30f', delimiter= '\t')



# fig, axs = plt.subplots(tight_layout=True, figsize=set_size(PgWidthPt, fraction=fraction))
# #plt.plot(press/1013.25,heights, label = 'pressure in hPa/' + str(np.around(max(press),3)) )
# #plt.plot(Source/max(Source),height_values, label = r'Source in $\frac{W}{m^2 sr}\frac{1}{\frac{1}{cm}}$/' + str(np.around(max(Source[0]),5)) )
# plt.plot(calc_temp,heights, color = 'darkred')# label = r'Source in K/' + str(np.around(max(temperature[0]),3)) )
#
# plt.plot(temp_values,height_values, color = 'red')# label = r'Source in K/' + str(np.around(max(temperature[0]),3)) )
#
# #plt.plot(LineInt,heights[minInd:maxInd], color = 'darkred')# label = r'Source in K/' + str(np.around(max(temperature[0]),3)) )
# #axs.legend()
# axs.tick_params(axis = 'x', labelcolor="darkred")
# ax2 = axs.twiny() # ax1 and ax2 share y-axis
# line3 = ax2.plot(press,heights, color = 'blue') #, label = 'pressure in hPa/' + str(np.around(max(press),3)) )
# ax2.spines['top'].set_color('blue')
# ax2.tick_params(labelcolor="blue")
# ax2.set_xlabel('Pressure in hPa')
# axs.set_ylabel('Height in km')
# axs.set_xlabel('Temperature in K')
# #axs.set_xlabel('Line intensity in cm / molecule')
# #axs.set_title()
# plt.savefig('PandQ.png')
# plt.show()


AO3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, temp_values)
A = 2*AO3

# ##

PgWidthPt = 421/2
fraction = 1.5
plt.rcParams.update({'font.size':  10,
                     'text.usetex': True,
                     'font.family' : 'serif',
                     'font.serif'  : 'cm',
                     'text.latex.preamble': r'\usepackage{bm, amsmath}'})
U, SingS, Vh = np.linalg.svd(A)#, full_matrices=True)


fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
for i in range(0,10):
    ax1.plot(Vh[i],height_values, linewidth = 0.85)
    ax1.text(1.15*max(Vh[i]), height_values[Vh[i] == max(Vh[i])], str(i+1))

ax1.set_ylabel(r'height in km')
ax1.set_title(r'first 10 right singular vectors $\bm{v}_i$ of $\bm{A}$', fontsize=10)
plt.savefig('SingVecA.png', dpi = dpi)

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
for i in range(20,len(SingS)):
    ax1.plot(Vh[i],height_values, linewidth = 0.85)
    ax1.text(1.15*max(Vh[i]), height_values[Vh[i] == max(Vh[i])], str(i+1))

ax1.set_ylabel(r'height in km')
ax1.set_title(r'right singular vectors $\bm{v}_i$ of $\bm{A}$', fontsize=10)
plt.savefig('NullVecA.png', dpi = dpi)

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
for i in range(10,20):
    ax1.plot(Vh[i],height_values, linewidth = 0.85)
    ax1.text(1.15*max(Vh[i]), height_values[Vh[i] == max(Vh[i])], str(i+1))

ax1.set_ylabel(r'height in km')
ax1.set_title(r'last 5 right singular vectors $\bm{v}_i$ of $\bm{A}$', fontsize=10)

plt.savefig('MiddleVecA.png', dpi = dpi)
#ax1.set_xlabel(r'index')
#ax1.text(max(Vh[19]), height_values[Vh[19] == max(Vh[19])], f'20')
#ax1.text(1.05*max(Vh[0]), height_values[Vh[0] == max(Vh[0])], f'1')
#ax1.text(1.05*max(Vh[3]), height_values[Vh[3] == max(Vh[3])], f'4')
#ax1.text(1.05*max(Vh[9]), height_values[Vh[9] == max(Vh[9])], f'10')

##
np.allclose(A, U[:, :len(SingS)] @ np.diag(SingS) @ Vh[:len(SingS),:])

#np.savetxt('SingSmore.txt', SingS, fmt = '%.30f', delimiter = '\t')

ExpIncSingSNormal = np.loadtxt('ExpIncSingSNormal.txt')
ExpDecSingSNormal = np.loadtxt('ExpDecSingSNormal.txt')
SingSNormal = np.loadtxt('SingSNormal.txt')
SingSfewer = np.loadtxt('SingSfewer.txt')
SingSmore = np.loadtxt('SingSmore.txt')
fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
ax1.scatter(range(1,len(ExpIncSingSNormal)+1),ExpIncSingSNormal, label = 'case 1',s=40 , marker = 'v')
ax1.scatter(range(1,len(ExpDecSingSNormal)+1),ExpDecSingSNormal,s = 25, label = 'case 2')
ax1.scatter(range(1,len(SingSNormal)+1),SingSNormal,s = 20 ,label = 'case 3', c = RegCol, marker = 's')

#ax1.scatter(range(len(SingSfewer)),SingSfewer,marker = '.', s= 10, c='r',label = 'case 5')
ax1.scatter(range(1,len(SingSmore)+1),SingSmore,marker = 'x', s= 15, c='g',label = 'case 4')
ax1.scatter(range(1,len(SingS)+1),SingS,marker = '.', s= 10, c='k',label = 'case 5')
ax1.axhline(max(SingS)/150, linestyle = '--', color = 'k', linewidth = 0.75)
ax1.axhline(min(SingS), linestyle = '--', color = 'k', linewidth = 0.75)
ax1.text(0.8 , 1.3*max(SingS)/150, r'SNR $\approx 150$', transform=ax1.get_yaxis_transform())

ax1.text(0.01, 1.2*min(SingS), rf'SNR $\approx {(max(SingS)/min(SingS)):1.0e}$', transform=ax1.get_yaxis_transform())
ax1.set_yscale('log')
ax1.set_xlim(0.01)
ax1.set_ylabel(r'singular values of $\bm{A}$ at height')
ax1.set_xlabel(r'index')
#ax2 = ax1.twinx()
#ax2.scatter(range(len(tang_heights_lin)),tang_heights_lin, c = 'C1', marker='*')
#ax2.set_ylabel('tangent height')
#fig3.savefig('EigAExp.png'
ax1.legend()
#plt.show(block = True)
plt.savefig('SingValA.png', dpi = dpi)

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
# ax1.scatter(range(len(meas_ang)),meas_ang1,s = 40, label = 'case 1', marker = 'v')
# ax1.scatter(range(len(meas_ang)),meas_ang2,s = 25, label = 'case 2' )
# ax1.scatter(range(len(meas_ang)),meas_ang, s = 10, label = 'case 3', c = RegCol, marker = 's')
ax1.scatter(range(1,len(meas_ang1)+1),tang_heights_lin1,s = 40, label = 'case 1', marker = 'v')
ax1.scatter(range(1,len(meas_ang2)+1),tang_heights_lin2,s = 25, label = 'case 2' )
ax1.scatter(range(1,len(tang_heights_linNormal)+1),tang_heights_linNormal, s = 20, label = 'case 3', c = RegCol, marker = 's')

#ax1.scatter(range(len(tang_heights_lin3)),tang_heights_lin3,marker = '.', s= 10, c='k',label = 'case 5')
ax1.scatter(range(1,len(tang_heights_lin3)+1),tang_heights_lin3,marker = 'x', s= 15, c='g',label = 'case 4')
ax1.scatter(range(1,len(tang_heights_linChosen)+1),tang_heights_linChosen,marker = '.', s= 10, c='k',label = 'case 5')
ax1.legend()
ax1.set_ylabel(r'tangent height of $\Gamma_j$')
ax1.set_xlabel(r'index j')
ax1.set_xlim(0.01)
plt.savefig('MeasTangHeight.png', dpi = dpi)
plt.show(block = True)

print('sing Vec')
##

# from pygsvd import *
#
# Ugvsd, Vgvsd, Xgvsd, Cgvsd, Sgvsd = gsvd(A, L)

##

# ColinsSNR1 = 140#
# Ax =np.matmul(A, VMR_O3 * theta_scale_O3)
#
# y, gam0 = add_noise(Ax.reshape((SpecNumMeas,1)), ColinsSNR1)
#fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
#xDelta = np.zeros((len(SingS),1))
# lowerHeight = np.zeros(len(VMR_O3))
# upperHeight = np.zeros(len(VMR_O3))
# deltaDelHeight = np.zeros(len(VMR_O3))
# UPPerdeltaDelHeight = np.zeros(len(VMR_O3))
# impuls =np.mean(VMR_O3 * theta_scale_O3)
# for i in range(0, len(VMR_O3)):
#     xDelta = np.zeros((len(VMR_O3), 1))
#     xDelta[i] = impuls#1#0e10
#     benchmark = A @ xDelta #* VMR_O3 * theta_scale_O3)
#     for j in range(0, i):
#         xDelta = np.zeros((len(VMR_O3), 1))
#         xDelta[j] = impuls# 10e10
#         #xDelta[i] = 1#0e10
#         compare = A @ xDelta #* VMR_O3 * theta_scale_O3)
#         diff = np.sum((benchmark - compare)**2)/ len(VMR_O3)
#         if diff <= (1 / gam0) :
#             print(j)
#             print(diff)
#             deltaDelHeight[i] = abs(height_values[j] - height_values[i])
#             break
#
#
# for i in range(0, len(VMR_O3)):
#     print('i ' + str(i))
#     xDelta = np.zeros((len(VMR_O3), 1))
#     xDelta[i] = impuls  # 1#0e10
#     benchmark = A @ xDelta
#     for j in range(len(VMR_O3)-1,i,-1):
#         print('j ' + str(j))
#         if j != i:
#             xDelta = np.zeros((len(VMR_O3), 1))
#             xDelta[j] = impuls#1
#             compare = A @ xDelta #* VMR_O3 * theta_scale_O3)
#             diff = np.sum((benchmark - compare) ** 2) /len(VMR_O3)
#             #UPPerdeltaDelHeight[i] = abs(height_values[j] - height_values[i])
#             if diff <= (1 / gam0):
#                 print(diff)
#                 UPPerdeltaDelHeight[i] = abs(height_values[j] - height_values[i])
#                 break


# fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
# #print(A @ xDelta)
# #ax1.plot( A @ xDelta, tang_heights_lin)
# ax1.plot( height_values, deltaDelHeight )
# ax1.plot( height_values, UPPerdeltaDelHeight )
# ax1.set_ylabel('delta height')
# #ax1.set_xscale('log')
# plt.show(block = True)

# fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
# ax1.errorbar( height_values,height_values[:,0], yerr= [deltaDelHeight,UPPerdeltaDelHeight] )
# ax1.set_ylabel('delta height')
# plt.show(block = True)

# ##
# maxJ = 5#4
# impuls =1#np.mean(VMR_O3 * theta_scale_O3)
# relDiff = np.zeros((len(VMR_O3), len(VMR_O3)))
# relDiff = np.zeros((len(VMR_O3),maxJ))
# relDiffLow = np.zeros((len(VMR_O3),maxJ))
# HDiff = np.zeros((len(VMR_O3),maxJ))
# HDiffLow = np.zeros((len(VMR_O3),maxJ))
# for j in range(1, maxJ):
#     for i in range(0, len(VMR_O3)-j):
#         xDelta = np.zeros((len(VMR_O3), 1))
#         xDelta[i] = impuls#1#0e10
#         benchmark = A @ xDelta
#         xDelta = np.zeros((len(VMR_O3), 1))
#         xDelta[i+j] = impuls#1#0e10
#         compare = A @ xDelta
#         relDiff[i,j] = np.linalg.norm(benchmark - compare) #/ np.linalg.norm(compare) #*100
#         HDiff[i, j] = abs(height_values[i] - height_values[i+j])
#     for i in range(j, len(VMR_O3)):
#         xDelta = np.zeros((len(VMR_O3), 1))
#         xDelta[i] = impuls  # 1#0e10
#         benchmark = A @ xDelta
#         xDelta = np.zeros((len(VMR_O3), 1))
#         xDelta[i - j] = impuls  # 1#0e10
#         compare = A @ xDelta
#         relDiffLow[i, j] = np.linalg.norm(benchmark - compare) #/ np.linalg.norm(compare)#*100
#         HDiffLow[i, j] = abs(height_values[i] - height_values[i-j])
#     # for j in range(0, len(VMR_O3)):
#     #     xDelta = np.zeros((len(VMR_O3), 1))
#     #     xDelta[j] = impuls#1#0e10
#     #     compare = A @ xDelta
#     #     relDiff[i,j] = np.linalg.norm(benchmark -compare)/np.linalg.norm(benchmark)
# relDiff[relDiff == 0] = np.nan
# relDiffLow[relDiffLow == 0] = np.nan
# fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
# for j in range(1, maxJ):
#     ax1.plot(height_values[:len(VMR_O3)-j]/len(VMR_O3), relDiff[:len(VMR_O3)-j,j], label = str(j))
#     ax1.plot(height_values[j:], relDiffLow[j:,j]/len(VMR_O3), label = str(j))
#     #ax1.errorbar(height_values[:,0] ,HDiffLow[:,j],[relDiffLow[:,j],relDiff[:,j]], label = str(j))
# # for i in range(0, len(VMR_O3)):
# #     ax1.plot(abs(height_values- height_values[i])[:i], relDiff[i,:i])
# #     ax1.plot(abs(height_values - height_values[i])[i:], relDiff[i, i:])
# ax1.axhline(np.sqrt(1/gam0))
# ax1.legend()
# ax1.set_yscale('log')
# ax1.set_ylabel('delta height')
# plt.show(block=True)
##
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
SNR = 60#35000
SNR = 150
#y, gam0 = add_noise(Ax.reshape((SpecNumMeas,1)), SNR)
##
#y = Ax.reshape((SpecNumMeas,1)) + np.random.normal(0,0.00025,size=(SpecNumMeas,1))

y, gam0  = add_noise_Blokk(Ax,SNR)
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)
ax1.scatter(y, tang_heights_lin, color = 'r')
ax1.axhline(height_values[startInd])
ax1.set_xscale('log')
plt.show(block = True)

xrms = 10 * np.log(np.sqrt(np.mean(np.random.normal(0,gam0,size=10000)**2)))

##

signal_power = np.sqrt(np.mean(np.abs(Ax) ** 2))
noise = np.random.normal(0, np.sqrt(1 / gam0), size =y.shape)
noise_power = np.sqrt(np.mean(np.abs(noise) ** 2))
snr = signal_power / noise_power

rms_noise = 10*np.log(np.sqrt(np.mean(np.random.normal(0, np.sqrt(1 / gam0), size =10000)**2)))

ColinsSNR = np.max(Ax)/rms_noise

ColinsSNR1 = np.max(y)/np.sqrt(1 / gam0)

nonLinA = calcNonLin(tang_heights_lin, A_lin_dx, height_values, pressure_values, ind, temp_values, VMR_O3, AscalConstKmToCm, wvnmbr, S, E,g_doub_prime)
OrgData = np.matmul(AO3 * nonLinA,VMR_O3 * theta_scale_O3)
DatErr = np.linalg.norm( OrgData -  Ax) / np.linalg.norm(OrgData) * 100
print('DataErr '+ str(DatErr))

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin, label = 'linear Data')
ax1.plot(OrgData , tang_heights_lin, label = 'nonlinear Data')
ax1.scatter(y, tang_heights_lin, color = 'r')
ax1.set_xscale('log')
ax1.legend()
plt.show()


##
noise = np.random.normal(0, np.sqrt(1 / gam0), size = OrgData.shape)
nonLinY = (OrgData + noise).reshape((SpecNumMeas,1))
y = (Ax + noise).reshape((SpecNumMeas,1))
np.savetxt('nonLinDataY.txt',nonLinY, fmt = '%.15f', delimiter= '\t')
np.savetxt('dataY.txt',y, fmt = '%.15f', delimiter= '\t')
np.savetxt('AMat.txt',A, fmt = '%.30f', delimiter= '\t')
np.savetxt('ALinMat.txt',A_lin, fmt = '%.15f', delimiter= '\t')
np.savetxt('nonLinA.txt',nonLinA, fmt = '%.15f', delimiter= '\t')
np.savetxt('gamma0.txt',[gam0], fmt = '%.15f', delimiter= '\t')


## some modifiaction
# to find correaltion bewteen tmep and others

y = np.copy(nonLinY)

#new_temp_values = np.mean(temp_values) * np.ones((SpecNumLayers,1))
#AO3, theta_scale_O3 = composeAforO3(A_lin, new_temp_values, pressure_values, ind, new_temp_values)
#A = 2*AO3
#ATA = np.matmul(A.T,A)
#Ax =np.matmul(A, VMR_O3 * theta_scale_O3)

APress, press_scale = composeAforPress(2*A_lin, temp_values, VMR_O3, ind)
np.savetxt('AP.txt', APress, fmt = '%.15f', delimiter= '\t')
ATemp, temp_scale = composeAforTemp(2*A_lin, pressure_values, VMR_O3, ind, temp_values)
np.savetxt('AT.txt', ATemp, fmt = '%.15f', delimiter= '\t')
APressTemp, press_temp_scale = composeAforTempPress(2*A_lin, VMR_O3, ind, temp_values)
np.savetxt('APT.txt', APressTemp, fmt = '%.15f', delimiter= '\t')
AxPT = np.matmul(APressTemp, pressure_values/temp_values[:,0])
AxT = np.matmul(ATemp, 1/temp_values[:,0])
AxP = np.matmul(APress, pressure_values)
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin, color = 'g', label = 'noise free')
ax1.plot(AxPT, tang_heights_lin, color = 'blue')
ax1.plot(AxP, tang_heights_lin,  color = 'orange')
ax1.plot(AxT, tang_heights_lin, color = 'm')
ax1.scatter(y, tang_heights_lin, color = 'r')
ax1.plot(y, tang_heights_lin, color = 'r', label = 'noisy')
ax1.scatter(nonLinY, tang_heights_lin, color = 'k')
ax1.plot(nonLinY, tang_heights_lin, color = 'k')
plt.legend()
plt.show()

#y = np.loadtxt('dataY.txt').reshape((SpecNumMeas,1))
ATy = np.matmul(A.T, y)

fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))

ax1.scatter(y, tang_heights_lin)
ax1.plot(y, tang_heights_lin, label = 'noisy')
ax1.plot(Ax, tang_heights_lin, label = 'noise free')
plt.legend()
plt.show()
#print(1/np.var(y))


#np.savetxt('dataY.txt', y, header = 'Data y including noise', fmt = '%.15f')
np.savetxt('ForWardMatrix.txt', A, header = 'Forward Matrix A', fmt = '%.15f', delimiter= '\t')
np.savetxt('height_values.txt', height_values, fmt = '%.30f', delimiter= '\t')
np.savetxt('tan_height_values.txt', tang_heights_lin, fmt = '%.15f', delimiter= '\t')

np.savetxt('pressure_values.txt', pressure_values, fmt = '%.30f', delimiter= '\t')
np.savetxt('temp_values.txt', temp_values, fmt = '%.15f', delimiter= '\t')

np.savetxt('VMR_O3.txt', VMR_O3, fmt = '%.30f', delimiter= '\t')
np.savetxt('theta_scale_O3.txt', [theta_scale_O3], fmt = '%.15f')

np.savetxt('calc_press.txt', calc_press[:maxInd], fmt = '%.30f', delimiter= '\t')
np.savetxt('actual_heights.txt', actual_heights[:maxInd], fmt = '%.30f', delimiter= '\t')

## change data length

# MaxAng = [np.arcsin((55+ R_Earth) / (R_Earth + ObsHeight))]
# MinAng = np.arcsin((height_values[0] + R_Earth) / (R_Earth + ObsHeight))
# pointAcc = 0.00045
# meas_ang = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))
# SpecNumMeas = len(meas_ang)
# m = SpecNumMeas
# A_lin_dx, tang_heights_lin, extraHeight = gen_forward_map(meas_ang,height_values,ObsHeight,R_Earth)
# A_lin = gen_sing_map(A_lin_dx, tang_heights_lin, height_values)
# AO3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind, temp_values)
# A = 2*AO3
# y = np.copy(y[:m])


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



##
""" finally calc f and g with a linear solver adn certain lambdas
 using the gmres"""

lam= np.logspace(-5,15,500)
f_func = np.zeros(len(lam))
g_func = np.zeros(len(lam))



for j in range(len(lam)):

    B = (ATA + lam[j] * L)

    #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], rtol=tol, restart=25)
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    # check if L L.H = B
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

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
''' check taylor series in f(lambda) and g(lambda)
around lam0 from gmres = '''

#taylor series arounf lam_0
lam0 = 1.5*minimum[1]
B = (ATA + lam0 * L)

LowTri = np.linalg.cholesky(B)
UpTri = LowTri.T
# check if L L.H = B
B_inv_A_trans_y0 = lu_solve(LowTri, UpTri,  ATy[0::, 0])



np.savetxt('B_inv_A_trans_y0.txt', B_inv_A_trans_y, fmt = '%.15f', delimiter= '\t')


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
f_0_4 = -1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y0)
f_0_5 = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_5) ,B_inv_A_trans_y0)
f_0_6 = -1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_6) ,B_inv_A_trans_y0)



g_0_1 = np.trace(B_inv_L)
g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
g_0_3 = 1 /6 * np.trace(B_inv_L_3)
g_0_4 = 0#-1 /24 * np.trace(B_inv_L_4)
g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
g_0_6 = 0#-1 /720 * np.trace(B_inv_L_6)

f_0 = f(ATy, y, B_inv_A_trans_y0)
g_0 = g(A, L,minimum[1] )
delG = (np.log(g(A, L, 1.2e4)) - np.log(g_0))/ (np.log(1.2e4) - np.log(minimum[1]))


##

'''do the sampling'''


number_samples = 20000
burnIn = 100


#wLam = 2e2#5.5e2
#wgam = 1e-5
#wdelt = 1e-1

alphaG = 1
alphaD = 1
rate = f_0 / 2 + betaG + betaD * lam0
# draw gamma with a gibs step
shape = SpecNumMeas/2 + alphaD + alphaG

#f_new = f_0
#g_old = g(A, L,  lambdas[0])
g_0 = g(A, L, lam0)
delG = (np.log(g(A, L, 1e4)) - np.log(g_0))/ (np.log(1e4) - np.log(lam0))


def MHwG(number_samples, burnIn, lam0, gamma0, f_0):
    wLam = lam0 * 0.8#8e3#7e1

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
        delta_lam_t = lambdas[t] - lam0
        delta_lam_p = lam_p - lam0
        delta_f = f_0_1 * delta_lam + f_0_2 * (delta_lam_p**2 - delta_lam_t**2) + f_0_3 *(delta_lam_p**3 - delta_lam_t**3) #+ f_0_4 * delta_lam**4 + f_0_5 * delta_lam**5
        #delta_g = g_0_1 * delta_lam + g_0_2 * (delta_lam_p**2 - delta_lam_t**2) + g_0_3 * (delta_lam_p**3 - delta_lam_t**3) #+ g_0_4 * delta_lam**4 + g_0_5 * delta_lam**5
        Glam_p = (np.log(lam_p) - np.log(lam0)) * delG + np.log(g_0)
        Gcurr = (np.log(lambdas[t]) - np.log(lam0)) * delG + np.log(g_0)
        delta_g = np.exp(Glam_p) - np.exp(Gcurr)
        log_MH_ratio = ((SpecNumLayers)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

        #accept or rejeict new lam_p
        u = uniform()

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
            delta_f = f_0_1 * delta_lam_p + f_0_2 * delta_lam_p ** 2 + f_0_3 * delta_lam_p ** 3
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
lambdas ,gammas, k = MHwG(number_samples, burnIn, lam0, gamma0, f_0)
elapsed = time.time() - startTime
print('MTC Done in ' + str(elapsed) + ' s')



print('acceptance ratio: ' + str(k/(number_samples+burnIn)))
deltas = lambdas * gammas
np.savetxt('samples.txt', np.vstack((gammas[burnIn::], deltas[burnIn::], lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')

gam_mean, gam_del, gam_tint, gam_d_tint= tauint([[gammas]],0)
lam_mean, lam_del, lam_tint, lam_d_tint = tauint([[lambdas]],0)

##

BinHist = 30#n_bins
lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density= True)
gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)

trace = [MinLogMargPost(np.array([lambdas[burnIn+ i],gammas[burnIn+ i]])) for i in range(number_samples)]


fig, axs = plt.subplots(3, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction) )#, dpi = dpi)

axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(gamBinEdges)[0])#10)

axs[0].axvline(gam0)
axs[0].set_xlabel(r'the noise precision $\gamma$')


axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)

axs[2].plot( range(number_samples), trace, color = 'k')
axs[2].set_ylabel(r'$\pi(\bm{\theta}|\bm{y})$')
axs[2].set_xlabel('number of samples')
plt.savefig('HistoPlot.png')
plt.show()
##
''' check taylor series in f(lambda) and g(lambda)
around lam0 from gmres = '''

#taylor series arounf lam_0
B = (ATA + lam0 * L)

LowTri = np.linalg.cholesky(B)
UpTri = LowTri.T
# check if L L.H = B
B_inv_A_trans_y0 = lu_solve(LowTri, UpTri,  ATy[0::, 0])



np.savetxt('B_inv_A_trans_y0.txt', B_inv_A_trans_y, fmt = '%.15f', delimiter= '\t')


B_inv_L = np.zeros(np.shape(B))

for i in range(len(B)):
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    B_inv_L[:, i] = lu_solve(LowTri, UpTri,  L[:, i])

#relative_tol_L = rtol
#CheckB_inv_L = np.matmul(B, B_inv_L)
#print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)

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

fCol = [0, 144/255, 178/255]
gCol = [230/255, 159/255, 0]
#gCol = [240/255, 228/255, 66/255]
#gCol = [86/255, 180/255, 233/255]
gmresCol = [204/255, 121/255, 167/255]

#lambBinEdges = np.linspace(100, 1e4, 50 )
delta_lam = lambBinEdges - lam0
#taylorG = g_tayl(delta_lam,g(A, L, minimum[1]), g_0_1, g_0_2, g_0_3, g_0_4, g_0_5, g_0_6)
taylorF = f_tayl(delta_lam, f_0, f_0_1, f_0_2, f_0_3,f_0_4, f_0_5, f_0_6)
taylorF = f_tayl(delta_lam, f_0, f_0_1, f_0_2,f_0_3,0, 0, 0)
g_0 = g(A, L,lam0 )
delG = (np.log(g(A, L, 1e4)) - np.log(g_0))/ (np.log(1e4) - np.log(lam0))
GApprox = (np.log(lambBinEdges) - np.log(lam0)) * delG  + np.log(g_0)
taylorG = np.exp(GApprox)


fig,axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)#, dpi = dpi)

axs.plot(lam,f_func, color = fCol, zorder = 2, linestyle=  'dotted')
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

axins.axvline( lam0, color = gmresCol, label = r'$\pi(\lambda_0|\bm{y}, \gamma)$')

axins.plot(lambBinEdges,taylorF , color = 'k', linewidth = 1, zorder = 1, label = 'Taylor series' )
axs.plot(lambBinEdges,taylorF , color = 'k', linewidth = 1, zorder = 2, label = 'Taylor series' )


axins.set_ylim(0.95 * taylorF[0],2 * taylorF[-1])
axins.set_xlabel('$\lambda$')
axins.set_yscale('log')
axins.set_xscale('log')



axins.tick_params(axis='y', which='both',  left=False, labelleft=False)

axins.tick_params(axis='y', which='both', length=0)

axin2 = axins.twinx()
axin2.spines['top'].set_visible(False)
axin2.spines['right'].set_visible(False)
axin2.spines['bottom'].set_visible(False)
axin2.spines['left'].set_visible(False)


axin2.tick_params(axis = 'y', which = 'both',labelright=False, right=False)
axin2.tick_params(axis='y', which='both', length=0)


# #axin2.set_xticks([np.mean(lambdas) -np.sqrt(np.var(lambdas)) , np.mean(lambdas), np.mean(lambdas) + np.sqrt(np.var(lambdas)) ] )
axin2.plot(lam,g_func, color = gCol, zorder=3, linestyle=  'dashed', linewidth = 3,label = '$g(\lambda)$')

axin2.plot(lambBinEdges, taylorG, color = 'k', linewidth = 1, zorder = 2 )

ax2.plot(lambBinEdges, taylorG , color = 'k', linewidth = 1, zorder = 1)
ax2.axvline( minimum[1], color = gmresCol)

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


axs.legend(np.append(lines2,lines),np.append(lab2,lab0), loc = 'lower right')

axins.set_xlim(min(lambBinEdges),max(lambBinEdges))
fig.savefig('f_and_g_paper.png', bbox_inches='tight')
plt.show()

# print max rel F taylor F error
f_Checkfunc = np.zeros(len(lambBinEdges))
for j in range(len(lambBinEdges)):

    B = (ATA + lambBinEdges[j] * L)

    #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], rtol=tol, restart=25)
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    # check if L L.H = B
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

    f_Checkfunc[j] = f(ATy, y, B_inv_A_trans_y)

relFErr = max(abs(f_Checkfunc - taylorF)/abs(f_Checkfunc))
ErrLam = lambBinEdges[abs(f_Checkfunc - taylorF)/abs(f_Checkfunc) == relFErr][0]
print(f'relative F error {relFErr *100} at {ErrLam}')

###draw paramter samples
paraSamp = 200#n_bins
Results = np.zeros((paraSamp,len(theta)))
NormRes = np.zeros(paraSamp)
xTLxRes = np.zeros(paraSamp)
SetGammas = gammas[np.random.randint(low=burnIn, high=len(gammas), size=paraSamp)]
SetDeltas  = deltas[np.random.randint(low=burnIn, high=len(deltas), size=paraSamp)]

startTimeX = time.time()
for p in range(paraSamp):
    # SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
    SetGamma = SetGammas[p] #minimum[0]
    SetDelta  = SetDeltas[p] #minimum[1]
    #W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
    W = np.random.normal(loc=0.0, scale=1.0, size=len(A))#.reshape((len(A),1))
    v_1 = np.sqrt(SetGamma) *  A.T @ W
    W2 = np.random.multivariate_normal(np.zeros(len(L)), L)
    # W2 = lowC_L @ np.random.multivariate_normal(np.zeros(len(L)), np.eye(len(L)) )
    # W2 = lowC_L @ np.random.normal(loc=0.0, scale=1.0, size=len(L))#.reshape((len(L),1))
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

    # B_inv_A_trans_y, exitCode = gmres(SetB, RandX, x0=B_inv_A_trans_y0, rtol=tol)
    #
    # # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    # if exitCode != 0:
    #     print(exitCode)
    LowTri = np.linalg.cholesky(SetB)
    UpTri = LowTri.T
    B_inv_A_trans_y = lu_solve(LowTri, UpTri, RandX)

    #CheckB_inv = np.matmul(SetB, SetB_inv)
    #print(np.linalg.norm(np.eye(len(SetB)) - CheckB_inv) / np.linalg.norm(np.eye(len(SetB))) < tol)

    Results[p, :] = B_inv_A_trans_y

    NormRes[p] = np.linalg.norm( np.matmul(A,B_inv_A_trans_y) - y[0::,0])
    xTLxRes[p] = np.sqrt(np.matmul(np.matmul(B_inv_A_trans_y.T, L), B_inv_A_trans_y))


##
def postMeanAndVar(margPDF, Grid, ATy, ATA, L, Var):
    gridLen = len(margPDF[0])
    MargResults = np.zeros((gridLen,len(L)))
    gamInt = np.zeros(gridLen)
    VarB = np.zeros((gridLen, len(L), len(L)))
    B_inv = np.zeros((gridLen, len(L), len(L)))
    if len(Grid[0]) != len(margPDF[0]):
        print('Grid not the same lenght as Marg PDF')

    IDiag = np.eye(len(L))
    for p in range(gridLen):
        SetGamma = Grid[0, p]
        SetLambda = Grid[1,p]

        SetB = ATA + SetLambda * L

        LowTri = np.linalg.cholesky(SetB)
        UpTri = LowTri.T
        B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

        MargResults[p, :] = B_inv_A_trans_y * margPDF[1,p]

        if Var == True:
            LowTri = np.linalg.cholesky(SetB)
            UpTri = LowTri.T
            for i in range(len(SetB)):
                B_inv[p, :, i] = lu_solve(LowTri, UpTri,IDiag[:, i])
            VarB[p] = B_inv[p] *  margPDF[1,p]
            gamInt[p] = 1/SetGamma *  margPDF[0,p]

    postMean = np.sum(MargResults,0)
    postVar = np.sum(gamInt) * np.sum(VarB,0)
    return postMean, postVar


##

BinHistStart = 3
print(BinHistStart)
oldpostMean = 0
for PostMeanBinHist in range(BinHistStart+1,100):

    lambHist, lambBinEdges = np.histogram(lambdas[burnIn:], bins= PostMeanBinHist, density =True)
    gamHist, gamBinEdges = np.histogram(gammas[burnIn:], bins= PostMeanBinHist, density =True)
    margPDF = np.array([gamHist/np.sum(gamHist) , lambHist/np.sum(lambHist)])
    Grid = np.array([ gamBinEdges[:-1] + (gamBinEdges[1:] - gamBinEdges[:-1])/2, lambBinEdges[:-1] + (lambBinEdges[1:] - lambBinEdges[:-1])/2])

    startTime = time.time()
    newPostMean, postVar = postMeanAndVar(margPDF, Grid, ATy, ATA, L, True)
    MargTime = time.time() - startTime

    newRelErr = np.linalg.norm(oldpostMean - newPostMean) / np.linalg.norm(newPostMean) * 100
    print(newRelErr)
    if  0.5 > newRelErr:
        print(f'break at {PostMeanBinHist}')
        break
    oldpostMean = np.copy(newPostMean)
    oldRelErr = np.copy(newRelErr)


MargX =  newPostMean / theta_scale_O3
MargVar = postVar / theta_scale_O3**2

NormMargRes = np.linalg.norm(np.matmul(A, newPostMean) - y[0::, 0])
xTLxMargRes = np.sqrt(np.matmul(np.matmul(newPostMean.T, L), newPostMean))


print('Post Mean in ' + str(MargTime) + ' s')

print('Post Mean has ' + str(len(MargX[MargX <0]) ) + ' entries smaller than zeror')

##
"Fitting prob distr to hyperparameter histogram"

def skew_norm_pdf(x,mean=0,w=1,skewP=0, scale = 0.1):
    # adapated from:
    # http://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy
    t = (x-mean) / w
    return 2.0 * w * scy.stats.norm.pdf(t) * scy.stats.norm.cdf(skewP*t) * scale




print('bla')

##
# '''
# L-curve refularoization
# '''

lamLCurve = np.logspace(1,7,200)
lamLCurve = np.logspace(0,7,200)
#lamLCurve = np.linspace(1e-15,1e3,200)

NormLCurve = np.zeros(len(lamLCurve))
xTLxCurve = np.zeros(len(lamLCurve))
xTLxCurve2 = np.zeros(len(lamLCurve))
for i in range(len(lamLCurve)):
    B = (ATA + lamLCurve[i] * L)

    #x, exitCode = gmres(B, ATy[0::, 0], rtol=tol)
    LowTri = np.linalg.cholesky(B)
    UpTri = LowTri.T
    x = lu_solve(LowTri, UpTri, ATy[0::, 0])
    # if exitCode != 0:
    #     print(exitCode)
    #     NormLCurve[i] = np.nan
    #     xTLxCurve[i] = np.nan
    #
    # else:
    NormLCurve[i] = np.linalg.norm( np.matmul(A,x) - y[0::,0])
    xTLxCurve[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))


startTime  = time.time()
lamLCurveZoom = np.logspace(0,7,200)
lamLCurveZoom = np.copy(lamLCurve)
#lamLCurveZoom = np.logspace(-5,15,200)
NormLCurveZoom = np.zeros(len(lamLCurveZoom))
xTLxCurveZoom = np.zeros(len(lamLCurveZoom))
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
#axs.scatter(LNormOpt ,xTLxOpt, zorder = 10, color = 'red', label = 'Opt. Tikh. regularization ')
#axs.scatter(opt_norm ,opt_regNorm, zorder = 10, color = 'red')
axs.scatter(NormRes, xTLxRes, color = ResCol, s = 2, marker = "+",label = r'posterior samples ')# ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')

#axs.scatter(NewNormRes, NewxTLxRes, color = 'red', label = 'MTC RTO method')#, marker = "." ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')

#axs.scatter(SampleNorm, SamplexTLx, color = 'green', marker = 's', s= 100)
axs.scatter(NormMargRes, xTLxMargRes, color = MeanCol, marker = '.', s= 50, label = 'posterior mean',zorder=2)
#E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[\mathbf{x}_{\lambda}]$
#axs.axvline(x = knee_point)
axs.scatter(knee_point, kneedle.knee_y, color = regCol, marker = 'v',label = 'max. curvature', s= 50,zorder=1)

# #zoom in
# x1, x2, y1, y2 = NormLCurveZoom[0], NormLCurveZoom[-31], xTLxCurveZoom[0], xTLxCurveZoom[-1] # specify the limits
# axins = axs.inset_axes([0.1,0.05,0.55,0.45])
# axins.scatter(NormRes, xTLxRes, color = ResCol, label = r'posterior samples ',marker = '+')#,$\mathbf{x} \sim \pi (\mathbf{x}| \mathbf{y}, \mathbf{\theta})$ s = 15)
# axins.scatter(NormLCurve,xTLxCurve, color =  DatCol,marker = 's', s= 10,zorder=0)
# axins.scatter(NormMargRes, xTLxMargRes, color = MeanCol, marker = '.', s= 100,zorder=2)
# axins.scatter(knee_point, kneedle.knee_y, color = RegCol, marker = 'v', s = 120,zorder=1)
# axins.set_xlim(x1-0.01, x2-1) # apply the x-limits
# #axins.set_ylim(y2,y1)
# axins.set_ylim(y2,max(xTLxRes)+0.001) # apply the y-limits (negative gradient)
# axins.tick_params(axis = 'x', which = 'both', labelbottom=False, bottom = False)
# axins.tick_params(axis = 'y', which = 'both', labelleft=False, left = False)
# axins.set_xscale('log')
# axins.set_yscale('log')
# handles2, labels2 = axins.get_legend_handles_labels()
# axs.indicate_inset_zoom(axins, edgecolor="none")
# mark_inset(axs, axins, loc1=1, loc2=3, fc="none", ec="0.5")

axs.set_xscale('log')
axs.set_yscale('log')
axs.set_ylabel(r'$ \sqrt{\bm{x}^T \bm{L}\bm{x}}$', style='italic')
axs.set_xlabel(r'$|| \bm{Ax} - \bm{y}||$')
#axs.set_title('L-curve for m=' + str(SpecNumMeas))


handles, labels = axs.get_legend_handles_labels()

axs.legend(handles = [handles[0],handles[1],handles[2]],loc = 'upper right',  frameon =True)
plt.savefig('LCurve.png')
#plt.savefig('LCurve.png')
#tikzplotlib.save("LCurve.tex")
plt.show(block = True)


#tikzplotlib_fix_ncols(fig)
#tikzplotlib.save("LCurve.pgf")
print('bla')

np.savetxt('RegSol.txt',x_opt /(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst), fmt = '%.15f', delimiter= '\t')
##


## make scatter plot for results

def fitFunc(x,loc, a, scale):
    y = (x - loc) / scale
    return scy.stats.skewnorm.pdf(y, a) / scale


BinHist = 30#n_bins
lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density= True)
#paramsSkew, covs = scy.optimize.curve_fit(skew_norm_pdf,lambBinEdges[1::], lambHist/ np.sum(lambHist), p0 = [np.mean(lambBinEdges[1::]),np.sqrt(np.var(lambdas)),0.01, 1] )#np.mean(new_lamb)+1e3
paramsSkew, covs = scy.optimize.curve_fit(fitFunc,lambBinEdges[1:], lambHist, p0 = [np.mean(lambdas), 1, np.sqrt(np.var(lambdas))], bounds=(0, np.inf))


fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction), gridspec_kw={'height_ratios': [3, 1]} )#, dpi = dpi)

axs[0].scatter(gammas[burnIn:],deltas[burnIn:], marker = '.', color = MTCCol)
#axs[0].axvline(gam0)
axs[0].set_xlabel(r'the noise precision $\gamma$')
axs[0].set_ylabel(r'the smoothnes parameter $\delta$')
#axs[1].hist(new_lamb,bins=BinHist, color = MTCCol, zorder = 0, density = True)#10)
axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(lambBinEdges)[0])#10)

#axs[1].plot(lambBinEdges[1::],  skew_norm_pdf(lambBinEdges[1::], *paramsSkew )/np.sum(skew_norm_pdf(lambBinEdges[1::], *paramsSkew )), zorder = 1, color =  gmresCol)#"#009E73")
axs[1].plot(lambBinEdges[1::], fitFunc(lambBinEdges[1::], *paramsSkew )/np.sum(fitFunc(lambBinEdges[1::], *paramsSkew )), zorder = 1, color = 'limegreen')#"#009E73")

axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)
# axs[1].axvline( lam_opt, color = RegCol,linewidth=2)
# axs[1].axvline(minimum[1], zorder = 1,color = gmresCol)
# xlabels = [item.get_text() for item in axs[1].get_xticklabels()]
# xticks = axs[1].get_xticks()
# xlabels = np.append(xlabels[:-1], [r'$\lambda_R$', r'$\lambda_0$'])
# xticks = np.append(xticks[:-1],[lam_opt, minimum[1]])
# axs[1].set_xticks( xticks, xlabels)
# axs[1].set_xlim(0)
plt.savefig('ScatterplusHisto.png')
plt.show()


##

gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)
fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction) )#, dpi = dpi)

axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(gamBinEdges)[0])#10)


axs[0].set_xlabel(r'the noise precision $\gamma$')


axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)
plt.savefig('HistoPlot.png')
plt.show()

##

gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)
deltHist, deltBinEdges = np.histogram(deltas, bins= BinHist, density= True)
fig, axs = plt.subplots(3, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction) )#, dpi = dpi)

axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(gamBinEdges)[0])#10)
axs[0].set_xlabel(r'the noise precision $\gamma$')
axT = axs[0].twinx()
gamX = gamBinEdges[1::]
normConst = np.sum(np.exp(-betaG * gamX ))
axT.plot(gamX,np.exp(-betaG *  gamX )/normConst )

axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_xlabel(r'$\lambda =\delta / \gamma$, the regularization parameter')
axT = axs[1].twinx()
lambX = lambBinEdges[1::]
normConst = 1#np.sum(np.exp(-1e-10 *  lambX * gamX ) * gamX )
#axT.plot(lambX, np.exp(-betaG *  lambX * gamX )/ normConst )

axs[2].bar(deltBinEdges[1::],deltHist*np.diff(deltBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(deltBinEdges)[0])#10)
axs[2].set_xlabel(r'$\delta$, the smoothness parameter')
axT = axs[2].twinx()
delX = deltBinEdges[1::]
normConst = np.sum(np.exp( -betaD * delX ))
axT.plot(delX,np.round(np.exp( -betaD *  delX ))/normConst )

plt.savefig('AllHistoPlot.png')
plt.show()

###
plt.close('all')

TrueCol = [50/255,220/255, 0/255]#'#02ab2e'

XOPT = x_opt /(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
postCol = 'C1'
FirstSamp = 100#len(y)
Sampls = np.random.multivariate_normal(MargX, MargVar,size=FirstSamp)
rejI = 0
totI = 0

for i in range(FirstSamp):
    totI += 1
    while any(Sampls[i] < 0):
        rejI += 1
        Sampls[i] = np.random.multivariate_normal(MargX, MargVar)

testTruncMean = np.mean(Sampls, axis = 0)
testTruncVar = np.var(Sampls, axis = 0)

fig3, ax2 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))
 # ax1 and ax2 share y-axis
line3 = ax2.scatter(y, tang_heights_lin, label = r'data $\bm{y}$', zorder = 0, marker = '*', color =DatCol )#,linewidth = 5

ax1 = ax2.twiny()
#ax1.scatter(VMR_O3,height_values,marker = 'o', facecolor = 'None', color = "#009E73", label = 'true profile', zorder=1, s =12)#,linewidth = 5)
ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 1.5, markersize =7)
ax1.axhline(height_values[startInd])
# edgecolor = [0, 158/255, 115/255]
#line1 = ax1.plot(VMR_O3,height_values, color = [0, 158/255, 115/255], linewidth = 10, zorder=0)
for n in range(0,paraSamp,20):
    Sol = Results[n, :] / theta_scale_O3

    ax1.plot(Sol,height_values[:,0],marker= '+',color = ResCol,label = r'$\bm{x} \sim \pi(\bm{x}|\bm{y}, \bm{\theta})$', zorder = 1, linewidth = 0.5, markersize = 5)
    with open('Samp' + str(n) +'.txt', 'w') as f:
        for k in range(0, len(Sol)):
            f.write('(' + str(Sol[k]) + ' , ' + str(height_values[k]) + ')')
            f.write('\n')

line3 = ax1.errorbar(testTruncMean,height_values[:,0], xerr = np.sqrt(testTruncVar), markeredgecolor =postCol, color = postCol ,zorder=3, marker = '.', markersize =3, linewidth =1, capsize = 3)
# ax1.plot(Sol, height_values, marker='+', color=ResCol, label='posterior samples ', zorder=4, linewidth=0.5,
# markersize=2, linestyle = 'none')
#$\mathbf{x} \sim \pi(\mathbf{x} |\mathbf{y}, \mathbf{\theta} ) $' , markerfacecolor = 'none'
ax1.plot(XOPT, height_values[:,0], markerfacecolor = 'none', markeredgecolor = RegCol, color = RegCol ,marker='v', zorder=1, label=r'$\bm{x}_{\lambda}$', markersize =8, linewidth = 2 )# color="#D55E00"
#line2 = ax1.errorbar(x,height_values,capsize=5, yerr = np.zeros(len(height_values)) ,color = MTCCol,zorder=5,markersize = 5, fmt = 'o',label = r'$\mathbf{x} \sim \pi(\mathbf{x} |\mathbf{y}, \mathbf{\theta} ) $')#, label = 'MC estimate')

line3 = ax1.plot(MargX,height_values[:,0], markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.',  label = r'$\text{E}_{\bm{x},\bm{\theta}|\bm{y}} [\bm{x}]$', markersize =3, linewidth =1)#, markerfacecolor = 'none'
line3 = ax1.errorbar(MargX,height_values,  xerr = np.sqrt(np.diag(MargVar)), markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.', label = 'posterior mean ', markersize =3, linewidth =1, capsize = 3)#, markerfacecolor = 'none'

#E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[h(\mathbf{x})]$
# markersize = 6
#line4 = ax1.errorbar(x, height_values,capsize=5, xerr = xerr,color = MTCCol, fmt = 'o', markersize = 5,zorder=5)#, label = 'MC estimate')
#line5 = ax1.errorbar(MargX,height_values, color = MargCol, markeredgecolor= MargCol, capsize=5,  markersize = 6, zorder=3, fmt = 's')
#xerr =MargXErr/2,yerr = np.zeros(len(height_values))



#line5 = ax1.plot(x_opt/(num_mole * S[ind,0] * f_broad * 1e-4 * scalingConst),height_values, color = 'crimson', linewidth = 7, label = 'reg. sol.', zorder=1)
ax1.set_ylim([heights[minInd-1], heights[maxInd-1]])
ax1.set_xlabel(r'ozone volume mixing ratio ')
#multicolor_ylabel(ax1,('(Tangent)','Height in km'),('k', dataCol),axis='y')
ax2.set_ylabel('(tangent) height in km')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Handles = [handles[0], handles[1], handles[2]]
# Labels =  [labels[0], labels[1], labels[2]]
# LegendVertical(ax1, Handles, Labels, 90, XPad=-45, YPad=12)

legend = ax1.legend(handles = [handles[-3], handles2[0], handles[0],handles[-2],handles[-1]])# loc='lower right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

#plt.ylabel('Height in km')
ax1.set_ylim([height_values[0], height_values[-1]])
#ax2.set_xlim([min(y),max(y)])
#ax1.set_xlim([min(x)-max(xerr)/2,max(x)+max(xerr)/2]) Ozone


ax2.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
ax2.tick_params(colors = DatCol, axis = 'x')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)
#ax2.spines['top'].set_color(pyTCol)
fig3.savefig('FirstRecRes.png')
plt.show()


relErr = np.linalg.norm(MargX - VMR_O3)/np.linalg.norm(VMR_O3) * 100


print(f'relative Error: {relErr:.2f} %')

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




