from APEXFunctions import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy as scy
from matplotlib.ticker import FuncFormatter
import time

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

TrueCol = [50/255,220/255, 0/255]#'#02ab2e'

ResCol = "#1E88E5"#"#0072B2"
MeanCol = 'k'#"#FFC107"#"#d62728"
RegCol = "#D81B60"#"#D55E00"
TrueCol = 'green' # "#004D40" #'k'
DatCol =  'gray' # 'k'"#332288"#"#009E73"

postCol = 'C1'


dpi = 300
PgWidthPt = 421/2
defBack = mpl.get_backend()
mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 10,
                     'text.usetex': True,
                     'font.family' : 'serif',
                     'font.serif'  : 'cm',
                     'text.latex.preamble': r'\usepackage{bm, amsmath, amssymb,amsfonts}'})




df = pd.read_excel('ExampleOzoneProfiles.xlsx')

#print the column names
print(df.columns)

#get the values for a given column
press = df['Pressure (hPa)'].values #in hectpascal or millibars
O3 = df['Ozone (VMR)'].values
O3[O3<0] = 0
#dir = '/home/lennartgolks/PycharmProjects/'
#dir = '/Users/lennart/PycharmProjects/'
dat = np.loadtxt('testProf.txt')
#dat = np.loadtxt('/home/lennartgolks/PycharmProjects/openData/testProf.txt')
press = dat[0,:]
O3 = dat[1,:]
O3[O3 < 0] = 0


minInd = 5
maxInd = 50#47#51#54#47
skipInd = 1
pressure_values = press[minInd:maxInd][::skipInd]#press[minInd:maxInd]
VMR_O3 = O3[minInd:maxInd][::skipInd]#O3[minInd:maxInd]
#VMR_O3[:] = np.mean(VMR_O3)
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
pressure_values = new_calc_press.reshape((SpecNumLayers,1))

SpecNumLayers = len(height_values)

R_Earth = 6356#6371 # earth radiusin km
ObsHeight = 500 # in km

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
pointAcc = 0.00085
meas_angChosen = np.array(np.arange(MinAng[0], MaxAng[0], pointAcc))[:30]
A_lin_dx, tang_heights_linChosen, extraHeight = gen_forward_map(meas_angChosen,height_values,ObsHeight,R_Earth)




A_lin_dx, tang_heights_lin, extraHeight = gen_forward_map(meas_angChosen,height_values,ObsHeight,R_Earth)
SpecNumMeas = len(tang_heights_lin)
m = SpecNumMeas
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
g_prime= np.zeros((size[0],1))

for i, lines in enumerate(data_set):
    wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
    S[i] = float(lines[0][16:25]) # in cm/mol
    F[i] = float(lines[0][26:35])
    g_air[i] = float(lines[0][35:40])
    g_self[i] = float(lines[0][40:45])
    E[i] = float(lines[0][46:55])
    n_air[i] = float(lines[0][55:59])
    g_doub_prime[i] = float(lines[0][148:153])
    g_prime[i] = float(lines[0][155:160])



np.savetxt('S.txt',S, delimiter= '\t')
#load constants in si annd convert to cgs units by multiplying
h = scy.constants.h #* 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b = constants.Boltzmann #* 1e7#in J K^-1
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






AO3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind)
A = 2*AO3


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
linNoiseFreeDat = Ax
#convolve measurements and add noise
#y = add_noise(Ax, 0.01)
#y[y<=0] = 0
SNR = 60#35000
SNR = 150
#y, gam0 = add_noise(Ax.reshape((SpecNumMeas,1)), SNR)
##
#y = Ax.reshape((SpecNumMeas,1)) + np.random.normal(0,0.00025,size=(SpecNumMeas,1))

y, gam0  = add_noise_Blokk(Ax,SNR)

##

nonLinA = calcNonLin(tang_heights_lin, A_lin_dx, height_values, pressure_values, ind, temp_values, VMR_O3, wvnmbr, S, E,g_doub_prime,g_prime)
OrgData = np.matmul(AO3 * nonLinA,VMR_O3 * theta_scale_O3)
DatErr = np.linalg.norm( OrgData -  Ax) / np.linalg.norm(OrgData) * 100
print('DataErr '+ str(DatErr))

Ax =np.matmul(2*AO3, VMR_O3 * theta_scale_O3)


noise = np.random.normal(0, np.sqrt(1 / gam0), size = OrgData.shape)
nonLinY = (OrgData + noise).reshape((SpecNumMeas,1))
Liny = (Ax + noise).reshape((SpecNumMeas,1))


fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))

ax1.scatter(nonLinY, tang_heights_lin)
ax1.plot(nonLinY, tang_heights_lin, label = 'noisy')
ax1.plot(Ax, tang_heights_lin, label = 'noise free')
plt.legend()
ax1.set_xscale('log')
plt.show(block = True)




##

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout=True)
ax1.plot((A/2 * nonLinA) @ (VMR_O3 * theta_scale_O3),tang_heights_lin, label = 'noise-free data', color = 'k')
ax1.plot(y,tang_heights_lin, label  = 'noisy data', linestyle = 'dotted', marker = 'o', markersize= 10, zorder = 1, color = 'r', linewidth = 1)
ax1.set_xscale('log')
ax1.legend()
ax1.set_ylabel(r'tangent height $h_{\ell}$ in km')
ax1.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
#fig3.savefig('DataPlot.png', dpi = dpi)
plt.show(block = True)



## new forwrad model



startInd = 23
EndInd = len(height_values[startInd::2]) + startInd

height_values[startInd:EndInd] = np.copy(height_values[startInd::2])
height_values = np.copy(height_values[:EndInd])


temp_values[startInd:EndInd] = np.copy(temp_values[startInd::2])
temp_values = np.copy(temp_values[:EndInd])

pressure_values[startInd:EndInd] = np.copy(pressure_values[startInd::2])
pressure_values = np.copy(pressure_values[:EndInd])

VMR_O3[startInd:EndInd] = np.copy(VMR_O3[startInd::2])
VMR_O3 = np.copy(VMR_O3[:EndInd])
SpecNumLayers = len(height_values)
n =len(height_values)

NOfNeigh = 2#4
neigbours = np.zeros((len(height_values),NOfNeigh))
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
#cholesky decomposition of L for W1 and v1
lowC_L = scy.linalg.cholesky(L, lower = True)

##
A_lin_dx, tang_heights_lin, extraHeight = gen_forward_map(meas_angChosen,height_values,ObsHeight,R_Earth)

A_lin = gen_sing_map(A_lin_dx, tang_heights_lin, height_values)
AO3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind)
A = 2*AO3
ATA = np.matmul(A.T,A)
y = np.copy(nonLinY)
ATy = np.matmul(A.T, y)
nonLinA = calcNonLin(tang_heights_lin, A_lin_dx, height_values, pressure_values, ind, temp_values, VMR_O3, wvnmbr, S, E,g_doub_prime,g_prime)





##
"""start the mtc algo with first guesses of noise and lumping const delta"""

theta = VMR_O3 * theta_scale_O3
vari = np.zeros((len(theta)-2,1))

for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])
    #vari[j - 1] = abs(-theta[j + 1] + 2*theta[j] - theta[j - 1])**2
if np.mean(vari) == 0:
    vari = 1
##
#find minimum for first guesses
'''params[1] = delta
params[0] = gamma'''
def MinLogMargPost(params):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gam = params[0]
    lamb = params[1]
    #print(lamb)
    if lamb < 0  or gam < 0:
        return np.nan

    n = SpecNumLayers
    m = SpecNumMeas
    #print(lamb)
    Bp = ATA + lamb * L

    LowTri = np.linalg.cholesky(Bp)
    UpTri = LowTri.T
    # check if L L.H = B
    B_inv_A_trans_y = scy.linalg.cho_solve((LowTri,True),  ATy[:,0])

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)

minimum = scy.optimize.fmin(MinLogMargPost, [gam0,1/gam0* 1/ np.mean(vari)/15], maxiter = 25)
gam0 = minimum[0]
lam0 = minimum[1]
print(minimum)



##
""" finally calc f and g with a linear solver adn certain lambdas
 using the gmres"""

lam= np.logspace(-10,15,500)
f_func = np.zeros(len(lam))
g_func = np.zeros(len(lam))



for j in range(len(lam)):

    B = (ATA + lam[j] * L)

    #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], rtol=tol, restart=25)
    LowTri = np.linalg.cholesky(B)

    B_inv_A_trans_y = scy.linalg.cho_solve((LowTri,True),  ATy[:,0])

    CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)

    f_func[j] = f(ATy, y, B_inv_A_trans_y)

    g_func[j] = g(A, L, lam[j])



##
''' check taylor series in f(lambda) and g(lambda)
around lam0 from gmres = '''

#taylor series arounf lam_0
lam0 = minimum[1]
B = (ATA + lam0 * L)

LowTri = np.linalg.cholesky(B)

B_inv_A_trans_y0 = scy.linalg.cho_solve((LowTri,True),  ATy[:,0])





LowTri = np.linalg.cholesky(B)
B_inv_L = scy.linalg.cho_solve((LowTri,True),  L)


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



# g_0_1 = np.trace(B_inv_L)
# g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
# g_0_3 = 1 /6 * np.trace(B_inv_L_3)
# g_0_4 = 0#-1 /24 * np.trace(B_inv_L_4)
# g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
# g_0_6 = 0#-1 /720 * np.trace(B_inv_L_6)

f_0 = f(ATy, y, B_inv_A_trans_y0)
#g_0 = g(A, L,minimum[1] )
#delG = (np.log(g(A, L, 4e3)) - np.log(g_0))/ (np.log(4e3) - np.log(minimum[1]))


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
index = 'sec'
gridSize = 20
#univarGridO3 = np.zeros((2, gridSize))
# for i in range(0, 2):
#     univarGridO3[i] = np.loadtxt(parentDir + '/TTDecomposition/'+index +'uniVarGridMargO3' + str(i) + '.txt')
#

#f_new = f_0
#g_old = g(A, L,  lambdas[0])
lamMax = 0.3 * lam0
g_0 = g(A, L, lam0)
delG = (g(A, L, lamMax) - g_0)/ (np.log(lamMax) - np.log(lam0))
def MHwG(number_samples, burnIn, lam0, gamma0, f_0, g_0):
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
        log_MH_ratio = ((SpecNumLayers)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

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


startTime = time.time()
lambdas ,gammas, k = MHwG(number_samples, burnIn, lam0, gam0, f_0, g_0)
elapsed = time.time() - startTime
print('MTC Done in ' + str(elapsed) + ' s')



print('acceptance ratio: ' + str(k/(number_samples+burnIn)))
deltas = lambdas * gammas
np.savetxt('samples.txt', np.vstack((gammas[burnIn::], deltas[burnIn::], lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')

# gam_mean, gam_del, gam_tint, gam_d_tint= tauint([[gammas]],0)
# lam_mean, lam_del, lam_tint, lam_d_tint = tauint([[lambdas]],0)

##

BinHist = 30#n_bins
lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density= True)
gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)
delHist, delBinEdges = np.histogram(deltas, bins= BinHist, density= True)
trace = [MinLogMargPost(np.array([lambdas[burnIn+ i],gammas[burnIn+ i]])) for i in range(number_samples)]


fig, axs = plt.subplots(3, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction) )#, dpi = dpi)

axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(gamBinEdges)[0])#10)

axs[0].axvline(gam0)
axs[0].set_xlabel(r'the noise precision $\gamma$')


axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)
axs[2].bar(delBinEdges[1::],delHist*np.diff(delBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(delBinEdges)[0])#10)
#axs[2].plot( range(number_samples), trace, color = 'k')
#axs[2].set_ylabel(r'$\pi(\bm{\theta}|\bm{y})$')
#axs[2].set_xlabel('number of samples')
#plt.savefig('HistoPlotMain.png')
plt.show()
##
''' check taylor series in f(lambda) and g(lambda)
around lam0 from gmres = '''

#taylor series arounf lam_0
B = (ATA + lam0 * L)

LowTri = np.linalg.cholesky(B)

B_inv_A_trans_y0 = scy.linalg.cho_solve((LowTri,True),  ATy[:,0])




LowTri = np.linalg.cholesky(B)

B_inv_L = scy.linalg.cho_solve((LowTri,True),  L)

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
g_0 = g(A, L,lam0)
lamMax = max(lambBinEdges)
delG = (g(A, L, lamMax) - g(A, L,min(lambBinEdges)))/ (np.log(lamMax) - np.log(min(lambBinEdges)))

GApprox = (np.log(lambBinEdges) - np.log(lam0)) * delG  + g_0
taylorG = GApprox
# fig,axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)#, dpi = dpi)
# axs.plot(lam,f_func, color = fCol, zorder = 2, linestyle=  'dotted')
# ax2 = axs.twinx() # ax1 and ax2 share y-axis
# ax2.plot(lam,g_func, color = gCol, zorder = 2, linestyle=  'dashed')
# axs.set_xscale('log')
# axs.set_yscale('log')
# plt.show(block =True)

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

axin2.set_ylim(1.002*taylorG[0],0.995 * taylorG[-1])
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
#fig.savefig('f_and_g_paper.png', bbox_inches='tight')
plt.show(block = True)
##
# print max rel F taylor F error
f_Checkfunc = np.zeros(len(lambBinEdges))
for j in range(len(lambBinEdges)):

    B = (ATA + lambBinEdges[j] * L)

    #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], rtol=tol, restart=25)
    LowTri = np.linalg.cholesky(B)
    B_inv_A_trans_y = scy.linalg.cho_solve((LowTri,True),  ATy[:,0])

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
    B_inv_A_trans_y = scy.linalg.cho_solve((LowTri,True),  RandX)

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
        B_inv_A_trans_y = scy.linalg.cho_solve((LowTri,True),  ATy[:,0])

        MargResults[p, :] = B_inv_A_trans_y * margPDF[1,p]

        if Var == True:
            LowTri = np.linalg.cholesky(SetB)
            B_inv[p] = scy.linalg.cho_solve((LowTri,True),  IDiag)
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


## do grid

def CurrMarg(lamb, gam,G, F, n, m, betaG, betaD):

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gam) + 0.5 * G + 0.5 * gam * F +  ( betaD *  lamb * gam + betaG *gam)

startTime  = time.time()
n = len(height_values)
m = len(tang_heights_lin)
gridSize = 50
lamGrid = np.linspace(min(lambdas), max(lambdas), gridSize)
gamGrid = np.linspace(min(gammas), max(gammas), gridSize)
margGrid = np.zeros((gridSize,gridSize))
unNormlamMarg = np.zeros(gridSize)
means = np.zeros((gridSize,len(height_values)))
CoVars = np.zeros((gridSize,len(height_values),len(height_values)))
postCovar = np.zeros((gridSize,len(height_values),len(height_values)))
I = np.eye(n)
for i in range(0,gridSize):
    #lambdas
    B = (ATA + lamGrid[i] * L)
    LowTri = np.linalg.cholesky(B)
    currX = scy.linalg.cho_solve((LowTri, True), ATy[:, 0])
    currF = f(ATy, y,currX)
    currG = 2* np.sum(np.log(np.diag(LowTri)))
    CoVars[i] = scy.linalg.cho_solve((LowTri, True), I)
    # gammas
    margGrid[i, :] = np.exp(-CurrMarg(lamGrid[i], gamGrid, currG, currF, n, m, betaG, betaD) - 500)
    unNormlamMarg[i] = np.sum(margGrid[i, :])

    postCovar[i] = CoVars[i] * unNormlamMarg[i]
    means[i] =  currX * unNormlamMarg[i]



unnormGamMarg = np.sum(margGrid, 0)
gamMarg = unnormGamMarg / np.sum(unnormGamMarg)
zLam = np.sum(unNormlamMarg)
postMean = np.sum(means, axis= 0)/zLam

lamMarg = unNormlamMarg/zLam


FinalPostCovar =  np.sum(postCovar, axis = 0)/zLam * np.sum( gamMarg / gamGrid)


FullPostTime = time.time() - startTime
print('Elapsed Time to calc: ' + str(FullPostTime))


##
fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))

ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 1.5, markersize =7)

line3 = ax1.plot(postMean,height_values[:,0], markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.',  label = r'$\text{E}_{\bm{x},\bm{\theta}|\bm{y}} [\bm{x}]$', markersize =3, linewidth =1)#, markerfacecolor = 'none'
line3 = ax1.errorbar(postMean,height_values,  xerr = np.sqrt(np.diag(FinalPostCovar)), markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.', label = 'posterior mean ', markersize =3, linewidth =1, capsize = 3)#, markerfacecolor = 'none'


ax1.set_xlabel(r'ozone volume mixing ratio ')


plt.show(block= True)
n_bins = 20
fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction), gridspec_kw={'height_ratios': [3, 1]} )#, dpi = dpi)

axs[0].hist(gammas, bins=n_bins)

axT = axs[0].twinx()
axT.plot(gamGrid,gamMarg)
axT.set_ylim(0)
axs[0].set_xlabel(r'the noise precision $\gamma$')
axs[0].set_ylabel(r'the smoothnes parameter $\delta$')

axs[1].hist(lambdas, bins=n_bins)
axT = axs[1].twinx()
axT.plot(lamGrid,lamMarg)
axT.set_ylim(0)
axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)

plt.show(block= True)



##
# '''
# L-curve refularoization
# '''

#lamLCurve = np.logspace(1,7,200)
lamLCurve = np.logspace(-7,-1,200)
#lamLCurve = np.linspace(1e-15,1e3,200)

NormLCurve = np.zeros(len(lamLCurve))
xTLxCurve = np.zeros(len(lamLCurve))
xTLxCurve2 = np.zeros(len(lamLCurve))
for i in range(len(lamLCurve)):
    B = (ATA + lamLCurve[i] * L)

    LowTri = np.linalg.cholesky(B)

    x = scy.linalg.cho_solve((LowTri,True),  ATy[:,0])

    NormLCurve[i] = np.linalg.norm( np.matmul(A,x) - y[0::,0])
    xTLxCurve[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))

import kneed
startTime  = time.time()
#lamLCurveZoom = np.logspace(0,7,200)
lamLCurveZoom = np.copy(lamLCurve)
#lamLCurveZoom = np.logspace(-5,15,200)
NormLCurveZoom = np.zeros(len(lamLCurveZoom))
xTLxCurveZoom = np.zeros(len(lamLCurveZoom))
for i in range(len(lamLCurveZoom)):
    B = (ATA + lamLCurveZoom[i] * L)

    LowTri = np.linalg.cholesky(B)
    x = scy.linalg.cho_solve((LowTri,True),  ATy[:,0])

    NormLCurveZoom[i] = np.linalg.norm( np.matmul(A,x) - y[:,0])
    xTLxCurveZoom[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))

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
x_opt = scy.linalg.cho_solve((LowTri,True),  ATy[:,0])
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
#plt.savefig('LCurve.png')
#plt.savefig('LCurve.png')
#tikzplotlib.save("LCurve.tex")
plt.show(block = True)


#tikzplotlib_fix_ncols(fig)
#tikzplotlib.save("LCurve.pgf")
print('bla')

#np.savetxt('RegSol.txt',x_opt /(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst), fmt = '%.15f', delimiter= '\t')
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
#plt.savefig('ScatterplusHisto.png')
plt.show()


##

gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)
fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction) )#, dpi = dpi)

axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(gamBinEdges)[0])#10)


axs[0].set_xlabel(r'the noise precision $\gamma$')


axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = MTCCol, zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_title(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)
#plt.savefig('HistoPlot.png')
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

#plt.savefig('AllHistoPlot.png')
plt.show()

###
plt.close('all')

TrueCol = [50/255,220/255, 0/255]#'#02ab2e'

XOPT = x_opt /theta_scale_O3
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
#ax1.axhline(height_values[startInd])
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

line3 = ax1.errorbar(postMean,height_values,  xerr = np.sqrt(np.diag(FinalPostCovar)), markeredgecolor ="r", color = "r" ,zorder=4, marker = '.', label = 'posterior mean ', markersize =3, linewidth =0.5, capsize = 3)#, markerfacecolor = 'none'



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
#fig3.savefig('FirstRecRes.png')
plt.show(block= True)


relErr = np.linalg.norm(MargX - VMR_O3)/np.linalg.norm(VMR_O3) * 100


print(f'relative Error: {relErr:.2f} %')

Samp = Results[::15,:] / theta_scale_O3

