import numpy as np
from AffineFunc import *
from numpy.random import uniform
import matplotlib as mpl
import time
import matplotlib.pyplot as plt
import scipy as scy
import os
cwd = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
from pathlib import Path
path = Path(cwd) # Path("/here/your/path/file.txt")
parentDir = str( path.parent.absolute())

fraction = 1.5
dpi = 300
PgWidthPt = 245
PgWidthPt = 421/2 #phd


print(mpl.get_backend())
plt.rcParams.update({'font.size':  10,
                     'text.usetex': True,
                     'font.family' : 'serif',
                     'font.serif'  : 'cm',
                     'text.latex.preamble': r'\usepackage{bm, amsmath}'})

ResCol = "#1E88E5"#"#0072B2"
MeanCol = 'k'#"#FFC107"#"#d62728"
RegCol = "#D81B60"#"#D55E00"
TrueCol = 'green' # "#004D40" #'k'
DatCol =  'gray' # 'k'"#332288"#"#009E73"
binCol = 'C0'
postCol = 'C1'
priorCol = 'k'
DatCol =  'gray'
TrueCol = 'C2'
alpha = 0.75


dir = parentDir + '/TTDecomposition/'
A_lin_dx = np.loadtxt(dir + 'A_lin_dx.txt')
tang_heights_lin = np.loadtxt(dir +'tan_height_values.txt')
height_values = np.loadtxt(dir +'height_values.txt')
height_values = height_values.reshape((len(height_values),1))
A = np.loadtxt(dir +'AMat.txt')
gamma0 = np.loadtxt(dir +'gamma0.txt')
#y = np.loadtxt(dir +'dataY.txt')
nonLinY = np.loadtxt(dir +'nonLinDataY.txt')
y = np.loadtxt(dir +'nonLinDataY.txt')
y = y.reshape((len(y),1))

ATA = np.matmul(A.T,A)
ATy = np.matmul(A.T, y)
#B_inv_A_trans_y0 = np.loadtxt('B_inv_A_trans_y0.txt')
VMR_O3 = np.loadtxt(dir +'VMR_O3.txt')
VMR_O3 = VMR_O3.reshape((len(VMR_O3), 1))
pressure_values = np.loadtxt(dir +'pressure_values.txt').reshape((len(height_values),1))
L = np.loadtxt(dir +'GraphLaplacian.txt')
theta_scale_O3 = np.loadtxt(dir + 'theta_scale_O3.txt')
num_mole = np.loadtxt(dir + 'num_mole.txt')
LineIntScal = np.loadtxt(dir +'LineIntScal.txt')
S = np.loadtxt(dir +'S.txt').reshape((909,1))
A_lin = np.loadtxt(dir +'ALinMat.txt')
nonLinA =  np.loadtxt(dir + 'nonLinA.txt')
temp_values = np.loadtxt(dir +'temp_values.txt').reshape((len(VMR_O3), 1))

wvnmbr = np.loadtxt(dir +'wvnmbr.txt').reshape((909,1))
E = np.loadtxt(dir +'E.txt').reshape((909,1))
g_doub_prime = np.loadtxt(dir +'g_doub_prime.txt').reshape((909,1))
g_prime = np.loadtxt(dir +'g_prime.txt').reshape((909,1))




AscalConstKmToCm = 1e3
ind = 623
betaG = 1e-35
betaD = 1e-35
alphaD = 1
alphaG = 1
m,n = A_lin.shape
SpecNumLayers = n
SpecNumMeas = m

## # check Amat
#mpl.use('qtagg')
Ax =np.matmul(A, VMR_O3 * theta_scale_O3)
nonLinAx = np.matmul( A/2 * nonLinA,VMR_O3 * theta_scale_O3)
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(PgWidthPt, fraction=fraction))
#ax1.plot(Ax, tang_heights_lin)
ax1.plot(nonLinAx, tang_heights_lin, color = 'k', label = 'noise-free data')
ax1.plot(y, tang_heights_lin, color = 'r', marker = 'o', linestyle = '--', linewidth = 0.5, label = 'noisy data')
ax1.set_xscale('log')
ax1.legend()
ax1.set_ylabel(r'tangent height $h_{\ell}$ in km')
ax1.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
plt.savefig('AffinePapData.png', dpi = dpi)
plt.show(block=True)
#plt.interactive(False)


relDiff = np.linalg.norm( nonLinAx -  y) / np.linalg.norm(nonLinAx) * 100
print(f'rel difference between data and noise free data is {relDiff:.2f}')

## calc Marg on Grid
gridSize = 25
GamBounds = [0.8e15, 1.2e16]
LambBounds = [1e-5, 8e-4]



n = len(height_values)
m = len(tang_heights_lin)

lamGrid = np.linspace(*LambBounds, gridSize)
gamGrid = np.linspace(*GamBounds, gridSize)
margGrid = np.zeros((gridSize,gridSize))
FGrid= np.zeros(gridSize)
GGrid= np.zeros(gridSize)
unNormlamMarg = np.zeros(gridSize)
means = np.zeros((gridSize,len(height_values)))



startTime  = time.time()

for i in range(0,gridSize):
    #lambdas
    B = (ATA + lamGrid[i] * L)
    LowTri = np.linalg.cholesky(B)
    currX = scy.linalg.cho_solve((LowTri, True), ATy[:, 0])
    FGrid[i] = f(ATy, y, currX)
    GGrid[i] = 2* np.sum(np.log(np.diag(LowTri)))

    # gammas
    margGrid[i, :] = np.exp(-CurrMarg(lamGrid[i], gamGrid, GGrid[i], FGrid[i], n, m, betaG, betaD) - 200)
    unNormlamMarg[i] = np.sum(margGrid[i, :])

    means[i] =  currX * unNormlamMarg[i]



unnormGamMarg = np.sum(margGrid, 0)
gamMarg = unnormGamMarg / np.sum(unnormGamMarg)
zLam = np.sum(unNormlamMarg)
postMean = np.sum(means, axis= 0)/zLam
lamMarg = unNormlamMarg/zLam


FullPostTime = time.time() - startTime
print(f'Elapsed Time to calc mean on a {gridSize} x {gridSize} grid: {FullPostTime:.5f}')


fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction))#, dpi = dpi)
axs[0].plot(gamGrid,gamMarg, color = 'k', marker = '.')
axs[0].set_ylim(0)
axs[0].set_xlabel(r'the noise precision $\gamma$')
axs[0].set_ylabel(r'$\pi(\gamma|\bm{y})$')

axs[1].plot(lamGrid,lamMarg, color = 'k', marker = '.')
axs[1].set_ylim(0)


axs[1].set_xlabel(r'the regularization parameter $\lambda =\delta / \gamma$')
axs[1].set_ylabel(r'$\pi(\lambda|\bm{y})$')
plt.savefig('AffinePapFirstMarg.png', dpi = dpi)

plt.show(block= True)

## f ang g on lambda grid
TrueGridSize = 200
TrueLamGrid = np.linspace(*LambBounds, TrueGridSize)
TrueFGrid= np.zeros(TrueGridSize)
TrueGGrid= np.zeros(TrueGridSize)
for i in range(0,TrueGridSize):
    #lambdas
    B = (ATA + TrueLamGrid[i] * L)
    LowTri = np.linalg.cholesky(B)
    currX = scy.linalg.cho_solve((LowTri, True), ATy[:, 0])
    TrueFGrid[i] = f(ATy, y, currX)
    TrueGGrid[i] = 2* np.sum(np.log(np.diag(LowTri)))


fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction))#, dpi = dpi)
axs[0].plot(lamGrid, FGrid, color = 'r', marker = '.', linestyle = 'none', label= 'grid points')
axs[0].plot(TrueLamGrid, TrueFGrid,color = 'k')

axs[0].set_xlabel(r'$\lambda$')
axs[0].set_ylabel(r'$f(\lambda)$')

axs[1].plot(lamGrid,GGrid, color = 'r', marker = '.', linestyle = 'none')
axs[1].plot(TrueLamGrid, TrueGGrid, color = 'k' , label = 'true function')


axs[1].set_xlabel(r'$\lambda$')
#axs[1].set_xscale('log')
axs[1].set_ylabel(r'$g(\lambda)$')
axs[0].legend()
axs[1].legend()
plt.savefig('AffinePapfandg.png', dpi = dpi)
plt.show(block= True)

##find affine map

# sample a test prof via RTO

numRTOSampl = SpecNumMeas
seeds = np.random.uniform(low=0.0, high=1.0, size=(numRTOSampl,2))
CDFLam = np.cumsum(lamMarg)
LowTriL = np.linalg.cholesky(L)
gamSampl = np.zeros(numRTOSampl)
#lamSampl = np.zeros(numRTOSampl)
lamSampl = np.interp(seeds[:, 0], CDFLam, lamGrid)
currF = np.interp(lamSampl, lamGrid, FGrid)
currG = np.interp(lamSampl, lamGrid, GGrid)
testO3Prof = np.zeros((numRTOSampl,SpecNumLayers))

for i in range(0, numRTOSampl):
    # shape = m / 2 + alphaD + alphaG
    # rate = currF[i] / 2 + betaG + betaD * lamSampl[i]
    # gamSampl[i] = np.random.gamma(shape=shape, scale=1 / rate)
    currCondMarg = np.exp(-CurrMarg(lamSampl[i], gamGrid, currG[i], currF[i], n, m, betaG, betaD) - 200)
    CDFGam = np.cumsum(currCondMarg/np.sum(currCondMarg))
    gamSampl[i] = np.interp(seeds[i, 1], CDFGam, gamGrid)
    B = (ATA + lamSampl[i] * L)
    LowTri = np.sqrt(gamSampl[i]) * np.linalg.cholesky(B)

    v = np.random.normal(0, 1, size=n + m)
    CurrATy = gamSampl[i] * ATy[:, 0] + np.sqrt(gamSampl[i]) * (A.T @ v[:m]) + np.sqrt(lamSampl[i] * gamSampl[i]) * (
                LowTriL @ v[m:])
    testO3Prof[i] = scy.linalg.cho_solve((LowTri, True), CurrATy)


# fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction))#, dpi = dpi)
# axs[0].plot(gamGrid,gamMarg, color = 'k')
# axsTw = axs[0].twinx()
# axsTw.hist(gamSampl)
# axs[0].set_xlabel(r'the noise precision $\gamma$')
# axs[0].set_ylabel(r'$\pi(\gamma|\bm{y})$')
#
# axs[1].plot(lamGrid,lamMarg, color = 'k')
# axsTw = axs[1].twinx()
# axsTw.hist(lamSampl)
# axs[1].set_xlabel(r'the regularization parameter $\lambda =\delta / \gamma$')
# axs[1].set_ylabel(r'$\pi(\lambda|\bm{y})$')
#
# plt.show(block= True)

##

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))

ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 3, markersize =15)

ax1.plot(postMean,height_values[:,0], markeredgecolor ='k', color = 'k' ,zorder=2, marker = '.', markersize =3, linewidth =1, label=r'posterior mean $\bm{\mu}_{\bm{x}|\bm{y}}$')
currProf = np.copy(testO3Prof[0])
currProf[currProf < 0] = 0
ax1.plot(currProf, height_values[:, 0], markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=3,
         linewidth=0.5, label = r'full cond.~post.~sample', alpha = alpha)

for i in range(1, numRTOSampl):
    currProf = np.copy(testO3Prof[i])
    currProf[currProf<0] = 0
    ax1.plot(currProf,height_values[:,0], markeredgecolor =binCol, color =binCol  ,zorder=1, marker = '.', markersize =3, linewidth =0.5, alpha = alpha)

ax1.set_xlabel(r'ozone volume mixing ratio ')
ax1.set_ylabel('height in km')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(loc = 'upper right')
ax1.set_ylim([height_values[0], height_values[-1]])
plt.savefig('AffinePapOzonSampl.png', dpi = dpi)
plt.show(block= True)


## find affine map

relMapErrDat = 1
RealMap, relMapErr, LinDataY, NonLinDataY, testO3 = genDataFindandtestMap(tang_heights_lin, A_lin_dx,  height_values, gamma0,postMean, testO3Prof, A_lin, temp_values, pressure_values, ind, relMapErrDat, wvnmbr, S, E, g_doub_prime, g_prime)



def nonLinF(O3):
    O3 = O3.reshape((len(O3),1))
    AO3, theta_scale_O3 = composeAforO3(A_lin, temp_values, pressure_values, ind)
    nonLinA = calcNonLin(tang_heights_lin, A_lin_dx, height_values, pressure_values, ind, temp_values, O3, wvnmbr,
                         S, E, g_doub_prime, g_prime)
    return np.matmul(AO3 * nonLinA, O3 * theta_scale_O3)


SinglTestO3 = testO3Prof[0].reshape((n,1))
linTestDat = np.matmul(A,SinglTestO3)
MapLinTestDat = np.matmul( RealMap @  A,SinglTestO3)
testNonLinA = calcNonLin(tang_heights_lin, A_lin_dx,  height_values, pressure_values, ind, temp_values, SinglTestO3, wvnmbr, S, E, g_doub_prime, g_prime)
nonLinTestDat = np.matmul( A/2 * testNonLinA,SinglTestO3)
testErr = np.sqrt(np.sum((MapLinTestDat - nonLinTestDat)**2)/np.sum((nonLinTestDat)**2))
print(testErr)
print('bla')



##



fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True)
ax4.plot(linTestDat,tang_heights_lin, linestyle = 'dotted', marker = '*', label = r'linear $\bm{A}_L\bm{x}$', markersize = 18 , zorder = 0, color = DatCol )
relErr = np.linalg.norm( MapLinTestDat -  nonLinTestDat) / np.linalg.norm(MapLinTestDat) * 100
ax4.plot(MapLinTestDat,tang_heights_lin, linestyle = 'dotted', marker = '*', label = r'$\bm{MA}_L\bm{x}$' + f', rel.~Err.: {relErr:.2f} \%', markersize = 7, zorder = 2, color ='k')
ax4.plot(nonLinTestDat,tang_heights_lin, linestyle = 'dotted', marker = 'o', label = r'non-linear $\bm{A}(\bm{x})$', markersize = 10, zorder = 1, color = 'r')
ax4.legend()
ax4.set_ylabel('(tangent) height in km')
ax4.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
plt.savefig('AffinePapAssMap.png', dpi = dpi)
plt.show(block= True)


##


A = RealMap @ np.copy(A)
ATA = A.T @ A
ATy = A.T @ y


## calc Marg on Grid for affine map
gridSize = 25
GamBounds = [0.8e15, 1.2e16]
LambBounds = [1e-5, 8e-4]
n = len(height_values)
m = len(tang_heights_lin)

lamGrid = np.linspace(*LambBounds, gridSize)
gamGrid = np.linspace(*GamBounds, gridSize)
margGrid = np.zeros((gridSize,gridSize))
FGrid= np.zeros(gridSize)
GGrid= np.zeros(gridSize)
unNormlamMarg = np.zeros(gridSize)
FinalMeans = np.zeros((gridSize,len(height_values)))


startTime  = time.time()

for i in range(0,gridSize):
    #lambdas
    B = (ATA + lamGrid[i] * L)
    LowTri = np.linalg.cholesky(B)
    currX = scy.linalg.cho_solve((LowTri, True), ATy[:, 0])
    FGrid[i] = f(ATy, y, currX)
    GGrid[i] = 2* np.sum(np.log(np.diag(LowTri)))

    # gammas
    margGrid[i, :] = np.exp(-CurrMarg(lamGrid[i], gamGrid, GGrid[i], FGrid[i], n, m, betaG, betaD) - 200)
    unNormlamMarg[i] = np.sum(margGrid[i, :])

    FinalMeans[i] =  currX * unNormlamMarg[i]



unnormGamMarg = np.sum(margGrid, 0)
gamMarg = unnormGamMarg / np.sum(unnormGamMarg)
zLam = np.sum(unNormlamMarg)
FinalPostMean = np.sum(FinalMeans, axis= 0)/zLam
lamMarg = unNormlamMarg/zLam


# final sample a test prof via RTO

numRTOSampl = 10000
seeds = np.random.uniform(low=0.0, high=1.0, size=(numRTOSampl,2))
CDFLam = np.cumsum(lamMarg)
LowTriL = np.linalg.cholesky(L)
gamSampl = np.zeros(numRTOSampl)
#lamSampl = np.zeros(numRTOSampl)
lamSampl = np.interp(seeds[:, 0], CDFLam, lamGrid)
currF = np.interp(lamSampl, lamGrid, FGrid)
currG = np.interp(lamSampl, lamGrid, GGrid)
FinalO3Sampl = np.zeros((numRTOSampl,SpecNumLayers))

for i in range(0, numRTOSampl):
    # shape = m / 2 + alphaD + alphaG
    # rate = currF[i] / 2 + betaG + betaD * lamSampl[i]
    # gamSampl[i] = np.random.gamma(shape=shape, scale=1 / rate)
    currCondMarg = np.exp(-CurrMarg(lamSampl[i], gamGrid, currG[i], currF[i], n, m, betaG, betaD) - 200)
    CDFGam = np.cumsum(currCondMarg/np.sum(currCondMarg))
    gamSampl[i] = np.interp(seeds[i, 1], CDFGam, gamGrid)
    B = (ATA + lamSampl[i] * L)
    LowTri = np.sqrt(gamSampl[i]) * np.linalg.cholesky(B)

    v = np.random.normal(0, 1, size=n + m)
    CurrATy = gamSampl[i] * ATy[:, 0] + np.sqrt(gamSampl[i]) * (A.T @ v[:m]) + np.sqrt(lamSampl[i] * gamSampl[i]) * (
                LowTriL @ v[m:])
    FinalO3Sampl[i] = scy.linalg.cho_solve((LowTri, True), CurrATy)

FullPostTime = time.time() - startTime
print(f'Elapsed Time to calc mean and covarianve on a {gridSize} x {gridSize} grid: {FullPostTime:.5f}')


FinalVar = np.var(FinalO3Sampl[:500], axis = 0)



fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction))#, dpi = dpi)
axs[0].plot(gamGrid,gamMarg, color = 'k', marker = '.')

axs[0].set_xlabel(r'the noise precision $\gamma$')
axs[0].set_ylabel(r'$\pi(\gamma|\bm{y})$')
axs[0].set_ylim(0)
axs[1].plot(lamGrid,lamMarg, color = 'k', marker = '.')
#axs[1].set_ylim(0)


axs[1].set_xlabel(r'the regularization parameter $\lambda =\delta / \gamma$')
axs[1].set_ylabel(r'$\pi(\lambda|\bm{y})$')
plt.savefig('AffinePapMargFinal.png', dpi = dpi)
## calc True Post Mean and Post Covar
gridSize = 200
GamBounds = [0.8e15, 1.2e16]
LambBounds = [1e-5, 8e-4]
lamGrid = np.linspace(*LambBounds, gridSize)
gamGrid = np.linspace(*GamBounds, gridSize)
margGrid = np.zeros((gridSize,gridSize))
FGrid= np.zeros(gridSize)
GGrid= np.zeros(gridSize)
unNormlamMarg = np.zeros(gridSize)
TrueMeans = np.zeros((gridSize,len(height_values)))
TrueCovars = np.zeros((gridSize,len(height_values),len(height_values)))
I = np.eye(n)

startTime  = time.time()

for i in range(0,gridSize):
    #lambdas
    B = (ATA + lamGrid[i] * L)
    LowTri = np.linalg.cholesky(B)
    currX = scy.linalg.cho_solve((LowTri, True), ATy[:, 0])
    FGrid[i] = f(ATy, y, currX)
    GGrid[i] = 2* np.sum(np.log(np.diag(LowTri)))

    # gammas
    margGrid[i, :] = np.exp(-CurrMarg(lamGrid[i], gamGrid, GGrid[i], FGrid[i], n, m, betaG, betaD) - 200)
    unNormlamMarg[i] = np.sum(margGrid[i, :])

    TrueMeans[i] =  currX * unNormlamMarg[i]
    TrueCovars[i] = scy.linalg.cho_solve((LowTri, True), I) * unNormlamMarg[i]



unnormGamMarg = np.sum(margGrid, 0)
gamMarg = unnormGamMarg / np.sum(unnormGamMarg)
zLam = np.sum(unNormlamMarg)
TruePostMean = np.sum(TrueMeans, axis= 0)/zLam
TruePostCovar =  np.sum(TrueCovars, axis = 0)/zLam * np.sum( gamMarg / gamGrid)
lamMarg = unNormlamMarg/zLam


lower_error = np.sqrt(FinalVar)
upper_error = np.sqrt(FinalVar)
lower_error[(FinalPostMean - lower_error) < 0] =  FinalPostMean[(FinalPostMean- lower_error) < 0]
lower_error[FinalPostMean < 0] =  0
asymmetric_error = np.array(list(zip(lower_error, upper_error))).T

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))

ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 3, markersize =15)

#ax1.errorbar(TruePostMean,height_values[:,0],  xerr =np.sqrt(np.diag(TruePostCovar)) , markeredgecolor ='r', color = 'r' ,zorder=2, marker = '.', markersize =5, linewidth =1.5, capsize = 3)
#ax1.errorbar(TruePostMean,height_values[:,0],  yerr =np.sqrt(np.diag(TruePostCovar)) * 0, markeredgecolor ='r', color = 'r' ,zorder=2, marker = '.', label = r' ``true`` post.~$\bm{\mu}_{\bm{x}|\bm{y}} \pm \bm{\Sigma}_{\bm{x}|\bm{y}}$', markersize =5, linewidth =1.5, capsize = 3)

line3 = ax1.errorbar(FinalPostMean,height_values[:,0],xerr = asymmetric_error, markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =3, linewidth =1, capsize = 3 )#, markerfacecolor = 'none'
line3 = ax1.errorbar(FinalPostMean,height_values[:,0],yerr = np.sqrt(FinalVar) * 0, label = r'$\bm{\mu}_{\bm{x}|\bm{y}} \pm$ sample based STD' ,markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =3, linewidth =1, capsize = 3 )#, markerfacecolor = 'none'


ax1.set_xlabel(r'ozone volume mixing ratio ')
ax1.set_ylabel('height in km')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(loc = 'upper right')
ax1.set_ylim([height_values[0], height_values[-1]])
plt.savefig('AffinePapOzoFinal.png', dpi = dpi)
plt.show(block= True)


CurrRelErrCovar = np.sqrt((np.sum(FinalVar -np.diag(TruePostCovar))**2)/np.sum(np.diag(TruePostCovar)**2))

print(CurrRelErrCovar)


CurrRelMeanErr = np.sqrt(np.sum(FinalPostMean - TruePostMean)**2/np.sum(TruePostMean)**2)
print(f'posterior mean RMS error: {CurrRelMeanErr*100}')
##
StartRange = 10
relErrCovar = np.zeros(numRTOSampl-StartRange)
for i in range(StartRange ,numRTOSampl+1):
    CurrVar = np.mean((FinalO3Sampl[-i:] - FinalPostMean)**2)

    relErrCovar[i-StartRange-1] = np.sqrt((np.sum(CurrVar - np.diag(TruePostCovar)) ** 2) / np.sum(
        np.diag(TruePostCovar) ** 2))

##

fig3, ax1 = plt.subplots(figsize=(PgWidthPt/ 72.27* 1.5,PgWidthPt/2 /72.27*1.5), tight_layout = True)
ax1.plot(range(StartRange ,numRTOSampl),relErrCovar*100, color = 'k')

ax1.set_xscale('log')
ax1.set_xlabel('number of samples N')
ax1.set_ylabel('rel.~RMS err.~in $\%$')
plt.savefig('AffinePapErr.png', dpi = dpi)
plt.show(block= True)

##
StartRange = 1
relErrCovar = np.zeros(numRTOSampl-StartRange)
for i in range(StartRange ,numRTOSampl+1):
    #CurrVar = np.mean((FinalO3Sampl[-i:] - FinalPostMean)**2)
    CurrVar = np.sqrt(np.var(FinalO3Sampl[-i:],axis = 0, mean = np.copy(FinalPostMean)))
    CurrMean = np.copy(FinalPostMean)
    #CurrMean = np.abs(np.mean(FinalO3Sampl[-i:],axis = 0))
    relErrCovar[i-StartRange-1] = np.mean(CurrVar /CurrMean)

##

fig3, ax1 = plt.subplots(figsize=(PgWidthPt/ 72.27* 1.5,PgWidthPt/2 /72.27*1.5), tight_layout = True)
ax1.plot(range(StartRange ,numRTOSampl),relErrCovar*100, color = 'k')

ax1.set_xscale('log')
ax1.set_xlabel('number of samples N')
ax1.set_ylabel('Coefficient of Variation~in $\%$')
plt.savefig('AffinePapCV.png', dpi = dpi)
plt.show(block= True)