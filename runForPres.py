import numpy as np
from RunFunc import *
from scipy import constants, optimize
from numpy.random import uniform, normal, gamma
import matplotlib as mpl
import time
import matplotlib.pyplot as plt
import scipy as scy


fraction = 1.5
dpi = 300
PgWidthPt = 245
PgWidthPt = 421/2 #phd

defBack = mpl.get_backend()
mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
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

dir = '/home/lennartgolks/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/TTDecomposition/'
#dir = '/home/lennartgolks/PycharmProjects/TTDecomposition/'

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
pressure_values = np.loadtxt(dir +'pressure_values.txt')
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
AscalConstKmToCm = 1e3
ind = 623
f_broad = 1
scalingConst = 1e11
betaD = 1e-10
betaG = 1e-10
m,n = A_lin.shape
SpecNumLayers = n
SpecNumMeas = m
lowC_L = scy.linalg.cholesky(L, lower = True)
### check Amat

Ax =np.matmul(A, VMR_O3 * theta_scale_O3)
fig3, ax1 = plt.subplots(tight_layout = True,figsize=set_size(245, fraction=fraction))
ax1.plot(Ax, tang_heights_lin)
ax1.scatter(y, tang_heights_lin, color = 'r')
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

minimum = optimize.fmin(MinLogMargPost, [gamma0,1/gamma0* 1/ np.mean(vari)/15], maxiter = 25)
gam0 = minimum[0]
lam0 = minimum[1]
print(minimum)





##
''' check taylor series in f(lambda) and g(lambda)
around lam0 from gmres = '''

#taylor series arounf lam_0

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


##

'''do the sampling'''
number_samples = 20000
burnIn = 100
f_0 = f(ATy, y, B_inv_A_trans_y0)
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
        delta_g = g_0_1 * delta_lam + g_0_2 * (delta_lam_p**2 - delta_lam_t**2) + g_0_3 * (delta_lam_p**3 - delta_lam_t**3) #+ g_0_4 * delta_lam**4 + g_0_5 * delta_lam**5

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
lambdas ,gammas, k = MHwG(number_samples, burnIn, lam0, gam0, f_0)
elapsed = time.time() - startTime
print('MTC Done in ' + str(elapsed) + ' s')



print('acceptance ratio: ' + str(k/(number_samples+burnIn)))
deltas = lambdas * gammas
np.savetxt('FirstSamples.txt', np.vstack((gammas[burnIn::], deltas[burnIn::], lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')



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

oldMargInteg = np.sum(MargResults,0) / theta_scale_O3
MargInteg= np.copy(oldMargInteg)
BinHist = BinHistStart
MargTime = time.time() - startTime
print('Post Mean in ' + str(MargTime) + ' s')
oldRelErr = 0
print(BinHistStart)

for BinHist in range(BinHistStart+1,100):

    lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density =True)
    gamHist, gamBinEdges = np.histogram(gammas, bins=BinHist, density=True)
    MargResults = np.zeros((BinHist,len(theta)))
    B_inv_Res = np.zeros((BinHist,len(theta)))
    VarB = np.zeros((BinHist, len(L), len(L)))
    gamInt = np.zeros(BinHist)
    meanGamInt = np.zeros(BinHist)
    IDiag = np.eye(len(SetB))
    startTime  = time.time()
    for p in range(BinHist):

        SetLambda =  lambBinEdges[p] + (lambBinEdges[p] + lambBinEdges[p+1])/2
        SetGamma = gamBinEdges[p] + (gamBinEdges[p] + gamBinEdges[p+1])/2
        SetB = ATA + SetLambda * L

        LowTri = np.linalg.cholesky(SetB)
        UpTri = LowTri.T
        B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

        MargResults[p, :] = B_inv_A_trans_y * lambHist[p]/ np.sum(lambHist)
        B_inv_Res[p, :] = B_inv_A_trans_y

        B_inv = np.zeros(SetB.shape)
        # startTime = time.time()
        LowTri = np.linalg.cholesky(SetB)
        UpTri = LowTri.T
        for j in range(len(B)):
            B_inv[:, j] = lu_solve(LowTri, UpTri, IDiag[:, j])

        VarB[p] = B_inv * lambHist[p]/ np.sum(lambHist)
        gamInt[p] = 1 / SetGamma * gamHist[p]/ np.sum(gamHist)
        meanGamInt[p] = SetGamma * gamHist[p]/ np.sum(gamHist)

    newMargInteg = np.sum(MargResults,0)  / theta_scale_O3
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

MargInteg= np.copy(newMargInteg)

#CondVar = scy.integrate.trapezoid(gamInt) * scy.integrate.trapezoid(VarB.T) / (theta_scale_O3) ** 2
CondVar =np.sum(gamInt) * np.sum(VarB,0)  / (theta_scale_O3) ** 2

MargTime = time.time() - startTime
print('Post Mean in ' + str(MargTime) + ' s')

relErrO3 = np.linalg.norm(MargInteg -VMR_O3[:,0]) / np.linalg.norm(MargInteg) * 100
print('rel Error to ground Truth:' + str(relErrO3))

print('MTC Done in ' + str(elapsed) + ' s')


#BinHist = 30#n_bins
lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density= True)
gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)
fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction) , dpi = 300)

axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], color = 'k', zorder = 0,width = np.diff(gamBinEdges)[0])#10)


axs[0].set_xlabel(r'the noise precision $\gamma$')


axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = 'k', zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_xlabel(r'$\lambda =\delta / \gamma$, the regularization parameter', fontsize = 12)
plt.savefig('HistoPlot.png')
plt.show()

##
TrueCol = [50/255,220/255, 0/255]#'#02ab2e'

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), dpi = 300)

#line3 = ax2.scatter(y, tang_heights_lin, label = r'data $\mathbf{y}$', zorder = 0, marker = '*', color =DatCol )#,linewidth = 5

#ax1 = ax2#.twiny()
#ax1.scatter(VMR_O3,height_values,marker = 'o', facecolor = 'None', color = "#009E73", label = 'true profile', zorder=1, s =12)#,linewidth = 5)
ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 1.5, markersize =7)

#line3 = ax1.plot(MargInteg,height_values[:,0], markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.',  label = r'$\text{E}_{\mathbf{x},\mathbf{\theta}|\mathbf{y}} [\mathbf{x}]$', markersize =3, linewidth =1)#, markerfacecolor = 'none'
line3 = ax1.errorbar(MargInteg,height_values[:,0],xerr = np.sqrt(np.diag(CondVar)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =3, linewidth =1, capsize = 3 )#, markerfacecolor = 'none'
ax1.errorbar(MargInteg,height_values[:,0],  yerr = np.zeros(len(height_values)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', label = r'posterior $\mu \pm \sigma$ ', markersize =3, linewidth =1, capsize = 3)


ax1.set_xlabel(r'ozone volume mixing ratio ')
#multicolor_ylabel(ax1,('(Tangent)','Height in km'),('k', dataCol),axis='y')
ax1.set_ylabel('height in km')
handles, labels = ax1.get_legend_handles_labels()
#handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend()
#legend = ax1.legend(handles = [handles[-3], handles2[0], handles[0],handles[-2],handles[-1]])# loc='lower right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

ax1.set_ylim([height_values[0], height_values[-1]])

# ax2.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
# ax2.tick_params(colors = DatCol, axis = 'x')
# ax2.xaxis.set_ticks_position('top')
# ax2.xaxis.set_label_position('top')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.xaxis.set_label_position('bottom')
# ax1.spines[:].set_visible(False)

fig3.savefig('FirstRecRes.png')
plt.show()


##find affine map
FirstSamp = len(y)
relMapErrDat = 4
Results = np.random.multivariate_normal(MargInteg, CondVar,size=FirstSamp)
currMap = np.eye(len(y))
RealMap, relMapErr, LinDataY, NonLinDataY, testO3 = genDataFindandtestMap(currMap, tang_heights_lin, A_lin_dx,  height_values, gamma0, VMR_O3, Results, AscalConstKmToCm, A_lin, temp_values, pressure_values, ind, scalingConst, relMapErrDat, wvnmbr, S, E,g_doub_prime)
np.savetxt('RealMap.txt',RealMap, fmt = '%.30f', delimiter= '\t')

linTestDat = np.matmul(A,VMR_O3 * theta_scale_O3)
MapLinTestDat = np.matmul( RealMap @ A,VMR_O3 * theta_scale_O3)
nonLinTestDat = np.matmul( A/2 * nonLinA,VMR_O3 * theta_scale_O3)

binCol = 'C0'
postCol = 'C1'
priorCol = 'k'
#TrueCol = 'C2'

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), dpi = 300)

ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 3, markersize =15)

#line3 = ax1.errorbar(MargInteg,height_values[:,0],xerr = np.sqrt(np.diag(CondVar)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =3, linewidth =1, capsize = 3 )#, markerfacecolor = 'none'
#ax1.errorbar(MargInteg,height_values[:,0],  yerr = np.zeros(len(height_values)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', label = r'posterior $\mu \pm \sigma$ ', markersize =3, linewidth =1, capsize = 3)
ax1.plot(Results[0], height_values, markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=2,
         linewidth=0.5, label='posterior sample', alpha = 0.5)
for i in range(1,FirstSamp):
    ax1.plot(testO3[i], height_values, markeredgecolor=binCol, color=binCol, zorder=1, marker='.', markersize=4,
             linewidth=0.75, alpha = 0.5)

ax1.set_xlabel(r'ozone volume mixing ratio ')

ax1.set_ylabel('height in km')
handles, labels = ax1.get_legend_handles_labels()
ax1.legend()
ax1.set_ylim([height_values[0], height_values[-1]])
fig3.savefig('FirstTestRes.png')
plt.show()

##

DatCol =  'gray'

fig4, ax4 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), tight_layout = True, dpi = 300)
ax4.plot(linTestDat,tang_heights_lin, linestyle = 'dotted', marker = '*', label = r'linear $\bm{A}_L\bm{x}$', markersize = 18 , zorder = 0, color = DatCol )
relErr = np.linalg.norm( MapLinTestDat -  nonLinTestDat) / np.linalg.norm(MapLinTestDat) * 100
ax4.plot(MapLinTestDat,tang_heights_lin, linestyle = 'dotted', marker = '*', label = r'mappped $\bm{A}_L\bm{x}$' + f', rel. Err.: {relErr:.1f} \%', markersize = 7, zorder = 2, color ='k')
ax4.plot(nonLinTestDat,tang_heights_lin, linestyle = 'dotted', marker = 'o', label = r'non-linear $\bm{A_{NL}x}$', markersize = 10, zorder = 1, color = 'r')
ax4.legend()
ax4.set_ylabel('(tangent) height in km')
ax4.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
#ax4.xaxis.set_ticks_position('top')
#ax4.xaxis.set_label_position('top')
plt.savefig('SampMapAssesment.png')
plt.show()


##

A = RealMap @ np.copy(A)
ATA = A.T @ A
ATy = A.T @ y

minimum = optimize.fmin(MinLogMargPost, [gamma0,1/gamma0* 1/ np.mean(vari)/15], maxiter = 25)
gam0 = minimum[0]
lam0 = minimum[1]
print(minimum)


##
''' check taylor series in f(lambda) and g(lambda)
around lam0 from gmres = '''

#taylor series arounf lam_0

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


## do sampling again


startTime = time.time()
lambdas ,gammas, k = MHwG(number_samples, burnIn, lam0, gam0, f_0)
elapsed = time.time() - startTime
print('MTC Done in ' + str(elapsed) + ' s')
print('acceptance ratio: ' + str(k/(number_samples+burnIn)))

deltas = lambdas * gammas
np.savetxt('AffineSamples.txt', np.vstack((gammas[burnIn::], deltas[burnIn::], lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')



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
BinHist = BinHistStart
MargTime = time.time() - startTime
print('Post Mean in ' + str(MargTime) + ' s')
oldRelErr = 0
print(BinHistStart)

for BinHist in range(BinHistStart+1,100):

    lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density =True)
    gamHist, gamBinEdges = np.histogram(gammas, bins=BinHist, density=True)
    MargResults = np.zeros((BinHist,len(theta)))
    B_inv_Res = np.zeros((BinHist,len(theta)))
    VarB = np.zeros((BinHist, len(L), len(L)))
    gamInt = np.zeros(BinHist)
    meanGamInt = np.zeros(BinHist)
    IDiag = np.eye(len(SetB))
    startTime  = time.time()
    for p in range(BinHist):

        SetLambda =  lambBinEdges[p] + (lambBinEdges[p] + lambBinEdges[p+1])/2
        SetGamma = gamBinEdges[p] + (gamBinEdges[p] + gamBinEdges[p+1])/2
        SetB = ATA + SetLambda * L

        LowTri = np.linalg.cholesky(SetB)
        UpTri = LowTri.T
        B_inv_A_trans_y = lu_solve(LowTri, UpTri, ATy[0::, 0])

        MargResults[p, :] = B_inv_A_trans_y * lambHist[p]/ np.sum(lambHist)
        B_inv_Res[p, :] = B_inv_A_trans_y

        B_inv = np.zeros(SetB.shape)
        # startTime = time.time()
        LowTri = np.linalg.cholesky(SetB)
        UpTri = LowTri.T
        for j in range(len(B)):
            B_inv[:, j] = lu_solve(LowTri, UpTri, IDiag[:, j])

        VarB[p] = B_inv * lambHist[p]/ np.sum(lambHist)
        gamInt[p] = 1 / SetGamma * gamHist[p]/ np.sum(gamHist)
        meanGamInt[p] = SetGamma * gamHist[p]/ np.sum(gamHist)

    newMargInteg = np.sum(MargResults,0)  / theta_scale_O3
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

MargInteg= np.copy(newMargInteg)

#CondVar = scy.integrate.trapezoid(gamInt) * scy.integrate.trapezoid(VarB.T) / (theta_scale_O3) ** 2
CondVar =np.sum(gamInt) * np.sum(VarB,0)  / (theta_scale_O3) ** 2

MargTime = time.time() - startTime
print('Post Mean in ' + str(MargTime) + ' s')
relErrO3 = np.linalg.norm(MargInteg -VMR_O3[:,0]) / np.linalg.norm(MargInteg) * 100
print('rel Error to ground Truth:' + str(relErrO3))

print('MTC Done in ' + str(elapsed) + ' s')

##
#BinHist = 30#n_bins
lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density= True)
gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)
fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction), dpi = 300 )#, dpi = dpi)

axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], color = 'k', zorder = 0,width = np.diff(gamBinEdges)[0])#10)


axs[0].set_xlabel(r'the noise precision $\gamma$')


axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = 'k', zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_xlabel(r'$\lambda =\delta / \gamma$, the regularization parameter')
plt.savefig('SecHistoPlot.png')
plt.show()

##
TrueCol = [50/255,220/255, 0/255]#'#02ab2e'

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), dpi = 300)


#ax1.scatter(VMR_O3,height_values,marker = 'o', facecolor = 'None', color = "#009E73", label = 'true profile', zorder=1, s =12)#,linewidth = 5)
ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 3, markersize =15)

#line3 = ax1.plot(MargInteg,height_values[:,0], markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.',  label = r'$\text{E}_{\mathbf{x},\mathbf{\theta}|\mathbf{y}} [\mathbf{x}]$', markersize =3, linewidth =1)#, markerfacecolor = 'none'
line3 = ax1.errorbar(MargInteg,height_values[:,0],xerr =3* np.sqrt(np.diag(CondVar)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =3, linewidth =1, capsize = 3)#, markerfacecolor = 'none'
ax1.errorbar(MargInteg,height_values[:,0],  yerr = np.zeros(len(height_values)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', label = r'posterior $\mu \pm 3\sigma$ ', markersize =3, linewidth =1, capsize = 3)


ax1.set_xlabel(r'ozone volume mixing ratio ')
#multicolor_ylabel(ax1,('(Tangent)','Height in km'),('k', dataCol),axis='y')
ax1.set_ylabel('height in km')
handles, labels = ax1.get_legend_handles_labels()
#handles2, labels2 = ax2.get_legend_handles_labels()

#legend = ax1.legend(handles = [handles[-3], handles2[0], handles[0],handles[-2],handles[-1]])# loc='lower right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

ax1.set_ylim([height_values[0], height_values[-1]])
ax1.legend()
# ax2.set_xlabel(r'spectral radiance in $\frac{\text{W} \text{cm}}{\text{m}^2 \text{sr}} $',labelpad=10)# color =dataCol,
# ax2.tick_params(colors = DatCol, axis = 'x')
# ax2.xaxis.set_ticks_position('top')
# ax2.xaxis.set_label_position('top')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.xaxis.set_label_position('bottom')
# ax1.spines[:].set_visible(False)

fig3.savefig('SecRecRes.png')
plt.show()

## make nice scatter plot with trace


trace = [MinLogMargPost(np.array([lambdas[burnIn+ i],gammas[burnIn+ i]])) for i in range(number_samples)]


fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction), gridspec_kw={'height_ratios': [3, 1]} , dpi =300)

axs[0].scatter(gammas[burnIn:],lambdas[burnIn:], marker = '.', color = binCol, s = 0.1)
axs[0].set_xlabel(r'$\gamma$')
axs[0].set_ylabel(r'$\lambda$')

axs[1].plot(trace, color = 'k')
axs[1].set_ylabel(r'$\pi(\lambda, \gamma| \bm{y})$')
axs[1].set_xlabel('number of samples')
plt.savefig('ScatterplusHisto.png')
plt.show()
##
# dimMargO3 = 2
# gridSize = 25
# index = 'first'
# dir = '/Users/lennart/PycharmProjects/TTDecomposition/'
# margPDFO3 = np.zeros((dimMargO3, gridSize))
# univarGridO3 = np.zeros((dimMargO3, gridSize))
# for i in range(0, dimMargO3):
#     margPDFO3[i] =  np.loadtxt(dir + index +'margPDFMargO3' + str(i) + '.txt')
#     univarGridO3[i] = np.loadtxt(dir + index +'uniVarGridMargO3' + str(i) + '.txt')
#
# #Create 2D map
# TTMarg = np.zeros((gridSize,gridSize))
# for i in range(0, gridSize):
#     for j in range(0, gridSize):
#         TTMarg[i,j] = margPDFO3[0,i] * margPDFO3[1,j]
#
# #viridis = mpl.cm.get_cmap('viridis', 12)
# viridis = mpl.colormaps.get_cmap('viridis')
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction), gridspec_kw={'height_ratios': [3, 1]} , dpi =300)
# #im = axs[0].imshow(TTMarg, zorder = 0)
# #axs[0].pcolormesh(univarGridO3[0],univarGridO3[1],TTMarg)
# #axs[0].scatter(gammas[burnIn:],lambdas[burnIn:], marker = '.', color = binCol, s = 2)
# for i in range(0,len(gammas[burnIn:])):
# #for i in range(0, 10):
#     gamVal = np.interp(gammas[burnIn+i],univarGridO3[0],margPDFO3[0])
#     lamVal = np.interp(lambdas[burnIn + i], univarGridO3[1], margPDFO3[1])
#     colVal = (gamVal*lamVal)/np.max(TTMarg)
#     #print(colVal)
#     sc = axs[0].scatter(gammas[burnIn+i], lambdas[burnIn + i], marker='.', color=viridis(colVal), s=0.15)
# cbar = plt.colorbar(sc)
# label = cbar.ax.get_yticks()
# newlabel = [np.round(lab * np.max(TTMarg),3) for lab in label]
# cbar.ax.set_yticks(label)
# cbar.ax.set_yticklabels(newlabel)
# cbar.set_label(r'$\pi(\lambda, \gamma| \bm{y})$')
# #axs[0].scatter(gammas[burnIn:],lambdas[burnIn:], marker = '.', color = binCol, s = 2)
# axs[0].set_xlabel(r'$\gamma$')
# axs[0].set_ylabel(r'$\lambda$')
#
# axs[1].plot(trace, color = 'k')
# axs[1].set_ylabel(r'$\pi(\lambda, \gamma| \bm{y})$')
# axs[1].set_xlabel('number of samples')
# plt.savefig('ScatterplusHistoPlusTT.png')
# plt.show()

##
BinHist = 30#n_bins
lambHist, lambBinEdges = np.histogram(lambdas, bins= BinHist, density= True)
gamHist, gamBinEdges = np.histogram(gammas, bins= BinHist, density= True)

fig, axs = plt.subplots(2, 1,tight_layout=True,figsize=set_size(PgWidthPt, fraction=fraction), dpi = 300 )#, dpi = dpi)
#axs[0].set_xlim([0.1e-7, 5e-7])
#axs[0].bar(gamBinEdges[1::],gamHist*np.diff(gamBinEdges)[0], color = 'k', zorder = 0,width = np.diff(gamBinEdges)[0])#10)
axs[0].hist(gammas, color = 'k', zorder = 0, bins = BinHist)
axs[0].set_xlabel(r'the noise precision $\gamma$')
axs[1].hist(lambdas, color = 'k', zorder = 0, bins = BinHist)
#axs[1].bar(lambBinEdges[1::],lambHist*np.diff(lambBinEdges)[0], color = 'k', zorder = 0,width = np.diff(lambBinEdges)[0])#10)
axs[1].set_xlabel(r'$\lambda =\delta / \gamma$, the regularization parameter')
plt.savefig('FinalHistoPlot.png')
plt.show()


## regularization
'''L-curve refularoization
'''

lamLCurve = np.logspace(1,7,200)
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
    NormLCurve[i] = np.linalg.norm( np.matmul(A,x) - y[0::,0])
    xTLxCurve[i] = np.sqrt(np.matmul(np.matmul(x.T, L), x))


startTime  = time.time()
lamLCurveZoom = np.logspace(0,7,200)
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
np.savetxt('lam_opt.txt',[lam_opt], fmt = '%.15f')

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

regCol = 'C3'

#generate samples and calc norms
sampSize = 100
xTLxRes = np.zeros(sampSize)
NormRes = np.zeros(sampSize)

for i in range(0,sampSize):
    currSamp = np.random.multivariate_normal(MargInteg* theta_scale_O3,CondVar * theta_scale_O3 ** 2)

    xTLxRes[i] =  np.sqrt(np.matmul(np.matmul(currSamp.T, L), currSamp))
    NormRes[i] = np.linalg.norm( np.matmul(A,currSamp) - y[0::,0])

currX = MargInteg* theta_scale_O3
NormMargRes = np.linalg.norm( np.matmul(A,currX) - y[0::,0])
xTLxMargRes = np.sqrt(np.matmul(np.matmul(currX.T, L), currX))



fig, axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), dpi = 300)
axs.scatter(NormLCurve,xTLxCurve, zorder = 0, color =  DatCol, s = 3.5, marker ='s', label = 'reg. solution')
axs.scatter(NormRes, xTLxRes, color = ResCol, s = 3, marker = "+",label = r'posterior samples ')# ,mfc = 'black' , markeredgecolor='r',markersize=10,linestyle = 'None')
axs.scatter(NormMargRes, xTLxMargRes, color = MeanCol, marker = '.', s= 50, label = 'posterior mean',zorder=2)
axs.scatter(knee_point, kneedle.knee_y, color = regCol, marker = 'v',label = 'max. curvature', s= 50,zorder=1)

axs.set_xscale('log')
axs.set_yscale('log')
axs.set_ylabel(r'$ \sqrt{\bm{x}^T \bm{L}\bm{x}}$', style='italic')
axs.set_xlabel(r'$|| \bm{Ax} - \bm{y}||$')
handles, labels = axs.get_legend_handles_labels()
axs.legend()
#axs.legend(handles = [handles[0],handles[1],handles[2]],loc = 'upper right',  frameon =True)
plt.savefig('LCurvePhD.png')

plt.show()

print('bla')

#np.savetxt('RegSol.txt',x_opt /(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst), fmt = '%.15f', delimiter= '\t')

##
fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), dpi = 300)
#ax1.scatter(VMR_O3,height_values,marker = 'o', facecolor = 'None', color = "#009E73", label = 'true profile', zorder=1, s =12)#,linewidth = 5)
ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 3, markersize =15)
#line3 = ax1.plot(MargInteg,height_values[:,0], markeredgecolor =MeanCol, color = MeanCol ,zorder=3, marker = '.',  label = r'$\text{E}_{\mathbf{x},\mathbf{\theta}|\mathbf{y}} [\mathbf{x}]$', markersize =3, linewidth =1)#, markerfacecolor = 'none'
line3 = ax1.errorbar(MargInteg,height_values[:,0],xerr =3* np.sqrt(np.diag(CondVar)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', markersize =3, linewidth =1, capsize = 3)#, markerfacecolor = 'none'
ax1.errorbar(MargInteg,height_values[:,0],  yerr = np.zeros(len(height_values)), markeredgecolor ='k', color = 'k' ,zorder=3, marker = '.', label = r'posterior $\mu \pm 3\sigma$ ', markersize =3, linewidth =1, capsize = 3)
ax1.plot(x_opt / theta_scale_O3,height_values[:,0],marker = 'v',markerfacecolor = RegCol, color = RegCol , label = r'reg. solution', zorder=2 ,linewidth = 2, markersize =8)


ax1.set_xlabel(r'ozone volume mixing ratio ')

ax1.set_ylabel('height in km')
handles, labels = ax1.get_legend_handles_labels()

ax1.set_ylim([height_values[0], height_values[-1]])
ax1.legend()


fig3.savefig('SecRecResinclReg.png')
plt.show()


##
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
    if np.linalg.norm(ATy[0::, 0]- CheckB_inv_ATy)/np.linalg.norm(ATy[0::, 0])<=1e-3:
        f_func[j] = f(ATy, y, B_inv_A_trans_y)
    else:
        print('tol not reached')
        f_func[j] = np.nan

    g_func[j] = g(A, L, lam[j])


''' check taylor series in f(lambda) and g(lambda)
around lam0 from gmres = '''

def g_tayl(delta_lam, g_0, trace_B_inv_L_1, trace_B_inv_L_2, trace_B_inv_L_3, trace_B_inv_L_4, trace_B_inv_L_5, trace_B_inv_L_6):

    return g_0 + trace_B_inv_L_1 * delta_lam + trace_B_inv_L_2 * delta_lam**2 + trace_B_inv_L_3 * delta_lam**3 + trace_B_inv_L_4 * delta_lam**4 + trace_B_inv_L_5 * delta_lam**5 + trace_B_inv_L_6 * delta_lam**6



def f_tayl( delta_lam, f_0, f_1, f_2, f_3, f_4, f_5, f_6):
    """calculate taylor series for """

    return f_0 + f_1 * delta_lam + f_2 * delta_lam**2 + f_3 * delta_lam**3 + f_4 * delta_lam**4 + f_5 * delta_lam**5 + f_6 * delta_lam**6


##


f_0 = f(ATy, y, B_inv_A_trans_y0)
g_0 = g(A, L, minimum[1])
fCol = [0, 144/255, 178/255]
gCol = [230/255, 159/255, 0]
gmresCol = [204/255, 121/255, 167/255]
#lam0 = minimum[1]
# find min ind and max ind for lambda
minInd = np.argmin(abs(lam - lambBinEdges[0]))
maxInd = np.argmin(abs(lam - lambBinEdges[-1]))
delta_lam = lambBinEdges - minimum[1]
taylorG = g_tayl(delta_lam,g_0, g_0_1, g_0_2, g_0_3, 0, 0,0)
taylorF = f_tayl(delta_lam, f_0, f_0_1, f_0_2, f_0_3, 0, 0, 0)

fig,axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction),  dpi = 300,tight_layout = True)

axs.plot(lam,f_func, color = fCol, zorder = 2, linestyle=  'dotted')
axs.set_yscale('log')
axs.set_xlabel('$\lambda$')
axs.set_ylabel('$f(\lambda)$')#, color = fCol)
axs.tick_params(axis = 'y',  colors=fCol, which = 'both')

ax2 = axs.twinx() # ax1 and ax2 share y-axis
ax2.plot(lam,g_func, color = gCol, zorder = 0, linestyle=  'dashed')
#ax2.scatter(minimum[1],g(A, L, minimum[1]), color = gmresCol, zorder=0, marker = 's')

#ax2.scatter(np.mean(lambdas),g(A, L, np.mean(lambdas) ), color = MTCCol, zorder=5)
#ax2.scatter(lamPyT,g(A, L, lamPyT) , color = pyTCol, zorder=6, marker = 'D')
#ax2.annotate('T-Walk $\lambda$ sample mean',(lamPyT+1e6,g(A_lin, L, lamPyT) +50), color = 'k')
ax2.set_ylabel('$g(\lambda)$')#,color = gCol)
ax2.tick_params(axis = 'y', colors= gCol)
axs.set_xscale('log')
axs.plot(lambBinEdges,taylorF , color = 'k', linewidth = 1, zorder = 2, label = 'Taylor series' )
ax2.plot(lambBinEdges, taylorG , color = 'k', linewidth = 1, zorder = 2)
ax2.axvline( minimum[1], color = gmresCol, ymin=0, ymax=0.5, zorder = 0)
ax2.axvline( minimum[1], color = gmresCol, ymin=0.95, ymax=1, zorder = 0)


#mark_inset(axs, axins, loc1=3, loc2=4, fc="none", ec="0.5")

#axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
#axs.spines['left'].set_color(fCol)
axs.spines['left'].set_color('k')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color('k')
#ax2.spines['right'].set_color(gCol)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)



axins = axs.inset_axes([0.05,0.5,0.4,0.45])
axins.plot(lam,f_func, color = fCol, zorder=0, linestyle=  'dotted', linewidth = 3, label = '$f(\lambda)$')
axins.axvline( minimum[1], color = gmresCol, label = r'$\pi(\lambda_0|\bm{y}, \gamma)$')

axins.plot(lambBinEdges,taylorF , color = 'k', linewidth = 1, zorder = 1, label = 'Taylor series' )

axins.set_ylim(0.95 * taylorF[0],1.5 * taylorF[-1])
axins.set_xlabel('$\lambda$')
axins.set_yscale('log')
axins.set_xscale('log')

axins.tick_params(axis='x', which='both',  bottom=False, labelbottom=False)

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
axin2.set_ylim(0.8 * taylorG[0],1.05 * taylorG[-1])
axin2.set_xlim(min(lambBinEdges),max(lambBinEdges))
axin2.set_xscale('log')
lines2, lab2 = axin2.get_legend_handles_labels()
lines, lab0 = axins.get_legend_handles_labels()
axins.set_xlim(min(lambBinEdges),max(lambBinEdges))
axs.legend(np.append(lines2,lines),np.append(lab2,lab0), loc = 'lower right')

fig.savefig('f_and_g_phd.png', bbox_inches='tight')
plt.show()


## prior for Ozone

test = 20


fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), dpi = 300)

ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 3, markersize =15)
for i in range(0,test):
    delt = np.random.gamma(shape = 1, scale = 1e10)
    priorTest = np.random.multivariate_normal(np.zeros(len(L)), delt * L)
    ax1.plot( priorTest ,height_values , markeredgecolor =binCol , color = binCol ,zorder=2, marker = '.', markersize =4, linewidth =0.75, label = 'prior sample', alpha = 0.25)

ax1.set_xlabel(r'ozone volume mixing ratio ')
ax1.set_ylabel('height in km')
handles, labels = ax1.get_legend_handles_labels()

ax1.set_ylim([height_values[0], height_values[-1]])
ax1.legend(handles[:2], labels[:2])


fig3.savefig('OzonePrior.png')
plt.show()

