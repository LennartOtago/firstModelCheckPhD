from minimax_tilting_sampler import *
from RunFunc import set_size
import matplotlib.pyplot as plt
import matplotlib as mpl

dir = '/home/lennartgolks/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/firstModelCheckPhD/'
dir = '/Users/lennart/PycharmProjects/TTDecomposition/'
#dir = '/home/lennartgolks/PycharmProjects/TTDecomposition/'

L = np.loadtxt(dir +'GraphLaplacian.txt')

VMR_O3 = np.loadtxt(dir +'VMR_O3.txt')
VMR_O3 = VMR_O3.reshape((len(VMR_O3), 1))

height_values = np.loadtxt(dir +'height_values.txt')
height_values = height_values.reshape((len(height_values),1))

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
TrueCol = 'green' # "#004D40" #'k'
binCol = 'C0'

d = len(VMR_O3)  # dimensions

# random mu and cov
mu = np.zeros(d)#np.random.rand(d)
# cov = 0.5 - np.random.rand(d ** 2).reshape((d, d))
# cov = np.triu(cov)
# cov += cov.T - np.diag(cov.diagonal())
# cov = np.dot(cov, cov)

# constraints
lb = np.zeros_like(mu)
ub = np.ones_like(mu) * np.inf

# create truncated normal and sample from it
n_samples = 1
# tmvn = TruncatedMVN(mu, cov, lb, ub)
# samples = tmvn.sample(n_samples)
## prior for Ozone

test = 50
priorTest = np.zeros((test,d))

fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), dpi = 300)
#ax1.hist(samples[0])

ax1.plot(VMR_O3,height_values[:,0],marker = 'o',markerfacecolor = TrueCol, color = TrueCol , label = r'true $\bm{x}$', zorder=0 ,linewidth = 3, markersize =15)
for i in range(0,test):
    delt = np.random.gamma(shape=1, scale=1e10)
    cov = delt * L
    tmvn = TruncatedMVN(mu, cov, lb, ub)
    priorTest[i] = tmvn.sample(n_samples)[:,0]
    # priorTest = np.random.multivariate_normal(np.zeros(len(L)), delt * L)
    # while any(priorTest < 0) or  delt < 0:
    #     delt = np.random.gamma(shape=1, scale=1e10)
    #     priorTest = np.random.multivariate_normal(np.zeros(len(L)), delt * L)
    print(i)
    ax1.plot( priorTest[i] ,height_values , markeredgecolor =binCol , color = binCol ,zorder=2, marker = '.', markersize =4, linewidth =0.75, label = 'prior sample', alpha = 0.25)

ax1.set_xlabel(r'ozone volume mixing ratio ')
ax1.set_ylabel('height in km')
handles, labels = ax1.get_legend_handles_labels()

ax1.set_ylim([height_values[0], height_values[-1]])
ax1.legend(handles[:2], labels[:2])

plt.show()
#fig3.savefig('OzonePrior.png')
##
fig3, ax1 = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction), dpi = 300)
ax1.hist(priorTest.flatten())
plt.show()
