import numpy as np
import matplotlib.pyplot as plt
VMR_O3 = np.loadtxt('VMR_O3.txt')
height_values = np.loadtxt('height_values.txt')
SecO3Mean = np.loadtxt('SecO3Mean.txt')
SecO3Var= np.loadtxt('SecO3Var.txt')
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.gaussian_process.kernels import RBF

from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

# Combine kernels
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

##

y = VMR_O3
X = height_values.reshape((len(height_values),1))
X_train, y_train = X[::2], y[::2]

kernel = 1 * RBF(length_scale=1.0)
kernel = kernel = 1e-6 * RBF(length_scale=1e-5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
    noise_level=1e-10, noise_level_bounds=(1e-11, 1e-7)
)

kernel = Matern() + WhiteKernel()

gp_opt = GaussianProcessRegressor(kernel=kernel, alpha = 0.0, n_restarts_optimizer=30)
gp_opt.fit(X_train, y_train)
mean_prediction, std_prediction = gp_opt.predict(X, return_std=True)

plt.plot( y, X, linestyle="dotted")
plt.scatter( y_train,X_train, label="Observations")
plt.plot(mean_prediction,X,  label="Mean prediction")

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on noise-free dataset")

plt.show(block = True)


##
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
param_grid = [{
    "alpha":  [1e-2, 1e-3],
    "kernel": [Matern(l) for l in np.logspace(-5, 10, 20)]
}, {
    "alpha":  [1e-2, 1e-3],
    "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 20)]
}, {
        "alpha": [1e-2, 1e-3],
        "kernel": [WhiteKernel(sigma_0) for sigma_0 in np.logspace(-1, 1, 20)]
    }]

# scores for regression
scores = ['explained_variance', 'r2']

gp = GaussianProcessRegressor()
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(estimator=gp, param_grid=param_grid, cv=4,
                       scoring='%s' % score)
    clf.fit(X.reshape(-1, 1), y)
    print(clf.best_params_)


##
kernel = Matern() + WhiteKernel()
kernel =RBF(length_scale=0.546)
kernel = Matern(length_scale=0.546, nu=1.5)
gp_opt = GaussianProcessRegressor(kernel=kernel, alpha = 0.0, n_restarts_optimizer=30)
gp_opt.fit(X_train, y_train)
mean_prediction, std_prediction = gp_opt.predict(X, return_std=True)

plt.plot( y, X, linestyle="dotted")
plt.scatter( y_train,X_train, label="Observations")
plt.plot(mean_prediction,X,  label="Mean prediction")

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("Gaussian process regression on noise-free dataset")

plt.show(block = True)

from scipy.special import gamma, kv

def matern_kernel(x1, x2, lengthscale, nu, sigma_sq=1.0):
    r = np.linalg.norm(x1 - x2)
    if r == 0:  # Handle the case where points are identical
        return sigma_sq

    arg = np.sqrt(2 * nu) * r / lengthscale
    term1 = sigma_sq * (2**(1 - nu)) / gamma(nu)
    term2 = (arg**nu) * kv(nu, arg)
    return term1 * term2
##
MatCov = np.zeros(( len(height_values), len(height_values)))
for i in range(0, len(height_values)):
    for j in range(0, len(height_values)):
        MatCov[i,j] = matern_kernel(height_values[i], height_values[j], 0.546, nu=1.5, sigma_sq=1e-10)

Tests = np.random.multivariate_normal(mean=np.zeros(VMR_O3.shape), cov=MatCov, size=5)

fig3, ax1 = plt.subplots()
#ax1.scatter(VMR_O3,height_values)
for i in range(0,len(Tests)):
    # np.random.uniform(1e-18, 0.7)
    # np.random.uniform(0.4,0.7)
    #Test = np.random.multivariate_normal(mean=VMR_O3, cov=MatCov, size=len(height_values))
    ax1.plot(Tests[i],height_values)
plt.show(block = True)
