from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
plt.rcParams.update({'font.size':  10,
                     'text.usetex': True,
                     'font.family' : 'serif',
                     'font.serif'  : 'cm',
                     'text.latex.preamble': r'\usepackage{bm, amsmath}'})
dimMargO3 = 2
gridSize = 100
margPDFO3 = np.zeros((dimMargO3, gridSize))
univarGridO3 = np.zeros((dimMargO3, gridSize))

for i in range(0, dimMargO3):
    margPDFO3[i] =  np.loadtxt('margPDFMargO3' + str(i) + '.txt')
    univarGridO3[i] = np.loadtxt('uniVarGridMargO3' + str(i) + '.txt')

#Create 2D map
TTMarg = np.zeros((gridSize,gridSize))
for i in range(0, gridSize):
    for j in range(0, gridSize):
        TTMarg[i,j] = margPDFO3[0,i] * margPDFO3[1,j]


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi = 300,tight_layout=True)
PlotX, PlotY = np.meshgrid(univarGridO3[0], univarGridO3[1])
# Plot the surface.
surf = ax.plot_surface(PlotX, PlotY,  TTMarg, cmap=cm.cool,
                       linewidth=0, antialiased=False)

ax.set_ylabel(r'$\pi(\lambda|\bm{y})$')
ax.set_xlabel(r'$\pi(\gamma|\bm{y})$')

# # Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# # A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

#ax.tick_params(axis='x', which='both',  bottom=False, labelbottom=False)
#ax.tick_params(axis='y', which='both',  left=False, labelleft=False)
#ax.tick_params(axis='z',  left=False, bottom=False, right=False, top=False, labeltop=False)
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])
plt.savefig('PosterMargTT.png')
plt.show()