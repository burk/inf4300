import numpy as np
from collections import defaultdict
from functools import partial
import matplotlib.pyplot as plt
import operator
from matplotlib.patches import Ellipse

class GaussianClassifier(object):

    def __init__(self, covar_type='single'):
        self.covar_type = covar_type

    # Estimate one single covariance matrix
    def sigma_single(self, X, Y):
        self.sigma = {}
        sig = np.zeros((X.shape[1], X.shape[1]),
                dtype=np.float64)

        for i in xrange(X.shape[0]):
            sig += np.outer(X[i] - self.mean[Y[i]], X[i] - self.mean[Y[i]])

        sig /= X.shape[0]

        for lab in self.labels:
            self.sigma[lab] = sig

    # Estimate one covariance matrix for each class
    def sigma_multiple(self, X, Y):
        self.sigma = {}

        for lab in self.labels:
            sig = np.zeros((X.shape[1], X.shape[1]),
                    dtype=np.float64)
            num = 0
            for feat in X[Y == lab, :]:
                sig += np.outer(feat - self.mean[lab], feat -
                        self.mean[lab])
                num += 1

            sig /= float(num)
            self.sigma[lab] = sig

    # Train the classifier
    def train(self, X, Y):
        self.labels = np.unique(Y)
        self.mean = defaultdict(lambda: np.array([0.0] * X.shape[1]))
        self.num  = defaultdict(int)
        self.prior = defaultdict(float)

        totnum = 0
        for r in xrange(X.shape[0]):
            self.mean[Y[r]] += X[r]
            self.num[Y[r]]  += 1
            totnum += 1

        for lab in self.labels:
            self.mean[lab] /= self.num[lab]
            self.prior[lab] = float(self.num[lab]) / totnum

        if self.covar_type == 'single':
            self.sigma_single(X, Y)
        else:
            self.sigma_multiple(X, Y)

    # Predict Y given X
    def predict(self, X):
        Y = np.zeros((X.shape[0]))

        w = {}
        w0 = {}
        for lab in self.labels:
            w[lab] = np.linalg.solve(self.sigma[lab], self.mean[lab])
            w0[lab] = -0.5 * self.mean[lab].dot(np.linalg.solve(
                        self.sigma[lab],
                        self.mean[lab])) \
                        - 0.5 * \
                        np.log(np.abs(
                            np.linalg.det(self.sigma[lab])
                        )) + \
                        np.log(self.prior[lab])


        for r in xrange(X.shape[0]):
            prob = {}
            for cl in self.labels:
                x = X[r]

                prob[cl] = x.dot(-0.5 * \
                        np.linalg.solve(self.sigma[cl], x)) + \
                        w[cl].dot(x) + w0[cl]
                if np.isnan(prob[cl]) and r % 1000 == 0:
                    prob[cl] = -np.inf

            # Get the class with the highest discriminant
            Y[r] = max(prob.iteritems(), key=operator.itemgetter(1))[0]

        return Y

    def test(self, X, Y):
        Ypred = self.predict(X)

        return float((Y == Ypred).sum()) / Y.shape[0]

def main():
    meanx1 = np.random.rand(1)[0]*2*2
    meanx2 = np.random.rand(1)[0]*2*2
    meany1 = np.random.rand(1)[0]*2*2
    meany2 = np.random.rand(1)[0]*2*2

    varx1 = np.random.rand(1)[0]*.2
    varx2 = np.random.rand(1)[0]*.2
    vary1 = np.random.rand(1)[0]*.2
    vary2 = np.random.rand(1)[0]*.2

    X1 = np.random.multivariate_normal((meanx1, meany1),
            [[varx1,0],[0,vary1]], 50)
    Y1 = np.array([0] * X1.shape[0])
    X2 = np.random.multivariate_normal((meanx2, meany2),
            [[varx2,0],[0,vary2]], 50)
    Y2 = np.array([1] * X1.shape[0])

    X = np.vstack((X1, X2))
    Y = np.concatenate((Y1, Y2))
    print Y.shape
    print Y
    print X.shape

    fit = GaussianClassifier(covar_type="multi")
    fit.train(X, Y)

    Ypred = fit.predict(X)
    print Ypred
    print "TRAINING ERROR:", (Ypred == Y).sum() / float(Y.shape[0])
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.hot)
    plt.subplot(122)
    plt.scatter(X[:,0], X[:,1], c=Ypred, cmap=plt.cm.hot)
    plot_cov_ellipse(fit.sigma[0], fit.mean[0], color=(0,0,0))
    plot_cov_ellipse(fit.sigma[1], fit.mean[1], color=(0,1,0))
    plt.show()

# Ellipse plotting from
# http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, color=(0,0,0), **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ellip.set_alpha(0.2)
    ellip.set_facecolor(color)

    ax.add_artist(ellip)
    return ellip

if __name__ == "__main__":
    main()

