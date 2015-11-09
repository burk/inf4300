
# coding: utf-8

get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.io
import sys

sys.path.append("../")

import glcm
import gaussian

# Load and inspect data

mosaic1_train = scipy.io.loadmat("newdata/mosaic1_train.mat")['mosaic1_train']

plt.hist(mosaic1_train.flatten(), 70)
plt.show()

plt.figure(1, figsize=(10,10))
plt.imshow(mosaic1_train, cmap=plt.cm.gray)
plt.show()

t1p0p1 = scipy.io.loadmat("newdata/texture1dx0dy1.mat")['texture1dx0dy1']
t1p1p0 = scipy.io.loadmat("newdata/texture1dx1dy0.mat")['texture1dx1dy0']
t1p1m1 = scipy.io.loadmat("newdata/texture1dx1dymin1.mat")['texture1dx1dymin1']
t1m1p1 = scipy.io.loadmat("newdata/texture1dxmin1dy1.mat")['texture1dxmin1dy1']

t2p0p1 = scipy.io.loadmat("newdata/texture2dx0dy1.mat")['texture2dx0dy1']
t2p1p0 = scipy.io.loadmat("newdata/texture2dx1dy0.mat")['texture2dx1dy0']
t2p1m1 = scipy.io.loadmat("newdata/texture2dx1dymin1.mat")['texture2dx1dymin1']
t2m1p1 = scipy.io.loadmat("newdata/texture2dxmin1dy1.mat")['texture2dxmin1dy1']

t3p0p1 = scipy.io.loadmat("newdata/texture3dx0dy1.mat")['texture3dx0dy1']
t3p1p0 = scipy.io.loadmat("newdata/texture3dx1dy0.mat")['texture3dx1dy0']
t3p1m1 = scipy.io.loadmat("newdata/texture3dx1dymin1.mat")['texture3dx1dymin1']
t3m1p1 = scipy.io.loadmat("newdata/texture3dxmin1dy1.mat")['texture3dxmin1dy1']

t4p0p1 = scipy.io.loadmat("newdata/texture4dx0dy1.mat")['texture4dx0dy1']
t4p1p0 = scipy.io.loadmat("newdata/texture4dx1dy0.mat")['texture4dx1dy0']
t4p1m1 = scipy.io.loadmat("newdata/texture4dx1dymin1.mat")['texture4dx1dymin1']
t4m1p1 = scipy.io.loadmat("newdata/texture4dxmin1dy1.mat")['texture4dxmin1dy1']

# Visualize GLCM matrices

arr = [t1p0p1, t1p1p0, t1p1m1, t1m1p1, t2p0p1, t2p1p0, t2p1m1, t2m1p1,
        t3p0p1, t3p1p0, t3p1m1, t3m1p1, t4p0p1, t4p1p0, t4p1m1, t4m1p1]
tit = ["(+0, +1)", "(+1, +0)", "(+1, -1)", "(-1, +1)"]
plt.figure(1, figsize=(6, 6))

for i, f in enumerate(arr):
    plt.subplot(4, 4, i+1)
    plt.imshow(f, interpolation="none")
    if i < 4:
        plt.title(tit[i])
plt.show()

# Add padding
train1_pad = np.pad(mosaic1_train, 15, mode='reflect')

# Calculate GLCM matrices
g1 = glcm.fast_glcm(train1_pad, size=31, stride=1, levels=16, dx=1, dy=0)
g2 = glcm.fast_glcm(train1_pad, size=31, stride=1, levels=16, dx=0, dy=1)

# Calculate selected quadrant features
quads = [(0, 0, 4, 4), (0, 4, 4, 8), (4, 4, 8, 8), (0, 8, 8, 16), (8, 8, 16, 16)]
tit = ["Q_11", "Q_12", "Q_14", "Q_2", "Q_4"]
plt.figure(1, figsize=(20,20))
i = 0
for quad, t in zip(quads, tit):
    plt.subplot(1, len(quads), i+1)
    plt.imshow(glcm.quadrant(g1, quad))
    plt.title(t)
    i += 1

plt.show()

quads = [(0, 0, 4, 4), (0, 4, 4, 8), (4, 4, 8, 8), (0, 8, 8, 16), (8, 8, 16, 16)]
tit = ["Q_11", "Q_12", "Q_14", "Q_2", "Q_4"]
plt.figure(1, figsize=(20,20))
i = 0
for quad, t in zip(quads, tit):
    plt.subplot(1, len(quads), i+1)
    plt.imshow(glcm.quadrant(g2, quad))
    plt.title(t)
    i += 1

plt.show()

# Load training mask
mask = scipy.io.loadmat("newdata/training_mask.mat")['training_mask']

plt.imshow(mask)
plt.show()

# Calculate and structure training features X for Gaussian classifier
feats = [g1, g2]
quads = [(0, 0, 4, 4), (0, 4, 4, 8), (4, 4, 8, 8), (0, 8, 8, 16), (8, 8, 16, 16)]

X = []
Y = []
for lab in [1,2,3,4]:
    features = []
    for gin in [0, 1]:
        g = feats[gin]
        for quad in quads:
            m = glcm.flat_quadrant(g[mask == lab], quad).flatten()
            features.append(m)
            
    features = np.array(features)
    X.append(features.transpose((1,0)))
    Y.append(np.array([lab] * features.shape[1]))

X = np.vstack(X)
Y = np.concatenate(Y)


# Calculate mean and standard deviation
mean_sub = X.mean(0)
std_div  = X.std(0)

X -= mean_sub
X /= std_div

print X.shape
print Y.shape

# Fit the Gaussian classifier using one covariance matrix per class
fit = gaussian.GaussianClassifier(covar_type='multiple')
fit.train(X, Y)

# Fit a Gaussian classifier with only one single covariance matrix
fit_single = gaussian.GaussianClassifier(covar_type='single')
fit_single.train(X, Y)

# Predict the training data to obtain the training error
Ytrain = fit.predict(X)

Ytrain_single = fit_single.predict(X)

print "Training accuracy:", ((Ytrain == Y).sum() /
        float(Ytrain.shape[0]))
print "Training accuracy (single):", ((Ytrain_single == Y).sum() /
        float(Ytrain_single.shape[0]))


# Plot the features against each other
import matplotlib.gridspec as gridspec

fig = plt.figure(1, figsize=(20,20), tight_layout=False)
gs1 = gridspec.GridSpec(10, 10)

nfeat = 10
stride = 1000

gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes.

for f1 in xrange(0, nfeat):
    for f2 in xrange(0, nfeat):
        ax = fig.add_subplot(gs1[f1 * nfeat + f2])
        ax.tick_params(
            axis='both',       # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            left='off',
            right='off',
            labelbottom='off', # labels along the bottom edge are off
            labelleft='off')
        plt.scatter(X[::stride,f1], X[::stride,f2],
                c=plt.cm.rainbow(Y[::stride]/4.0))

        for lab in fit.labels:
            gaussian.plot_cov_ellipse(fit.sigma[lab][[[f1], [f2]],[f1, f2]],
                                      fit.mean[lab][[f1, f2]],
                                      ax=ax,
                                      color=plt.cm.rainbow(lab/4.0))
plt.show()

# Load the testing images
mosaic2_test = scipy.io.loadmat("newdata/mosaic2_test.mat")['mosaic2_test']
mosaic3_test = scipy.io.loadmat("newdata/mosaic3_test.mat")['mosaic3_test']

test2_pad = np.pad(mosaic2_test, 15, mode='reflect')
test3_pad = np.pad(mosaic3_test, 15, mode='reflect')

t21 = glcm.fast_glcm(test2_pad, size=31, stride=1, levels=16, dx=1, dy=0)
t22 = glcm.fast_glcm(test2_pad, size=31, stride=1, levels=16, dx=0, dy=1)
t31 = glcm.fast_glcm(test3_pad, size=31, stride=1, levels=16, dx=1, dy=0)
t32 = glcm.fast_glcm(test3_pad, size=31, stride=1, levels=16, dx=0, dy=1)

# Caculate the quadrant features given the two GLCM images
def calc_features(f1, f2):
    res = []
    feats = [f1, f2]
    quads = [(0, 0, 4, 4), (0, 4, 4, 8), (4, 4, 8, 8), (0, 8, 8, 16), (8, 8, 16, 16)]

    features = []
    for gin in [0, 1]:
        g = feats[gin]
        for quad in quads:
            m = glcm.quadrant(g, quad).flatten()
            features.append(m)
            
    features = np.array(features)
    res.append(features.transpose((1,0)))

    res = np.vstack(res)
    return res

# Test image 2
Xtest2 = calc_features(t21, t22)

Xtest2 -= mean_sub
Xtest2 /= std_div

Ypred2 = fit.predict(Xtest2)

# Plot the test images
plt.imshow(mosaic2_test, cmap=plt.cm.gray)
plt.show()

plt.imshow(mosaic3_test, cmap=plt.cm.gray)
plt.show()

# Plot the test image 2 predictions
plt.imshow(np.array(Ypred2).reshape((512, 512)))
plt.show()

# Calculate features for test image 3
Xtest3 = calc_features(t31, t32)

Xtest3 -= mean_sub
Xtest3 /= std_div

Ypred3 = fit.predict(Xtest3)

# Show the predictions for test image 3
plt.imshow(np.array(Ypred3).reshape((512, 512)))
plt.show()

# Calculate the confusion matrix given the predictions for a 512x512
# image
def confusion(Y):
    true_labels = np.zeros((512, 512))
    true_labels[0:256, 0:256] = 1
    true_labels[0:256, 256:]  = 2
    true_labels[256:, 0:256]  = 3
    true_labels[256:, 256:]   = 4
    true_labels = true_labels.flatten()

    confusion = np.zeros((5, 5))
    for true in xrange(1, 5):
        for est in xrange(1, 5):
            confusion[true-1, est-1] = (Y[true_labels == true] == est).sum()
    for i in xrange(4):
        confusion[i,-1] = confusion[i,:].sum()
        confusion[-1,i] = confusion[:,i].sum()
    confusion[4, 4] = confusion[-1].sum()
    return confusion


conf = confusion(Ypred2)
print (conf).astype(int)

conf = confusion(Ypred3)
print (conf).astype(int)

mosaic3_test_filt = cv2.medianBlur(mosaic3_test, 3)
test3_pad_filt = np.pad(mosaic3_test_filt, 15, mode='reflect')

# Try prediction on a median filtered image
plt.figure(1, figsize=(10, 10))
plt.imshow(mosaic3_test_filt, cmap=plt.cm.gray)
plt.show()

t31_filt = glcm.fast_glcm(test3_pad_filt, size=31, stride=1, levels=16, dx=1, dy=0)
t32_filt = glcm.fast_glcm(test3_pad_filt, size=31, stride=1, levels=16, dx=0, dy=1)

Xtest3_filt = calc_features(t31_filt, t32_filt)

Xtest3_filt -= mean_sub
Xtest3_filt /= std_div

Ypred3_filt = fit.predict(Xtest3_filt)

plt.imshow(np.array(Ypred3_filt).reshape((512, 512)))
plt.show()

