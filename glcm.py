import numpy as np
import cv2
import matplotlib.pyplot as plt

def fast_glcm(img, size=15, stride=1, levels=5, dx=1, dy=0):

    h, w = img.shape

    res = np.zeros((h-size+1, w-size+1, levels, levels), dtype=np.float64)

    if stride != 1:
        raise ValueError('stride not equal to 1')

    pad = (size - 1) / 2
    per_level = 255 / levels

    for r in xrange(0, h, stride):
        if r + dy >= h or r + dy < 0:
            continue
        print "At row {}".format(r)
        for c in xrange(0, w, stride):
            if c + dx >= w or c + dx < 0:
                continue

            cur = min(img[r,c] / per_level, levels-1)
            nex = min(img[r+dy,c+dx] / per_level, levels-1)

            start = max(pad, r - pad)
            end   = min(h - pad, r + pad + 1)
            for rg in xrange(start, end):
                if abs(rg - r) > pad:
                    continue
                start = max(pad, c - pad)
                end   = min(w - pad, c + pad + 1)
                for cg in xrange(start, end):
                    if abs(cg - c) > pad:
                        continue
                    res[rg - pad, cg - pad, cur, nex] += 1

    for r in xrange(res.shape[0]):
        for c in xrange(res.shape[1]):
            res[r,c] =  (res[r,c] + res[r,c].T) / ((h - dy) * (w - dx) * 2)

    return res

def glcm(img, size=15, stride=7, levels=5, dx=1, dy=0):

    h, w = img.shape

    res = np.zeros((h-size+1, w-size+1, levels, levels), dtype=np.float64)

    for r in xrange(0, h, stride):
        if r + size >= h:
            continue
        print "At row {}".format(r)
        for c in xrange(0, w, stride):
            if c + size >= w:
                continue
            g = patch_glcm(img[r:(r+size), c:(c+size)], levels=levels,
                    dx=dx, dy=dy)
            res[r:(r+size), c:(c+size), :, :] = g

    return res

def patch_glcm(patch, levels=5, dx=1, dy=0):
    texel = np.zeros((levels, levels), dtype=np.float64)

    # Assume patch is [0, 255]
    per_level = 255 / levels

    h, w = patch.shape

    for r in xrange(h):
        if r + dy >= h:
            continue
        for c in xrange(w):
            if c + dx >= w:
                continue

            cur = min(patch[r,c] / per_level, levels-1)
            nex = min(patch[r+dy,c+dx] / per_level, levels-1)

            texel[cur, nex] += 1

    return (texel + texel.T) / ((h - dy) * (w - dx) * 2)

def asm(g):

    h, w, levels, _ = g.shape
    res = np.zeros((h, w), dtype=np.float64)

    for r in xrange(h):
        for c in xrange(w):
            res[r,c] = np.sum(g[r,c]**2)

    return res


def entropy(g):

    h, w, levels, _ = g.shape
    res = np.zeros((h, w), dtype=np.float64)

    for r in xrange(h):
        for c in xrange(w):
            res[r,c] = np.sum(g[r,c] * np.log(g[r,c] + 0.01))

    return res

def inertia(g):

    h, w, levels, _ = g.shape
    res = np.zeros((h, w), dtype=np.float64)

    ii = np.array([[i] * levels for i in xrange(1, levels+1)])
    jj = np.array([range(1, levels+1)] * levels)

    for r in xrange(h):
        for c in xrange(w):
            res[r,c] = np.sum(g[r,c] * (ii - jj)**2)

    return res

def cluster_shade(g):

    h, w, levels, _ = g.shape
    res = np.zeros((h, w), dtype=np.float64)

    ii = np.array([[i] * levels for i in xrange(1, levels+1)])
    jj = np.array([range(1, levels+1)] * levels)

    for r in xrange(h):
        for c in xrange(w):
            mux = np.sum(ii * g[r, c])
            muy = np.sum(jj * g[r, c])

            res[r,c] = np.sum(g[r, c] * (ii + jj - mux - muy)**3)

    return res

def quadrant(g, rect):

    r0, c0, r1, c1 = rect

    h, w, levels, _ = g.shape
    res = np.zeros((h, w), dtype=np.float64)

    for r in xrange(h):
        for c in xrange(w):

            res[r,c] = float(np.sum(g[r, c][r0:r1, c0:c1])) / np.sum(g[r, c])

    return res

def flat_quadrant(g, rect):

    r0, c0, r1, c1 = rect

    l, levels, _ = g.shape
    res = np.zeros((l), dtype=np.float64)

    for i in xrange(l):
        res[i] = float(np.sum(g[i][r0:r1, c0:c1])) / np.sum(g[i])

    return res

def main():
    img = cv2.imread('zebra_1.tif', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("ORIG", img)

if __name__ == "__main__":
    main()
