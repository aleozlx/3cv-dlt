import numpy as np
import cv2
from matplotlib import pyplot as plt

def homography_cv2(src, dst):
    """ OpenCV implementation as golden standard """
    src_pts = src.reshape(-1,1,2)
    dst_pts = dst.reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask.ravel().tolist()

def homography_kron(x, y):
    """ Kronecker product implementing the decomposed cross product """
    return np.kron(np.array([
        [0, -y[2], y[1]],
        [-y[2], 0, y[0]],
        [-y[1], y[0], 0]
    ]), x.T)

def dlt(X, Y):
    """ Random DLT expecting shape (3,?) """
    choices = np.random.permutation(X.shape[1])[:4]
    A = np.vstack([homography_kron(x, y) for x,y in zip(X[:, choices].T, Y[:, choices].T)])
    V = np.linalg.svd(A)[2].T
    H = V[:, -1].reshape(-1, X.shape[0]) / V[-1,-1]
    H[2,0] = H[2,1] = 0 # Enforce affine transformation
    return H

def homography(src, dst):
    X, Y = np.vstack([src.T, np.ones(len(src))]), np.vstack([dst.T, np.ones(len(src))])
    d = lambda H: np.sum((Y-np.dot(H,X))**2, axis=0) + np.sum((X-np.dot(np.linalg.inv(H),Y))**2, axis=0)
    
    # RANSAC
    inliers = lambda H: d(H)<5**2*2
    H = max((dlt(X, Y) for i in range(2000)), key=lambda H:np.sum(inliers(H)))
    return H, inliers(H).astype(int).tolist()

def main():
    # SIFT matching
    sift = cv2.xfeatures2d.SIFT_create()
    (kpsA, desA) = sift.detectAndCompute(imA, None)
    (kpsB, desB) = sift.detectAndCompute(imB, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desA,desB,k=2)

    # Outlier detection
    good_matches = [m for m,n in matches if m.distance < 0.7*n.distance] # Lowe's ratio test
    if len(good_matches)>10:
        M, matchesMask = homography(
            np.float32([ kpsA[m.queryIdx].pt for m in good_matches ]),
            np.float32([ kpsB[m.trainIdx].pt for m in good_matches ]))
        print(M, matchesMask)
        h,w = imA.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        cv2.polylines(imB,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print("Not enough matches are found")
        matchesMask = None
    
    # Plot transformed bounding box and matching points
    imC = cv2.drawMatches(imA,kpsA,imB,kpsB,good_matches,None,
        matchColor = (0,255,0), # draw matches in green color
        singlePointColor = None,
        matchesMask = matchesMask, # draw only inliers
        flags = 2)
    plt.figure(figsize=(20,10))
    plt.imshow(imC, 'gray')
    plt.show()

    # print(homography(
    #     np.float32([ kpsA[m.queryIdx].pt for m in good_matches ]),
    #     np.float32([ kpsB[m.trainIdx].pt for m in good_matches ])))

if __name__ == '__main__':
    imA = cv2.cvtColor(cv2.imread("A.png"), cv2.COLOR_BGR2GRAY)
    imB = cv2.cvtColor(cv2.imread("B1.png"), cv2.COLOR_BGR2GRAY)
    main()
