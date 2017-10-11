import numpy as np
import cv2
from matplotlib import pyplot as plt

def dlt(X, Y):
    """ Random DLT expecting shape (3,?) """
    homography_kron = lambda x,y: np.kron(np.array([
        [0, -y[2], y[1]],
        [-y[2], 0, y[0]],
        [-y[1], y[0], 0]
    ]), x.T)
    choices = np.random.permutation(X.shape[1])[:4]
    A = np.vstack([homography_kron(x, y) for x,y in zip(X[:, choices].T, Y[:, choices].T)])
    V = np.linalg.svd(A)[2].T
    H = V[:, -1].reshape(-1, X.shape[0]) / V[-1,-1]
    H[2,0] = H[2,1] = 0 # Enforce affine transformation
    return H

def homography(src, dst):
    """ Compute homography (should drop-in replace homography_cv2) """
    X, Y = np.vstack([src.T, np.ones(len(src))]), np.vstack([dst.T, np.ones(len(src))])   
    inliers = lambda H: np.sum((Y-np.dot(H,X))**2, axis=0) + np.sum((X-np.dot(np.linalg.inv(H),Y))**2, axis=0)<5**2*2
    H = max((dlt(X, Y) for i in range(2000)), key=lambda H:np.sum(inliers(H))) # RANSAC
    return H, inliers(H).astype(int).tolist()

# def homography_cv2(src, dst):
#     """ OpenCV implementation as golden standard """
#     src_pts = src.reshape(-1,1,2)
#     dst_pts = dst.reshape(-1,1,2)
#     H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     return H, mask.ravel().tolist()

def plot_matching(imA,kpsA,imB,kpsB,matches,inliers):
    im = cv2.drawMatches(imA,kpsA,imB,kpsB,matches,None,
        matchColor = (0,255,0), # draw matches in green color
        singlePointColor = None,
        matchesMask = inliers, # draw only inliers
        flags = 2)
    plt.figure(figsize=(20,10))
    plt.imshow(im, 'gray')

def plot_stitching(imA,imB,H):
    is_gray = len(imA.shape)==2
    # Transform imA bounding box
    h,w = imA.shape[:2]
    transformed_box = cv2.perspectiveTransform(np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2),H)
    # Find bounding box
    bounding_box = np.array([
        min(0, *transformed_box[:,0,1]),              # [0,0] top
        max(imB.shape[0]-1, *transformed_box[:,0,1]), # [0,1] bottom
        min(0, *transformed_box[:,0,0]),              # [1,0] left
        max(imB.shape[1]-1, *transformed_box[:,0,0]), # [1,1] right
    ]).reshape((2,2))
    translate_transform = -np.expand_dims(bounding_box[:,0], 1)
    shape_imC = np.ceil((bounding_box+translate_transform)[:,1]).astype(int)
    box_imB = (np.array([0, imB.shape[0]-1, 0, imB.shape[1]-1]).reshape((2,2))+translate_transform).astype(int)
    # Allocate new image
    imC = np.zeros(shape_imC if is_gray else tuple(shape_imC)+(3,))
    # Copy imB onto imC
    imC[box_imB[0,0]:box_imB[0,0]+imB.shape[0],box_imB[1,0]:box_imB[1,0]+imB.shape[1]] = imB
    # Copy imA onto imC
    H = H.copy()
    H[:2,2] += np.squeeze(translate_transform)[::-1] # adjust translation to imC
    _imA = cv2.warpPerspective(imA,H,imC.shape[:2][::-1])
    imC[_imA!=0]=_imA[_imA!=0]
    # print(shape_imC)
    plt.figure(figsize=(20,10))
    plt.imshow(imC[...,::-1]/255)

def main():
    # Read images
    imA = cv2.cvtColor(cv2.imread("A.png"), cv2.COLOR_BGR2GRAY)
    imB = cv2.cvtColor(cv2.imread("B1.png"), cv2.COLOR_BGR2GRAY)
    # SIFT+FLANN matching
    sift = cv2.xfeatures2d.SIFT_create()
    (kpsA, desA) = sift.detectAndCompute(imA, None)
    (kpsB, desB) = sift.detectAndCompute(imB, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm = 0, trees = 5), dict(checks = 50))
    # Lowe's ratio test
    matches = [m for m,n in flann.knnMatch(desA,desB,k=2) if m.distance < 0.7*n.distance] 
    if len(matches)>10:
        while 1: # Just try again when getting singular matrix
            try:
                H, inliers = homography(
                    np.float32([ kpsA[m.queryIdx].pt for m in matches ]),
                    np.float32([ kpsB[m.trainIdx].pt for m in matches ]))
                break
            except np.linalg.linalg.LinAlgError:
                continue
    else:
        print("Not enough matches are found")
        sys.exit(1)
    plot_matching(imA,kpsA,imB,kpsB,matches,inliers)
    plot_stitching(cv2.imread("A.png"),cv2.imread("B1.png"),H)
    plt.show()

if __name__ == '__main__':
    main()
