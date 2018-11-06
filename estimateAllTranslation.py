import numpy as np
import cv2
import skimage.transform as transform
import detectFace
from getFeatures import getFeatures
from scipy.signal import convolve2d

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):

    # gray-scale image
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    It = img2_gray * 1.0 - img1_gray * 1.0

    box_size = 10
    box_offset = box_size / 2
    shape = img1.shape

    # calculate the window
    left_bound = np.max([0, int(startX - box_offset + 1)])
    right_bound = np.min([int(startX + box_offset), shape[1]])
    up_bound = np.max([0, int(startY - box_offset + 1)])
    low_bound = np.min([int(startY + box_offset), shape[0]])

    # get all the necessary data
    Ix_window = Ix[up_bound: low_bound , left_bound: right_bound]
    Iy_window = Iy[up_bound: low_bound , left_bound: right_bound]
    It_window = It[up_bound: low_bound, left_bound: right_bound]

    # get the left-hand matrix
    left_matrix = np.zeros((2,2))
    left_matrix[0, 0] = np.sum(np.multiply(Ix_window, Ix_window))
    left_matrix[0, 1] = np.sum(np.multiply(Ix_window, Iy_window))
    left_matrix[1, 0] = left_matrix[0, 1]
    left_matrix[1, 1] = np.sum(np.multiply(Iy_window, Iy_window))

    # get the right-hand matrix
    right_matrix = np.zeros((2,1))
    right_matrix[0, 0] = np.sum(np.multiply(Ix_window, It_window))
    right_matrix[1, 0] = np.sum(np.multiply(Iy_window, It_window))
    right_matrix = right_matrix

    # solve for u and v
    result = np.linalg.solve(left_matrix, right_matrix)
    return (startX + result[0], startY + result[1])

def estimateAllTranslation(startXs, startYs, img1, img2):

    shape = startXs.shape
    # calculate the gradient in x, y direction
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    x_kernel = np.array([[-1, 1]])
    y_kernel = np.array([[-1], [1]])
    Ix = (convolve2d(img1_gray, x_kernel, "same") + convolve2d(img2_gray, x_kernel, "same"))
    Iy = (convolve2d(img1_gray, y_kernel, "same") + convolve2d(img2_gray, y_kernel, "same"))

    # estimate translation for all the features
    newStartXs = np.array(startXs)
    newStartYs = np.array(startYs)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if startXs[i, j] >= 0:

                # estimate the translation for one pixel
                x = startXs[i, j]
                y = startYs[i, j]
                new_pos = estimateFeatureTranslation(x, y, Ix, Iy, img1, img2)
                newStartXs[i, j] = new_pos[0]
                newStartYs[i, j] = new_pos[1]

    return (newStartXs, newStartYs)

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    # for each face bounding box, estimate the trasformation.
    threshold = 4
    shape = startXs.shape
    Xs = np.multiply(np.ones(shape), -1)
    Ys = np.multiply(np.ones(shape), -1)
    new_bbox = np.array(bbox)
    N1 = -1
    for i in range(shape[1]):

        # find index of the last valid feature index
        box_x = startXs[:, i]
        box_y = startYs[:, i]
        last_index = shape[0]
        if last_index == 0:
            continue
        for j in range(shape[0]):
            if box_x[j] < 0:
                last_index = j
                break

        # get the valid startXs and startYs
        xs = box_x[0:last_index]
        ys = box_y[0:last_index]
        xs_new = newXs[0:last_index, i]
        ys_new = newYs[0:last_index, i]

        src = np.column_stack((xs, ys))
        des = np.column_stack((xs_new, ys_new))

        # eliminate the outliers
        inliers_old = np.empty((0, 2))
        inliers_new = np.empty((0, 2))
        count = 0

        # check for outliers, only keep inliers.
        for k in range(last_index):
            distance = np.linalg.norm(src[k, :] - des[k, :])
            if distance < threshold:
                inliers_old = np.append(inliers_old, src[[k], :], axis=0)
                inliers_new = np.append(inliers_new, des[[k], :], axis=0)
                count += 1
        if count > N1:
            N1 = count

        # estimate homography for all the inliers.
        inlier_transformation = transform.SimilarityTransform()
        Xs[:inliers_new.shape[0], i] = inliers_new[:, 0]
        Ys[:inliers_new.shape[0], i] = inliers_new[:, 1]
        inlier_transformation.estimate(inliers_new, inliers_old)
        if last_index < 3:
            continue

        # calculate the new bounding box
        corners = bbox[i]
        X = (np.append(corners, np.ones((4,1)), axis=1)).transpose()
        newX = np.dot(inlier_transformation.params, X).transpose()
        new_bbox[i] = newX[:, :2]

    # truncate the Xs and Ys to be size N1
    Xs = Xs[:N1, :]
    Ys = Ys[:N1, :]
    return (Xs, Ys, new_bbox)

def extractFrames(rawVideo):
    frames = []
    while (rawVideo.isOpened()):
        ret, frame = rawVideo.read()
        if not ret:
            rawVideo.release()
            return frames
        frame_resized = cv2.resize(frame, (0, 0), fx = 1, fy = 1)
        frames.append(frame_resized)
    return frames

def facetracking(rawVideo):

    frames = extractFrames(rawVideo)

    # first frame
    preFrame = frames[0]
    bbox = detectFace.detectFace(preFrame)
    preFrame_gray = cv2.cvtColor(preFrame, cv2.COLOR_BGR2GRAY)
    [x, y] = getFeatures(preFrame_gray, bbox)
    preX = x
    preY = y
    preBbox = bbox * 1.0
    trackedImgs = []
    count = 0

    for frame in frames[1:]:

        # start with the second frame.
        curFrame = frame
        (XCur, YCur) = estimateAllTranslation(preX, preY, preFrame, curFrame)
        (XNewCur, YNewCur, bboxcur) = applyGeometricTransformation(preX, preY, XCur, YCur, preBbox)

        marked = np.array(curFrame)

        # draw feature points
        for i in range(XNewCur.shape[0]):
            for j in range(XNewCur.shape[1]):
                if XNewCur[i, j] == -1:
                    continue
                drawX = XNewCur.astype(int)
                drawY = YNewCur.astype(int)
                cv2.circle(marked,(drawX[i,j], drawY[i, j]), 2, (0,0,255), 1)

        # draw bounding boxes
        for i in range(bboxcur.shape[0]):
            one_box = bboxcur[i]
            pts = np.array([[one_box[0, 1], one_box[0, 0]], [one_box[1, 1], one_box[1, 0]], [one_box[2, 1], one_box[2, 0]], [one_box[3, 1], one_box[3, 0]]])
            pts_int = pts.astype(int)
            cv2.polylines(marked, [pts_int],
                          True, (0, 255, 0), 2)

        trackedImgs.append(marked)
        cv2.imshow('frame', marked)
        cv2.waitKey(0)
        preX = XNewCur
        preY = YNewCur
        preFrame = curFrame
        preBbox = bboxcur
        count += 1

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('output.mov', fourcc, 12, (preFrame.shape[1], preFrame.shape[0]))
    for frame in trackedImgs:
        out.write(frame)

    # return the video
    return out