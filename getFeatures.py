import numpy as np
import cv2
from skimage.feature import corner_shi_tomasi, corner_peaks

def deleteOutsideFeatures(corners, bbox):
    # Record the number of features for this box
    N = corners.shape[0]
    
    # Construct the contour of the bounding box
    contour = np.array(bbox, dtype=np.int32)
    
    # Initialize the place holders
    deletedCorners = np.empty((0, 2))
    
    # Loop over all feature points to examine whether it lies inside the bounding box
    for i in range(N):
        if cv2.pointPolygonTest(contour, (corners[i, 1], corners[i, 0]), False):
            deletedCorners = np.append(deletedCorners, corners[i][np.newaxis, :], axis=0)
            
    return deletedCorners
    

def getFeatures(img, bbox):
    # Record the size of the image and number of faces to track
    H, W = img.shape
    F = bbox.shape[0]
    
    # Initialize the place holders
    x = np.empty((0, 0))
    y = np.empty((0, 0))
    
    # Loop over F faces
    for i in range(F):
        # Extract the coordinates of this bounding box
        Xs, Ys = bbox[i, :, 1], bbox[i, :, 0]
        
        Xmin, Xmax, Ymin, Ymax = np.min(Xs), np.max(Xs), np.min(Ys), np.max(Ys)
        
        # Slice the face to use detectors
        face = img[int(Ymin):int(Ymax), int(Xmin):int(Xmax)]
        
        # Use corner detectors
        corners = corner_peaks(corner_shi_tomasi(face, sigma=1.5), min_distance=3)
        
        # Delete the features outside the tilted bounding box
        corners = deleteOutsideFeatures(corners, bbox[i, :, :])
        
        new_xs, new_ys = corners[:, 1], corners[:, 0]
        new_xs = new_xs + Xmin
        new_ys = new_ys + Ymin
        
        num_of_new_corners = corners.shape[0]
        max_num_of_corners = x.shape[0]
        
        # Update the results, using -1 padding at the end of each column
        if num_of_new_corners > max_num_of_corners:
            
            # If the number of new corners is greater than the current maximum number of corners,
            # then pad the current matrix with a minus-one chunk before append the new coordinates
            minus_one_pad = - np.ones(((num_of_new_corners - max_num_of_corners), i))
            x = np.append(x, minus_one_pad, axis=0)
            x = np.append(x, new_xs[:, np.newaxis], axis=1)
            
            y = np.append(y, minus_one_pad, axis=0)
            y = np.append(y, new_ys[:, np.newaxis], axis=1)
            
        # If the opposite, pad the new coordinates columns with minus ones and then append
        else:
            x_pad = - np.ones(max_num_of_corners)
            x_pad[:num_of_new_corners] = new_xs
            x = np.append(x, x_pad[:, np.newaxis], axis=1)
                
            y_pad = - np.ones(max_num_of_corners)
            y_pad[:num_of_new_corners] = new_ys
            y = np.append(y, y_pad[:, np.newaxis], axis=1)
        
    
    return x, y
