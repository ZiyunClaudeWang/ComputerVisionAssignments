import cv2
import numpy as np

def detectFace(img):
    model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = model.detectMultiScale(img, 1.1, 5)
    num_faces = len(faces)

    #check if there are any faces
    if num_faces == 0:
        print "There are no faces in the image."
        return None
    face_matrix = np.zeros((num_faces, 4, 2))
    index = 0

    # store all faces
    for (x,y,w,h) in faces:
        one_face = np.zeros((4,2))
        #left-up
        one_face[0, :] = [y, x]
        #left_down
        one_face[1, :] = [y+h, x]
        #right_down
        one_face[2, :] = [y+h, x+w]
        #right_up
        one_face[3, :] = [y, x+w]
        face_matrix[index, :, :] = one_face
        index += 1
    return face_matrix.astype(int)
