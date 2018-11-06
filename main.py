import cv2
from estimateAllTranslation import facetracking

if __name__ == "__main__":
    cap = cv2.VideoCapture('StrangerThings.mp4')
    facetracking(cap)