import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def detect_faces(frame):
    
    faces = detector.detect_faces(frame)

    if faces:
        x, y, width, height = faces[0]['box']  
        face_img = frame[y:y+height, x:x+width]
        
        # Resize for CNN model input
        face_img = cv2.resize(face_img, (224, 224))
        
        # Convert to float32 & normalize
        face_img = face_img.astype("float32") / 255.0

        return face_img  # Return processed face
    return None
