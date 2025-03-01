import cv2
import numpy as np
from tensorflow.keras.models import load_model
from extract_frames import extract_frames
from face_detection import detect_faces


model = load_model("deepfake_model.h5")

def predict_video(video_path, display_frequency=10):
    
    total_predictions = []
    frame_count = 0

    for frame in extract_frames(video_path):
        frame_count += 1

        face = detect_faces(frame)
        
        if face is not None:
            
            face = np.expand_dims(face, axis=0).astype("float32")

    
            prediction = model.predict(face, verbose=0)[0][0]
            deepfake_probability = prediction * 100
            total_predictions.append(deepfake_probability)

    
            if frame_count % display_frequency == 0:
                print(f"Frame {frame_count} → Deepfake: {deepfake_probability:.2f}%")


    if not total_predictions:
        print("\nNo faces detected in the video.")
        return

    avg_probability = np.mean(total_predictions)
    print(f"\nFinal Deepfake Score: {avg_probability:.2f}%")

    if avg_probability > 50:
        print("❌ The video is likely FAKE.")
    else:
        print("✅ The video is likely REAL.")


video_path = "ds1.mp4"  
predict_video(video_path)
