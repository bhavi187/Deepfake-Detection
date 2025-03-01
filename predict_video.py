import cv2
import numpy as np
from tensorflow.keras.models import load_model
from extract_frames import extract_frames
from face_detection import detect_faces

# Load the trained deepfake model
model = load_model("deepfake_model.h5")

def predict_video(video_path, display_frequency=10):
    """
    Predicts deepfake probability on a video.
    - Detects faces from all frames.
    - Processes every frame but only displays results every 'display_frequency' frames.
    """
    total_predictions = []
    frame_count = 0

    for frame in extract_frames(video_path):
        frame_count += 1

        # Detect face in current frame
        face = detect_faces(frame)
        
        if face is not None:
            # Prepare for model input
            face = np.expand_dims(face, axis=0).astype("float32")

            # Get deepfake prediction
            prediction = model.predict(face, verbose=0)[0][0]
            deepfake_probability = prediction * 100
            total_predictions.append(deepfake_probability)

            # Only print every 'display_frequency' frames
            if frame_count % display_frequency == 0:
                print(f"Frame {frame_count} → Deepfake: {deepfake_probability:.2f}%")

    # Final results
    if not total_predictions:
        print("\nNo faces detected in the video.")
        return

    avg_probability = np.mean(total_predictions)
    print(f"\nFinal Deepfake Score: {avg_probability:.2f}%")

    if avg_probability > 50:
        print("❌ The video is likely FAKE.")
    else:
        print("✅ The video is likely REAL.")

# Run the function
video_path = "ds1.mp4"  # Replace with actual video file
predict_video(video_path)
