import cv2

def extract_frames(video_path):
    """Extracts frames from the given video."""
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when video ends
        yield frame  # Return frame one by one

    cap.release()
