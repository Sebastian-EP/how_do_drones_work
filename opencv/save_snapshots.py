import cv2
import time
import argparse
import os

def save_snaps(width=1280, height=720, name="snapshot", folder="."):
    # Correct GStreamer pipeline for Jetson CSI camera
    gst_pipeline = (
        f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ ERROR: Could not open camera with GStreamer pipeline.")
        return

    if not os.path.exists(folder):
        os.makedirs(folder)

    nSnap = 0
    fileName = f"{folder}/{name}_{width}_{height}_"

    print("📸 Press SPACE to capture, Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ WARNING: Failed to read frame.")
            continue

        cv2.imshow("camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            path = f"{fileName}{nSnap}.jpg"
            cv2.imwrite(path, frame)
            print(f"✅ Saved {path}")
            nSnap += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Jetson CSI Camera Snapshot Tool")
    parser.add_argument("--folder", default=".", help="Save folder")
    parser.add_argument("--name", default="snapshot", help="Base file name")
    parser.add_argument("--dwidth", type=int, default=1280, help="Width")
    parser.add_argument("--dheight", type=int, default=720, help="Height")
    args = parser.parse_args()

    save_snaps(width=args.dwidth, height=args.dheight, name=args.name, folder=args.folder)

if __name__ == "__main__":
    main()
