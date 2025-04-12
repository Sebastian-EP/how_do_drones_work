import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math

#--- Marker size in cm
marker_size = 27.94

#--- Load calibration parameters
calib_path = ""
camera_matrix = np.loadtxt(calib_path + 'cameraMatrix_webcam.txt', delimiter=',')
camera_distortion = np.loadtxt(calib_path + 'cameraDistortion_webcam.txt', delimiter=',')

#--- 180 deg flip matrix around x-axis
R_flip = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1]], dtype=np.float32)

#--- Use 6x6 dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

#--- GStreamer pipeline
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink"
)

print("🎥 Opening CSI camera...")
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ ERROR: Failed to open camera.")
    sys.exit(1)

font = cv2.FONT_HERSHEY_PLAIN

def toEuler(R):
    assert R.shape == (3, 3)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])


marker_previously_detected = False
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ WARNING: Failed to capture frame.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters,
                                                 cameraMatrix=camera_matrix, distCoeff=camera_distortion)

    if ids is not None:
        if not marker_previously_detected:
            print("🔁 Marker detected! [Simulated signal to Arduino]")
            marker_previously_detected = True
        aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            ret = aruco.estimatePoseSingleMarkers([corners[i]], marker_size, camera_matrix, camera_distortion)
            rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

            aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

            # Display ID and position
            cv2.putText(frame, f"ID: {marker_id}", (10, 100 + i*70), font, 1.3, (0, 255, 0), 2)
            cv2.putText(frame, f"Pos x={tvec[0]:.1f} y={tvec[1]:.1f} z={tvec[2]:.1f}",
                        (10, 125 + i*70), font, 1.2, (0, 255, 0), 2)

            R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
            R_tc = R_ct.T
            pos_camera = -R_tc @ np.matrix(tvec).T
            euler_angles = toEuler(R_flip @ R_tc)

            # Fixed formatting for numpy matrix output
            cv2.putText(frame, f"CAM x={pos_camera[0,0]:.1f} y={pos_camera[1,0]:.1f} z={pos_camera[2,0]:.1f}",
                        (10, 150 + i*70), font, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f"Rot r={euler_angles[0]:.1f} p={euler_angles[1]:.1f} y={euler_angles[2]:.1f}",
                        (10, 175 + i*70), font, 1.2, (0, 255, 0), 2)

    else:
        marker_previously_detected = False
    cv2.imshow("Jetson CSI Camera - ArUco Multi Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
