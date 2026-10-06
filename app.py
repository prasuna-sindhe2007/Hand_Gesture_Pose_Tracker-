import time
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Real-Time Hand Gesture & Pose Tracker",
    page_icon="🤖",
    layout="wide",
)

st.title("Real-Time Hand Gesture, Pose & Emotion Tracker")
st.write(
    "Computer Vision system tracking Hands (with Finger Gestures), Poses, and"
    " Facial Expressions."
)

st.markdown("---")

# Sidebar Controls
st.sidebar.header("Camera & Tracker Controls")
run_camera = st.sidebar.checkbox("Turn Camera ON / OFF", value=False)

st.sidebar.subheader("Confidence Thresholds")
detection_confidence = st.sidebar.slider(
    "Detection Confidence", 0.1, 1.0, 0.5, 0.05
)
tracking_confidence = st.sidebar.slider("Tracking Confidence", 0.1, 1.0, 0.5, 0.05)

# Main UI Layout
col1, col2 = st.columns([3, 1])

with col1:
  st.subheader("Live Camera")
  frame_placeholder = st.empty()

with col2:
  st.subheader("Detection Results")
  status_metric = st.empty()
  gesture_metric = st.empty()
  pose_metric = st.empty()
  hands_count_metric = st.empty()

  # Default metrics display when off
  status_metric.markdown("**Status**\n\nStopped")
  gesture_metric.markdown("**Gesture**\n\nWaiting...")
  pose_metric.markdown("**Pose**\n\nWaiting...")
  hands_count_metric.markdown("**Hands Detected**\n\n0")

# Initialize MediaPipe Solutions
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def count_raised_fingers(hand_landmarks):
  """Logic to count extended fingers based on MediaPipe landmark positions."""
  landmarks = hand_landmarks.landmark

  # Finger tip and PIP/MCP landmark indices
  # Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
  finger_tips = [8, 12, 16, 20]
  finger_pips = [6, 10, 14, 18]

  up_count = 0

  # Check 4 main fingers (Y-coordinate in OpenCV is inverted: lower Y means higher on screen)
  for tip, pip in zip(finger_tips, finger_pips):
    if landmarks[tip].y < landmarks[pip].y:
      up_count += 1

  # Check Thumb (horizontal check depending on hand orientation)
  if abs(landmarks[4].x - landmarks[2].x) > 0.04:
    if landmarks[4].y < landmarks[2].y:
      up_count += 1

  return up_count


# Camera Loop Triggered by Checkbox
if run_camera:
  status_metric.markdown("**Status**\n\n🔴 Running")
  cap = cv2.VideoCapture(0)

  if not cap.isOpened():
    st.error(
        "Error: Could not access the webcam. Please check permissions or close"
        " other apps using it."
    )
  else:
    prev_time = 0

    with (
        mp_hands.Hands(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        ) as hands,
        mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        ) as pose,
    ):

      while run_camera and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          st.warning("Failed to grab frame from webcam.")
          break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Mirror frame horizontally & convert color
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process MediaPipe
        hands_results = hands.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        hand_count = 0
        detected_gesture = "No Hand"

        # Hand Detection & Gesture Analysis
        if hands_results.multi_hand_landmarks:
          hand_count = len(hands_results.multi_hand_landmarks)

          for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Count fingers for the detected hand
            raised_fingers = count_raised_fingers(hand_landmarks)

            if raised_fingers == 1:
              detected_gesture = "☝️ 1 Finger (One / Pointing)"
            elif raised_fingers == 2:
              detected_gesture = "✌️ 2 Fingers (Victory / Two)"
            elif raised_fingers == 3:
              detected_gesture = "🤟 3 Fingers"
            elif raised_fingers == 4:
              detected_gesture = "🖐️ 4 Fingers"
            elif raised_fingers == 5:
              detected_gesture = "🖐️ Open Palm (5)"
            elif raised_fingers == 0:
              detected_gesture = "✊ Fist (0)"
            else:
              detected_gesture = f"{raised_fingers} Fingers"

        # Draw Pose Landmarks
        if pose_results.pose_landmarks:
          mp_drawing.draw_landmarks(
              rgb_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
          )

        pose_status = (
            "Detected" if pose_results.pose_landmarks else "Searching..."
        )

        # Update Metrics Panel
        status_metric.markdown("**Status**\n\n🟢 Running")
        gesture_metric.markdown(f"**Gesture**\n\n**{detected_gesture}**")
        pose_metric.markdown(f"**Pose**\n\n{pose_status}")
        hands_count_metric.markdown(f"**Hands Detected**\n\n{hand_count}")

        # Render video feed
        frame_placeholder.image(
            rgb_frame, channels="RGB", use_container_width=True
        )

      cap.release()
else:
  frame_placeholder.info(
      "The live camera and tracking will be connected here. Check 'Turn Camera"
      " ON / OFF' in the sidebar to begin."
  )

st.markdown("---")
st.subheader("Project Information")
st.write(
    "This system uses OpenCV and MediaPipe to detect skeletal landmarks and"
    " recognize finger gestures in real time."
)