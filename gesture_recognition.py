import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def count_fingers(hand_landmarks):

    count = 0

    # Index finger
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        count += 1

    # Middle finger
    if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
        count += 1

    # Ring finger
    if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
        count += 1

    # Pinky
    if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
        count += 1

    return count


def recognize_gesture(finger_count):

    if finger_count == 0:
        return "FIST"

    elif finger_count == 1:
        return "ONE FINGER"

    elif finger_count == 2:
        return "PEACE"

    elif finger_count == 4:
        return "OPEN PALM"

    else:
        return "UNKNOWN"


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    gesture = "NO HAND"

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        finger_count = count_fingers(hand_landmarks)

        gesture = recognize_gesture(finger_count)

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Gesture Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()