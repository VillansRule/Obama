import os
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
SMOOTH_FACTOR = 0.2  # Controls smoothness of mouse movement
VIDEO_PATH = r"/home/alex/Downloads/3d prints/Watch President Obama deliver his final speech at United Nations.mp4"
SENSITIVITY = 5.0  # Adjust sensitivity for mouse movement
PAUSE_KEY = 'p'  # Key to pause/resume hand tracking
DRAG_HOLD_DURATION = 0.5  # Time in seconds to hold for dragging

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)  # Lower detection confidence to 0.5 for testing
mp_draw = mp.solutions.drawing_utils

# Get screen size for mouse control scaling
screen_width, screen_height = pyautogui.size()

# Initialize variables for smoothing mouse movement
prev_x, prev_y = 0, 0
clicking = False  # Track clicking state
paused = False  # Track whether hand tracking is paused
dragging = False  # Track dragging state
drag_start_time = None

# Colors for drawing on screen
CLICK_COLOR = (0, 255, 0)  # Green for left click
RIGHT_CLICK_COLOR = (0, 0, 255)  # Red for right click
DRAG_COLOR = (255, 255, 0)  # Yellow for dragging


def process_frame(img):
    """Process the frame for hand tracking and return landmarks if detected."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    results = hands.process(img_rgb)  # Detect hand landmarks
    return results.multi_hand_landmarks if results.multi_hand_landmarks else None


def calculate_screen_coordinates(landmark, img_shape):
    """Convert hand landmark position to screen coordinates."""
    x = int(landmark.x * img_shape[1])
    y = int(landmark.y * img_shape[0])
    screen_x = screen_width / img_shape[1] * x * SENSITIVITY
    screen_y = screen_height / img_shape[0] * y * SENSITIVITY
    return screen_x, screen_y


def smooth_movement(screen_x, screen_y):
    """Smooth the mouse movement by adjusting the position incrementally."""
    global prev_x, prev_y
    screen_x = prev_x + (screen_x - prev_x) * SMOOTH_FACTOR
    screen_y = prev_y + (screen_y - prev_y) * SMOOTH_FACTOR
    prev_x, prev_y = screen_x, screen_y
    return screen_x, screen_y


def clamp_screen_coordinates(screen_x, screen_y):
    """Clamp the screen coordinates to stay within screen bounds."""
    screen_x = max(0, min(screen_width - 1, screen_x))
    screen_y = max(0, min(screen_height - 1, screen_y))
    return screen_x, screen_y


def handle_gestures(landmarks, img):
    """Detect gestures and perform mouse clicks or dragging based on hand positions."""
    global clicking, dragging, drag_start_time

    # Extract important landmarks for gestures
    index_finger_tip = landmarks.landmark[8]
    thumb_tip = landmarks.landmark[4]
    middle_finger_tip = landmarks.landmark[12]

    # Gesture: Left click (index down, thumb up)
    if index_finger_tip.y > landmarks.landmark[6].y and thumb_tip.y < landmarks.landmark[2].y:
        if not clicking:
            pyautogui.click(button='left')
            clicking = True
            cv2.circle(img, (int(index_finger_tip.x * img.shape[1]), int(index_finger_tip.y * img.shape[0])), 10, CLICK_COLOR, -1)

    # Gesture: Right click (index & middle down, thumb up)
    elif (index_finger_tip.y > landmarks.landmark[6].y and
          middle_finger_tip.y > landmarks.landmark[10].y and
          thumb_tip.y < landmarks.landmark[2].y):
        if not clicking:
            pyautogui.click(button='right')
            clicking = True
            cv2.circle(img, (int(index_finger_tip.x * img.shape[1]), int(index_finger_tip.y * img.shape[0])), 10, RIGHT_CLICK_COLOR, -1)

    # Gesture: Right-click hold for dragging
    elif index_finger_tip.y > landmarks.landmark[6].y and middle_finger_tip.y < landmarks.landmark[10].y:
        if not dragging:
            drag_start_time = pyautogui.time.time()
            dragging = True
        if pyautogui.time.time() - drag_start_time > DRAG_HOLD_DURATION:
            pyautogui.mouseDown(button='right')
            cv2.circle(img, (int(index_finger_tip.x * img.shape[1]), int(index_finger_tip.y * img.shape[0])), 10, DRAG_COLOR, -1)

    else:
        clicking = False  # Reset click state when fingers are raised
        if dragging:
            pyautogui.mouseUp(button='right')
            dragging = False


def toggle_pause():
    """Toggle between pausing and resuming the hand tracking functionality."""
    global paused
    paused = not paused
    if paused:
        print("Hand tracking paused.")
    else:
        print("Hand tracking resumed.")


def main():
    """Main loop to capture video frames and handle hand tracking."""
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # Check for pause toggle
        if cv2.waitKey(1) & 0xFF == ord(PAUSE_KEY):
            toggle_pause()

        # Optional: Flip image horizontally for natural webcam-like view
        img = cv2.flip(img, 1)

        if not paused:
            # Process frame for hand landmarks
            landmarks_list = process_frame(img)
            if landmarks_list:
                for landmarks in landmarks_list:
                    # Draw hand landmarks
                    mp_draw.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)

                    # Calculate screen coordinates for index finger
                    screen_x, screen_y = calculate_screen_coordinates(landmarks.landmark[8], img.shape)

                    # Print debug info for screen coordinates
                    print(f"Screen coordinates: x={screen_x}, y={screen_y}")

                    # Smooth the mouse movement and clamp to avoid FailSafeException
                    screen_x, screen_y = smooth_movement(screen_x, screen_y)
                    screen_x, screen_y = clamp_screen_coordinates(screen_x, screen_y)

                    # Move mouse cursor
                    pyautogui.moveTo(screen_x, screen_y)

                    # Handle click gestures
                    handle_gestures(landmarks, img)

        # Show the video feed with hand landmarks
        cv2.imshow("Hand Tracking", img)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
