import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from ctypes import cast, POINTER, c_ulong, windll
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import subprocess
import pyautogui
import psutil
import screen_brightness_control as sbc
import logging  # Added for logging system events

# Configuration
VOLUME_RANGE = [10, 100]
MOUSE_SMOOTHING = 5
OVERLAY_ALPHA = 0.5
BRIGHTNESS_RANGE = [10, 100]

# Media key constants
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        # Audio control
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface, POINTER(IAudioEndpointVolume))
        self.vol_min, self.vol_max = self.volume.GetVolumeRange()[:2]

        # State tracking
        self.prev_time = 0
        self.media_cooldown = 0
        self.last_media_state = None
        self.vol_history = []
        self.mouse_history = []
        self.spotify_opened = False
        self.left_click_triggered = False
        self.right_click_triggered = False
        self.next_triggered = False
        self.prev_triggered = False
        self.frame_width = 640
        self.frame_height = 480

        # Brightness control
        self.auto_brightness_enabled = True  # Track auto-brightness state
        self.current_brightness = 50  # Default brightness level

        # Initialize logging
        logging.basicConfig(filename="gesture_control.log", level=logging.INFO, 
                            format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Gesture Controller initialized.")

        # Hand presence tracking
        self.hands_present = False

        # Gesture alert tracking
        self.gesture_alert = None  # To store the current gesture alert
        self.gesture_alert_time = None  # To track when the alert was set

    def toggle_auto_brightness(self):
        """Toggle the auto-brightness state."""
        self.auto_brightness_enabled = not self.auto_brightness_enabled

    def get_system_info(self):
        battery = psutil.sensors_battery()
        return {
            'battery': battery.percent if battery else "N/A",
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent
        }

    def auto_brightness(self, frame):
        """Adjust brightness automatically if auto-brightness is enabled."""
        if not self.auto_brightness_enabled:
            return self.current_brightness  # Return current brightness if auto-brightness is off

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            light_level = cv2.mean(gray)[0]
            brightness = np.interp(light_level, [0, 255], BRIGHTNESS_RANGE)
            current_brightness = sbc.get_brightness(display=0)[0]

            if abs(current_brightness - brightness) > 5:  # Adjust threshold as needed
                sbc.set_brightness(int(brightness))
            self.current_brightness = int(brightness)
            return self.current_brightness
        except Exception as e:
            print(f"Failed to set brightness: {e}")
            return "N/A"

    def process_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        hands = {'left': {'landmarks': None}, 'right': {'landmarks': None}}
        
        if results.multi_hand_landmarks:
            self.hands_present = True  # Hands detected
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_type = results.multi_handedness[i].classification[0].label.lower()
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]
                hands[hand_type]['landmarks'] = landmarks

                # Draw landmarks and connections
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Add a custom joint between landmarks 2 and 5
                x1, y1 = landmarks[2]
                x2, y2 = landmarks[5]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White line for the custom joint
        else:
            self.hands_present = False  # No hands detected

        return hands, frame

    def create_overlay(self, frame, sys_info, auto_brightness_message=None):
        """Create a circular overlay with system details without overlapping."""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Position and size for the black circle
        center_x, center_y = 115, 115  # Center position for the overlay
        main_radius = 110  # Radius of the main black circle
        cv2.circle(overlay, (center_x, center_y), main_radius, (0, 0, 0), -5)  # Black circle for background
        cv2.circle(overlay, (center_x, center_y), main_radius, (255, 255, 255), 3)  # White border

        # Battery percentage as a ring with dynamic color
        battery_percentage = sys_info['battery'] if sys_info['battery'] != "N/A" else 0
        battery_angle = int((battery_percentage / 100) * 360)

        # Determine the color based on the battery percentage
        if battery_percentage >= 80:
            battery_color = (0, 255, 0)  # Green
        elif 50 <= battery_percentage < 80:
            battery_color = (255, 0, 0)  # Blue
        elif 30 <= battery_percentage < 50:
            battery_color = (0, 165, 255)  # Orange
        elif 10 <= battery_percentage < 30:
            battery_color = (0, 0, 255)  # Red
        else:
            battery_color = (128, 128, 128)  # Gray for very low or unavailable battery

        for angle in range(0, battery_angle):  # Draw the ring in continuous segments
            x1 = int(center_x + main_radius * np.cos(np.radians(angle)))
            y1 = int(center_y + main_radius * np.sin(np.radians(angle)))
            x2 = int(center_x + (main_radius - 10) * np.cos(np.radians(angle)))
            y2 = int(center_y + (main_radius - 10) * np.sin(np.radians(angle)))
            cv2.line(overlay, (x1, y1), (x2, y2), battery_color, 3)  # Dynamic color for battery ring

        # Brightness percentage as a ring inside the black circle
        brightness_percentage = self.current_brightness  # Use the current brightness value
        brightness_angle = int((brightness_percentage / 100) * 360)
        for angle in range(0, brightness_angle):  # Draw the ring in continuous segments
            x1 = int(center_x + (main_radius - 20) * np.cos(np.radians(angle)))
            y1 = int(center_y + (main_radius - 20) * np.sin(np.radians(angle)))
            x2 = int(center_x + (main_radius - 30) * np.cos(np.radians(angle)))
            y2 = int(center_y + (main_radius - 30) * np.sin(np.radians(angle)))
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow ring for brightness

        # Smaller circles for other details inside the brightness ring
        small_radius = 25  # Radius of the smaller circles
        inner_circle_radius = main_radius - 60  # Position the smaller circles inside the brightness ring
        details = [
            {"label": "CPU", "value": f"{sys_info['cpu']}%", "color": (255, 0, 0), "angle": 90},
            {"label": "MEM", "value": f"{sys_info['memory']}%", "color": (0, 255, 0), "angle": 180},
            {"label": "FPS", "value": "N/A", "color": (255, 255, 0), "angle": 270}  # FPS removed
        ]

        for detail in details:
            angle_rad = np.radians(detail["angle"])
            x = int(center_x + inner_circle_radius * np.cos(angle_rad))
            y = int(center_y + inner_circle_radius * np.sin(angle_rad))

            # Draw the circle
            cv2.circle(overlay, (x, y), small_radius, detail["color"], -1)
            cv2.circle(overlay, (x, y), small_radius, (255, 255, 255), 2)  # White border
            cv2.putText(overlay, detail["label"], (x - 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(overlay, detail["value"], (x - 15, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Display auto-brightness toggle message if provided
        if auto_brightness_message:
            cv2.putText(overlay, auto_brightness_message, (center_x - 80, center_y + main_radius + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)  # Make the black circle fully opaque

        return frame

    def draw_text_with_emojis(self, frame, text, position, font_size=0.5, color=(255, 255, 255)):
        """Draw text on the frame using OpenCV (no emojis)."""
        try:
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1, cv2.LINE_AA)
            return frame
        except Exception as e:
            print(f"Failed to render text: {e}")
            return frame

    def send_media_key(self, key_code):
        windll.user32.keybd_event(key_code, 0, 0, 0)
        windll.user32.keybd_event(key_code, 0, 2, 0)

    def is_powerpoint_open(self):
        """Check if PowerPoint is running."""
        for process in psutil.process_iter(['name']):
            if process.info['name'] and process.info['name'].lower() == 'powerpnt.exe':
                return True
        return False

    def display_gesture_alert(self, frame):
        """Display the current gesture alert on the right side of the screen."""
        if self.gesture_alert and time.time() - self.gesture_alert_time < 2:  # Show alert for 2 seconds
            text_size = cv2.getTextSize(self.gesture_alert, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 10  # Position 10px from the right edge
            text_y = 50  # Fixed vertical position
            cv2.putText(frame, self.gesture_alert, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif self.gesture_alert:
            self.gesture_alert = None  # Clear the alert after 2 seconds

    def control_media(self, hand):
        if not hand or time.time() - self.media_cooldown < 1:
            return None
        index_tip, thumb_tip, middle_tip, ring_tip, pinky_tip = hand[8], hand[4], hand[12], hand[16], hand[20]
        index_mcp, thumb_mcp, middle_mcp, ring_mcp, pinky_mcp = hand[5], hand[2], hand[9], hand[13], hand[17]

        next_gesture = (index_tip[1] < index_mcp[1] and
                        middle_tip[1] > middle_mcp[1] and
                        ring_tip[1] > ring_mcp[1] and
                        pinky_tip[1] > pinky_mcp[1])

        prev_gesture = (thumb_tip[1] < thumb_mcp[1] and
                        index_tip[1] > index_mcp[1] and
                        middle_tip[1] > middle_mcp[1] and
                        ring_tip[1] > ring_mcp[1] and
                        pinky_tip[1] > pinky_mcp[1])

        play_gesture = all(tip[1] < mcp[1] for tip, mcp in [(index_tip, index_mcp), (middle_tip, middle_mcp), 
                                                             (ring_tip, ring_mcp), (pinky_tip, pinky_mcp)])

        pause_gesture = (index_tip[1] < index_mcp[1] and
                         middle_tip[1] < middle_mcp[1] and
                         ring_tip[1] > ring_mcp[1] and
                         pinky_tip[1] > pinky_mcp[1])

        is_ppt_open = self.is_powerpoint_open()

        if next_gesture and not self.next_triggered:
            if is_ppt_open:
                pyautogui.press('right')  # Next slide
                action = "NEXT SLIDE"
            else:
                self.send_media_key(VK_MEDIA_NEXT_TRACK)  # Next track
                action = "NEXT TRACK"
            self.next_triggered = True
            self.media_cooldown = time.time()
            self.gesture_alert = action  # Set gesture alert
            self.gesture_alert_time = time.time()
            return action
        elif not next_gesture:
            self.next_triggered = False

        if prev_gesture and not self.prev_triggered:
            if is_ppt_open:
                pyautogui.press('left')  # Previous slide
                action = "PREVIOUS SLIDE"
            else:
                self.send_media_key(VK_MEDIA_PREV_TRACK)  # Previous track
                action = "PREVIOUS TRACK"
            self.prev_triggered = True
            self.media_cooldown = time.time()
            self.gesture_alert = action  # Set gesture alert
            self.gesture_alert_time = time.time()
            return action
        elif not prev_gesture:
            self.prev_triggered = False

        if play_gesture and self.last_media_state != 'play':
            self.send_media_key(VK_MEDIA_PLAY_PAUSE)
            self.last_media_state = 'play'
            self.media_cooldown = time.time()
            self.gesture_alert = "PLAY"  # Set gesture alert
            self.gesture_alert_time = time.time()
            return "PLAY"
        elif pause_gesture and self.last_media_state != 'pause':
            self.send_media_key(VK_MEDIA_PLAY_PAUSE)
            self.last_media_state = 'pause'
            self.media_cooldown = time.time()
            self.gesture_alert = "PAUSE"  # Set gesture alert
            self.gesture_alert_time = time.time()
            return "PAUSE"

        return None

    def control_volume(self, hand, frame):
        try:
            if not hand or len(hand) < 9:
                return None
            thumb_tip, index_tip = hand[4], hand[8]

            # Prevent volume control during mouse movement
            index_mcp = hand[5]
            if index_tip[1] < index_mcp[1]:  # Index finger is up (used for mouse movement)
                return None

            length = hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

            # Adjust volume range sensitivity
            length = max(VOLUME_RANGE[0], min(length, VOLUME_RANGE[1]))

            cv2.line(frame, thumb_tip, index_tip, (0, 255, 0), 2)
            vol = np.interp(length, [VOLUME_RANGE[0], VOLUME_RANGE[1]], [self.vol_min, self.vol_max])
            self.vol_history.append(vol)
            if len(self.vol_history) > 5:
                self.vol_history.pop(0)
            avg_vol = np.mean(self.vol_history)
            self.volume.SetMasterVolumeLevel(avg_vol, None)
            self.gesture_alert = f"Volume: {int(np.interp(avg_vol, [self.vol_min, self.vol_max], [10, 100]))}%"  # Set gesture alert
            self.gesture_alert_time = time.time()
            return int(np.interp(avg_vol, [self.vol_min, self.vol_max], [10, 100]))
        except Exception as e:
            print(f"Failed to control volume: {e}")
            return None

    def open_spotify(self, hand):
        if not hand or len(hand) < 20:
            self.spotify_opened = False
            return
        pinky_up = hand[20][1] < hand[17][1]
        other_fingers_closed = all(hand[i][1] > hand[i-3][1] for i in [8, 12, 16])
        
        if pinky_up and other_fingers_closed and not self.spotify_opened:
            try:
                # Try to open Spotify
                subprocess.Popen("start spotify:", shell=True)
                self.spotify_opened = "spotify"
            except FileNotFoundError:
                # If Spotify is not available, open YouTube in the default browser
                subprocess.Popen("start https://www.youtube.com", shell=True)
                self.spotify_opened = "youtube"
        elif not pinky_up and self.spotify_opened:
            # Close Spotify or YouTube
            if self.spotify_opened == "spotify":
                for process in psutil.process_iter(['name']):
                    if process.info['name'] and process.info['name'].lower() == 'spotify.exe':
                        process.terminate()
            elif self.spotify_opened == "youtube":
                pyautogui.hotkey('ctrl', 'w')  # Close the browser tab
            self.spotify_opened = False

    def control_mouse(self, hand, frame):
        if not hand or len(hand) < 17:
            return None
        index_tip, middle_tip, ring_tip, pinky_tip = hand[8], hand[12], hand[16], hand[20]
        index_mcp, middle_mcp, ring_mcp, pinky_mcp = hand[5], hand[9], hand[13], hand[17]
        
        index_up = index_tip[1] < index_mcp[1]
        middle_up = middle_tip[1] < middle_mcp[1]
        ring_up = ring_tip[1] < ring_mcp[1]
        pinky_down = pinky_tip[1] > pinky_mcp[1]

        # Mouse movement: Index finger up, others down
        if index_up and not middle_up and not ring_up and pinky_down:
            screen_w, screen_h = pyautogui.size()
            # Adjust mapping to ensure full screen coverage
            mouse_x = np.interp(index_tip[0], (0, self.frame_width), (0, screen_w))
            mouse_y = np.interp(index_tip[1], (0, self.frame_height), (0, screen_h))
            self.mouse_history.append((mouse_x, mouse_y))
            if len(self.mouse_history) > MOUSE_SMOOTHING:
                self.mouse_history.pop(0)
            avg_x, avg_y = np.mean([x for x, _ in self.mouse_history]), np.mean([y for _, y in self.mouse_history])
            pyautogui.moveTo(max(0, min(avg_x, screen_w)), max(0, min(avg_y, screen_h)), duration=0.05)

            # Draw a blue circle at the index tip to indicate the mouse pointer
            cv2.circle(frame, index_tip, 10, (255, 0, 0), -1)  # Blue circle

        # Left click: Victory gesture (index and middle up, ring and pinky down)
        if index_up and middle_up and not ring_up and pinky_down and not self.left_click_triggered:
            pyautogui.click(button='left')
            self.left_click_triggered = True
            self.gesture_alert = "LEFT CLICK"  # Set gesture alert
            self.gesture_alert_time = time.time()
            return "LEFT"
        elif not (index_up and middle_up and not ring_up and pinky_down):
            self.left_click_triggered = False

        # Right click: Index, middle, ring up, pinky down
        if index_up and middle_up and ring_up and pinky_down and not self.right_click_triggered:
            pyautogui.click(button='right')
            self.right_click_triggered = True
            self.gesture_alert = "RIGHT CLICK"  # Set gesture alert
            self.gesture_alert_time = time.time()
            return "RIGHT"
        elif not (index_up and middle_up and ring_up and pinky_down):
            self.right_click_triggered = False
        return None

    def run(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logging.error("Webcam not detected.")
                raise RuntimeError("Error: Webcam not detected. Please connect a webcam and restart the application.")

            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print("Press 'q' to quit the application.")
            print("Press 'b' to toggle auto-brightness.")
            frame_skip = 2  # Process every 2nd frame to improve FPS
            frame_count = 0
            auto_brightness_message = None
            message_start_time = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logging.error("Unable to read from webcam.")
                    raise RuntimeError("Error: Unable to read from webcam. Exiting...")

                # Flip the frame horizontally to correct the orientation
                frame = cv2.flip(frame, 1)

                # Skip frames to improve FPS
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                hands_data, frame = self.process_hands(frame)

                # Skip further processing if no hands are detected
                if not self.hands_present:
                    cv2.putText(frame, "No hands detected", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Gesture Control', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                brightness = self.auto_brightness(frame)
                sys_info = self.get_system_info()

                left_hand = hands_data['left']['landmarks']
                right_hand = hands_data['right']['landmarks']
                media_action = self.control_media(left_hand) if left_hand else None
                vol_level = self.control_volume(right_hand, frame) if right_hand else None
                self.open_spotify(right_hand) if right_hand else None
                mouse_action = self.control_mouse(right_hand, frame) if right_hand else None

                # Check if the auto-brightness message should still be displayed
                if message_start_time and time.time() - message_start_time > 3:
                    auto_brightness_message = None
                    message_start_time = None

                # Create overlay with system details
                frame = self.create_overlay(frame, sys_info, auto_brightness_message)

                # Display the gesture alert at the top of the frame
                self.display_gesture_alert(frame)

                # Add exit instruction at the bottom of the frame
                exit_text = "Press 'q' to quit"
                auto_brightness_text = "Press 'b' to toggle auto-brightness"
                text_position = (10, self.frame_height - 30)
                auto_brightness_position = (10, self.frame_height - 10)
                # White outline for both lines
                cv2.putText(frame, exit_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, auto_brightness_text, auto_brightness_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Black text for both lines
                cv2.putText(frame, exit_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                cv2.putText(frame, auto_brightness_text, auto_brightness_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

                # Display the full frame
                cv2.imshow('Gesture Control', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Application exited by user.")
                    print("Exiting application...")
                    break
                elif key == ord('b'):
                    self.toggle_auto_brightness()
                    auto_brightness_message = f"Auto-Brightness: {'ON' if self.auto_brightness_enabled else 'OFF'}"
                    message_start_time = time.time()

            cap.release()
            cv2.destroyAllWindows()
        except RuntimeError as e:
            logging.error(str(e))
            print(str(e))
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    controller = GestureController()
    controller.run()