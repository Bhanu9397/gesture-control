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
import os  # Added for checking file paths
from PIL import ImageFont, ImageDraw, Image  # Added for custom font rendering

# Configuration
VOLUME_RANGE = [10, 100]
MOUSE_SMOOTHING = 5
OVERLAY_ALPHA = 0.7  # Increased for better visibility
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
        self.auto_brightness_enabled = True
        self.current_brightness = 50

        # Initialize logging
        logging.basicConfig(filename="gesture_control.log", level=logging.INFO, 
                            format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info("Gesture Controller initialized.")

        # Hand presence tracking
        self.hands_present = False

        # Gesture alert tracking
        self.gesture_alert = None
        self.gesture_alert_time = None

        # Settings menu tracking
        self.settings_menu_open = False
        self.settings_button_coords = None

        # Pinky gesture action
        self.pinky_gesture_action = "Spotify"

        # Fullscreen tracking
        self.fullscreen_triggered = False
        self.fullscreen_state = False

    def toggle_auto_brightness(self):
        """Toggle the auto-brightness state."""
        self.auto_brightness_enabled = not self.auto_brightness_enabled

    def get_system_info(self):
        battery = psutil.sensors_battery()
        return {
            'battery': battery.percent if battery else "N/A",
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'fps': "N/A"
        }

    def auto_brightness(self, frame):
        """Adjust brightness automatically even when no hands are detected."""
        if not self.auto_brightness_enabled:
            return self.current_brightness

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            light_level = cv2.mean(gray)[0]
            brightness = np.interp(light_level, [0, 255], BRIGHTNESS_RANGE)
            current_brightness = sbc.get_brightness(display=0)[0]

            if abs(current_brightness - brightness) > 5:
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
            self.hands_present = True
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_type = results.multi_handedness[i].classification[0].label.lower()
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]
                hands[hand_type]['landmarks'] = landmarks

                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                x1, y1 = landmarks[2]
                x2, y2 = landmarks[5]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        else:
            self.hands_present = False

        return hands, frame

    def create_overlay(self, frame, sys_info, auto_brightness_message=None):
        """Enhanced circular overlay with modern design"""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Main circle with gradient effect
        center_x, center_y = 80, 80
        main_radius = 70
        cv2.circle(overlay, (center_x, center_y), main_radius, (30, 30, 30), -1)
        cv2.circle(overlay, (center_x, center_y), main_radius, (100, 100, 100), 2)

        # Battery ring
        battery = sys_info['battery'] if sys_info['battery'] != "N/A" else 0
        battery_angle = int((battery / 100) * 360)
        battery_color = (0, 255, 100) if battery >= 50 else (255, 100, 0)
        for angle in range(0, battery_angle, 2):
            x1 = int(center_x + main_radius * np.cos(np.radians(angle)))
            y1 = int(center_y + main_radius * np.sin(np.radians(angle)))
            x2 = int(center_x + (main_radius - 10) * np.cos(np.radians(angle)))
            y2 = int(center_y + (main_radius - 10) * np.sin(np.radians(angle)))
            cv2.line(overlay, (x1, y1), (x2, y2), battery_color, 2)

        # Brightness ring
        brightness_angle = int((self.current_brightness / 100) * 360)
        for angle in range(0, brightness_angle, 2):
            x1 = int(center_x + (main_radius - 20) * np.cos(np.radians(angle)))
            y1 = int(center_y + (main_radius - 20) * np.sin(np.radians(angle)))
            x2 = int(center_x + (main_radius - 30) * np.cos(np.radians(angle)))
            y2 = int(center_y + (main_radius - 30) * np.sin(np.radians(angle)))
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # System info text with better spacing
        details = [
            {"label": "CPU", "value": f"{sys_info['cpu']}%", "color": (255, 100, 100)},
            {"label": "MEM", "value": f"{sys_info['memory']}%", "color": (100, 255, 100)},
            {"label": "FPS", "value": f"{sys_info['fps']}", "color": (100, 200, 255)}
        ]
        text_y = height - 80
        for detail in details:
            cv2.putText(overlay, f"{detail['label']}: {detail['value']}", (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, detail['color'], 1, cv2.LINE_AA)
            text_y += 20

        # Settings button (gear icon)
        button_center = (center_x, center_y)
        button_radius = 20
        cv2.circle(overlay, button_center, button_radius, (80, 80, 80), -1)
        cv2.circle(overlay, button_center, button_radius, (150, 150, 150), 2)
        for angle in range(0, 360, 45):
            angle_rad = np.radians(angle)
            x1 = int(button_center[0] + button_radius * np.cos(angle_rad))
            y1 = int(button_center[1] + button_radius * np.sin(angle_rad))
            x2 = int(button_center[0] + (button_radius + 6) * np.cos(angle_rad))
            y2 = int(button_center[1] + (button_radius + 6) * np.sin(angle_rad))
            cv2.line(overlay, (x1, y1), (x2, y2), (150, 150, 150), 2)

        self.settings_button_coords = (button_center[0] - button_radius, button_center[1] - button_radius,
                                     button_radius * 2, button_radius * 2)

        # Auto-brightness message
        if auto_brightness_message:
            cv2.putText(overlay, auto_brightness_message, (center_x - 60, center_y + main_radius + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        return cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0)

    def show_settings_menu(self, frame):
        """Modern settings menu with better visuals"""
        if self.settings_menu_open:
            menu_x, menu_y = 50, 140
            menu_w, menu_h = 180, 100
            # Semi-transparent background with shadow
            overlay = frame.copy()
            cv2.rectangle(overlay, (menu_x-2, menu_y-2), (menu_x + menu_w+2, menu_y + menu_h+2), (50, 50, 50), -1)
            cv2.rectangle(overlay, (menu_x, menu_y), (menu_x + menu_w, menu_y + menu_h), (40, 40, 40), -1)
            frame[:] = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

            actions = ["Spotify", "WhatsApp", "YouTube"]
            for i, action in enumerate(actions):
                button_x, button_y = menu_x + 10, menu_y + 10 + i * 30
                button_w, button_h = menu_w - 20, 25
                
                color = (0, 200, 100) if action == self.pinky_gesture_action else (80, 80, 80)
                cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), color, -1)
                cv2.rectangle(frame, (button_x, button_y), (button_x + button_w, button_y + button_h), (150, 150, 150), 1)
                
                text_x = button_x + 10
                text_y = button_y + 18
                cv2.putText(frame, action, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                if not hasattr(self, 'settings_buttons'):
                    self.settings_buttons = []
                if len(self.settings_buttons) < len(actions):
                    self.settings_buttons.append((button_x, button_y, button_w, button_h, action))

    def open_spotify(self, hand):
        """Silently open apps or web versions with pinky gesture"""
        if not hand or len(hand) < 20:
            self.spotify_opened = False
            return

        pinky_up = hand[20][1] < hand[17][1]
        other_fingers_closed = all(hand[i][1] > hand[i-3][1] for i in [8, 12, 16])

        if pinky_up and other_fingers_closed and not self.spotify_opened:
            app_commands = {
                "Spotify": ("shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify", "https://open.spotify.com"),
                "WhatsApp": ("shell:AppsFolder\\5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App", "https://web.whatsapp.com"),
                "YouTube": ("shell:AppsFolder\\GoogleInc.YouTube_yourappid!App", "https://www.youtube.com")
            }

            app_cmd, web_url = app_commands[self.pinky_gesture_action]
            try:
                subprocess.Popen(f"start {app_cmd}", shell=True)
                self.spotify_opened = self.pinky_gesture_action.lower()
            except:
                subprocess.Popen(f"start {web_url}", shell=True)
                self.spotify_opened = f"{self.pinky_gesture_action.lower()}_web"

        elif not pinky_up and self.spotify_opened:
            # Do not close the app or page when the pinky finger is closed
            pass

    def display_gesture_alert(self, frame):
        """Improved gesture alert display"""
        if self.gesture_alert and time.time() - self.gesture_alert_time < 2:
            overlay = frame.copy()
            text_size = cv2.getTextSize(self.gesture_alert, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 20
            text_y = 40
            cv2.rectangle(overlay, (text_x-10, text_y-20), (text_x + text_size[0]+10, text_y+10), (40, 40, 40), -1)
            frame[:] = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
            cv2.putText(frame, self.gesture_alert, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        elif self.gesture_alert:
            self.gesture_alert = None

    def control_volume(self, hand, frame):
        """Control system volume using thumb and index finger distance"""
        try:
            if not hand or len(hand) < 9:
                return None
            thumb_tip, index_tip = hand[4], hand[8]

            # Prevent volume control during mouse movement
            index_mcp = hand[5]
            if index_tip[1] < index_mcp[1]:  # Index finger is up
                return None

            thumb_mcp = hand[2]
            if thumb_tip[1] < thumb_mcp[1]:
                return None

            length = hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
            length = max(VOLUME_RANGE[0], min(length, VOLUME_RANGE[1]))

            cv2.line(frame, thumb_tip, index_tip, (0, 255, 100), 2)
            vol = np.interp(length, [VOLUME_RANGE[0], VOLUME_RANGE[1]], [self.vol_min, self.vol_max])
            self.vol_history.append(vol)
            if len(self.vol_history) > 5:
                self.vol_history.pop(0)
            avg_vol = np.mean(self.vol_history)
            self.volume.SetMasterVolumeLevel(avg_vol, None)
            self.gesture_alert = f"Volume: {int(np.interp(avg_vol, [self.vol_min, self.vol_max], [10, 100]))}%"
            self.gesture_alert_time = time.time()
            return int(np.interp(avg_vol, [self.vol_min, self.vol_max], [10, 100]))
        except Exception as e:
            logging.error(f"Volume control error: {e}")
            return None

    def control_mouse(self, hand, frame):
        """Control mouse movement and clicks using hand gestures with smoother movement and better screen reach."""
        if not hand or len(hand) < 17:
            return None

        index_tip, middle_tip, ring_tip, pinky_tip = hand[8], hand[12], hand[16], hand[20]
        index_mcp, middle_mcp, ring_mcp, pinky_mcp = hand[5], hand[9], hand[13], hand[17]

        index_up = index_tip[1] < index_mcp[1]
        middle_up = middle_tip[1] < middle_mcp[1]
        ring_up = ring_tip[1] < ring_mcp[1]
        pinky_down = pinky_tip[1] > pinky_mcp[1]

        # Debugging: Draw indicators to verify detection
        cv2.circle(frame, index_tip, 5, (255, 255, 255), -1)  # White dot on index tip
        cv2.circle(frame, index_mcp, 5, (255, 255, 255), -1)  # White dot on index MCP

        if index_up and not middle_up and not ring_up and pinky_down and not self.left_click_triggered:
            screen_w, screen_h = pyautogui.size()
            mouse_x = np.interp(index_tip[0], (0, self.frame_width), (0, screen_w))
            mouse_y = np.interp(index_tip[1], (0, self.frame_height), (0, screen_h))
            self.mouse_history.append((mouse_x, mouse_y))
            if len(self.mouse_history) > MOUSE_SMOOTHING:
                self.mouse_history.pop(0)
            avg_x, avg_y = np.mean([x for x, _ in self.mouse_history]), np.mean([y for _, y in self.mouse_history])
            pyautogui.moveTo(max(0, min(avg_x, screen_w)), max(0, min(avg_y, screen_h)), duration=0.05)  # Reduced duration for smoother movement
            cv2.circle(frame, index_tip, 10, (255, 0, 0), -1)

        if index_up and middle_up and not ring_up and pinky_down and not self.left_click_triggered:
            pyautogui.click(button='left')
            self.left_click_triggered = True
            self.gesture_alert = "LEFT CLICK"
            self.gesture_alert_time = time.time()
            return "LEFT"
        elif not (index_up and middle_up and not ring_up and pinky_down):
            self.left_click_triggered = False

        if index_up and middle_up and ring_up and pinky_down and not self.right_click_triggered:
            pyautogui.click(button='right')
            self.right_click_triggered = True
            self.gesture_alert = "RIGHT CLICK"
            self.gesture_alert_time = time.time()
            return "RIGHT"
        elif not (index_up and middle_up and ring_up and pinky_down):
            self.right_click_triggered = False

        if not index_up and not middle_up and not ring_up and not pinky_down:
            return None

        return None

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
        is_youtube_active = any('youtube' in proc.info['name'].lower() for proc in psutil.process_iter(['name']) if proc.info['name'])

        if next_gesture and not self.next_triggered:
            if is_ppt_open:
                pyautogui.press('right')
                action = "NEXT SLIDE"
            elif is_youtube_active:
                pyautogui.press('down')  # Navigate to next YouTube Short
                action = "NEXT SHORT"
            else:
                self.control_web_media("next", is_shorts=False)
                action = "NEXT TRACK"
            self.next_triggered = True
            self.media_cooldown = time.time()
            self.gesture_alert = action
            self.gesture_alert_time = time.time()
            return action
        elif not next_gesture:
            self.next_triggered = False

        if prev_gesture and not self.prev_triggered:
            if is_ppt_open:
                pyautogui.press('left')
                action = "PREVIOUS SLIDE"
            elif is_youtube_active:
                pyautogui.press('up')  # Navigate to previous YouTube Short
                action = "PREVIOUS SHORT"
            else:
                self.control_web_media("previous", is_shorts=False)
                action = "PREVIOUS TRACK"
            self.prev_triggered = True
            self.media_cooldown = time.time()
            self.gesture_alert = action
            self.gesture_alert_time = time.time()
            return action
        elif not prev_gesture:
            self.prev_triggered = False

        if play_gesture and self.last_media_state != 'play':
            self.send_media_key(VK_MEDIA_PLAY_PAUSE)
            self.last_media_state = 'play'
            self.media_cooldown = time.time()
            self.gesture_alert = "PLAY"
            self.gesture_alert_time = time.time()
            return "PLAY"
        elif pause_gesture and self.last_media_state != 'pause':
            self.send_media_key(VK_MEDIA_PLAY_PAUSE)
            self.last_media_state = 'pause'
            self.media_cooldown = time.time()
            self.gesture_alert = "PAUSE"
            self.gesture_alert_time = time.time()
            return "PAUSE"

        return None

    def send_media_key(self, key_code):
        """Send a media key event to the system."""
        windll.user32.keybd_event(key_code, 0, 0, 0)
        windll.user32.keybd_event(key_code, 0, 2, 0)

    def control_youtube_fullscreen(self, hand):
        """Toggle YouTube fullscreen mode using pinky gesture."""
        if not hand or len(hand) < 20:
            return

        pinky_up = hand[20][1] < hand[17][1]
        other_fingers_down = all(hand[i][1] > hand[i-3][1] for i in [8, 12, 16])

        if pinky_up and other_fingers_down and not self.fullscreen_triggered:
            pyautogui.press('F')
            self.fullscreen_triggered = True
            self.gesture_alert = "Full Screen On" if not self.fullscreen_state else "Full Screen Off"
            self.fullscreen_state = not self.fullscreen_state
            self.gesture_alert_time = time.time()
        elif not pinky_up:
            self.fullscreen_triggered = False

    def control_web_media(self, action, is_shorts=False):
        """Control web media playback or navigation."""
        if action == "play_pause":
            pyautogui.press('Space')
        elif action == "next":
            if is_shorts:
                pyautogui.press('Down')
            else:
                pyautogui.hotkey('ctrl', 'Right')
        elif action == "previous":
            if is_shorts:
                pyautogui.press('Up')
            else:
                pyautogui.hotkey('ctrl', 'Left')

    def control_scroll(self, hand, frame):
        """Control scrolling with ring and index finger gestures, with even higher scroll values."""
        try:
            if not hand or len(hand) < 17:
                return None

            # Get landmark positions
            thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = hand[4], hand[8], hand[12], hand[16], hand[20]
            thumb_mcp, index_mcp, middle_mcp, ring_mcp, pinky_mcp = hand[2], hand[5], hand[9], hand[13], hand[17]

            # Scroll Down: Ring finger down, other fingers up
            ring_down = ring_tip[1] > ring_mcp[1]
            other_fingers_up = all(tip[1] < mcp[1] for tip, mcp in [(thumb_tip, thumb_mcp),
                                                                    (index_tip, index_mcp),
                                                                    (middle_tip, middle_mcp),
                                                                    (pinky_tip, pinky_mcp)])

            # Scroll Up: Index finger down, other fingers up
            index_down = index_tip[1] > index_mcp[1]
            other_fingers_up_for_index = all(tip[1] < mcp[1] for tip, mcp in [(thumb_tip, thumb_mcp),
                                                                              (middle_tip, middle_mcp),
                                                                              (ring_tip, ring_mcp),
                                                                              (pinky_tip, pinky_mcp)])

            if ring_down and other_fingers_up:
                pyautogui.scroll(-60)  # Further increased scroll amount for faster scrolling
                cv2.circle(frame, ring_tip, 10, (255, 0, 255), 2)  # Magenta circle outline for scroll down
                self.gesture_alert = "Scroll Down"
                self.gesture_alert_time = time.time()
                return "Down"
            elif index_down and other_fingers_up_for_index:
                pyautogui.scroll(60)  # Further increased scroll amount for faster scrolling
                cv2.circle(frame, index_tip, 10, (255, 255, 0), 2)  # Yellow circle outline for scroll up
                self.gesture_alert = "Scroll Up"
                self.gesture_alert_time = time.time()
                return "Up"

            return None
        except Exception as e:
            logging.error(f"Scroll control error: {e}")
            return None

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for settings menu interaction"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.settings_menu_open and hasattr(self, 'settings_buttons'):
                for button_x, button_y, button_w, button_h, action in self.settings_buttons:
                    if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
                        self.pinky_gesture_action = action
            if self.settings_button_coords:
                button_x, button_y, button_w, button_h = self.settings_button_coords
                if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
                    self.settings_menu_open = not self.settings_menu_open

    def handle_slider_input(self, frame, x, y, is_click):
        """Handle slider input for settings menu."""
        menu_x, menu_y, menu_w, menu_h = 50, 150, 200, 100
        if is_click:
            actions = ["Spotify", "WhatsApp", "YouTube"]
            for i, action in enumerate(actions):
                if menu_x + 10 <= x <= menu_x + menu_w - 10 and menu_y + 60 + i * 20 <= y <= menu_y + 80 + i * 20:
                    self.pinky_gesture_action = action

    def draw_settings_button(self, frame):
        """Draw the settings button on the frame."""
        button_center = (100, 100)
        button_radius = 20
        cv2.circle(frame, button_center, button_radius, (200, 200, 200), -1)
        cv2.circle(frame, button_center, button_radius, (0, 0, 0), 2)
        for angle in range(0, 360, 45):
            angle_rad = np.radians(angle)
            x1 = int(button_center[0] + button_radius * np.cos(angle_rad))
            y1 = int(button_center[1] + button_radius * np.sin(angle_rad))
            x2 = int(button_center[0] + (button_radius + 8) * np.cos(angle_rad))
            y2 = int(button_center[1] + (button_radius + 8) * np.sin(angle_rad))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.circle(frame, button_center, button_radius - 8, (255, 255, 255), -1)
        return (button_center[0] - button_radius, button_center[1] - button_radius, button_radius * 2, button_radius * 2)

    def check_settings_button_click(self, x, y, button_x, button_y, button_w, button_h):
        """Check if the settings button was clicked."""
        return button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h

    def is_powerpoint_open(self):
        """Check if Microsoft PowerPoint is currently running."""
        for process in psutil.process_iter(['name']):
            if process.info['name'] and process.info['name'].lower() == 'powerpnt.exe':
                return True
        return False

    def detect_exit_gesture(self, left_hand, right_hand):
        """Detect exit gesture when both hands show all fingers up."""
        if not left_hand or not right_hand:
            return False

        # Check if all fingers are up for both hands
        def all_fingers_up(hand):
            return all(hand[i][1] < hand[i - 3][1] for i in [8, 12, 16, 20])

        left_palm_up = all_fingers_up(left_hand)
        right_palm_up = all_fingers_up(right_hand)

        return left_palm_up and right_palm_up

    def run(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logging.error("Webcam not detected.")
                raise RuntimeError("Error: Webcam not detected.")

            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cv2.namedWindow('Gesture Control', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Gesture Control', 400, 300)
            screen_width = pyautogui.size().width
            cv2.moveWindow('Gesture Control', screen_width - 410, 10)
            cv2.setWindowProperty('Gesture Control', cv2.WND_PROP_TOPMOST, 1)

            frame_skip = 2
            frame_count = 0
            auto_brightness_message = None
            message_start_time = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logging.error("Unable to read from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                cv2.setMouseCallback('Gesture Control', self.mouse_callback, param=frame)

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                hands_data, frame = self.process_hands(frame)
                brightness = self.auto_brightness(frame)
                sys_info = self.get_system_info()
                fps = 1 / (time.time() - self.prev_time)
                self.prev_time = time.time()
                sys_info['fps'] = int(fps) if fps > 0 else "N/A"

                left_hand = hands_data['left']['landmarks']
                right_hand = hands_data['right']['landmarks']

                # Detect exit gesture
                if self.detect_exit_gesture(left_hand, right_hand):
                    logging.info("Exit gesture detected. Exiting application.")
                    break

                self.control_media(left_hand) if left_hand else None
                self.control_volume(right_hand, frame) if right_hand else None
                self.control_scroll(right_hand, frame) if right_hand else None
                self.open_spotify(right_hand) if right_hand else None
                self.control_youtube_fullscreen(left_hand) if left_hand else None
                self.control_mouse(right_hand, frame) if right_hand else None

                if message_start_time and time.time() - message_start_time > 3:
                    auto_brightness_message = None
                    message_start_time = None

                frame = self.create_overlay(frame, sys_info, auto_brightness_message)
                self.show_settings_menu(frame)
                self.display_gesture_alert(frame)

                cv2.putText(frame, "Q: Quit | B: Auto-Brightness", (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow('Gesture Control', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logging.info("Application exited by user.")
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