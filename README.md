# Gesture Control System

This project is a gesture control system that uses a webcam to detect hand gestures and perform various actions such as controlling media, adjusting brightness, scrolling, and interacting with applications like Spotify, YouTube, and WhatsApp.

### Features
- **Media Control**: Play, pause, next track, and previous track for both web and installed apps.
- **Brightness Adjustment**: Auto-brightness adjustment based on ambient light.
- **Mouse Control**: Smooth mouse pointer movement and left/right clicks using gestures.
- **Scrolling**: Scroll up and down using index and ring finger gestures.
- **Application Control**: Open Spotify, YouTube, and WhatsApp using gestures.
- **Customizable UI**: Circular overlay with system information (CPU, Memory, FPS).
- **Settings Menu**: Change pinky gesture actions via a settings menu.
- **Exit Gesture**: Exit the application by showing both palms with all fingers up.
- **Face Mask Feature**: Toggle a smiling face mask overlay on the user's face using the 'M' key.

### Requirements
- Python 3.7 or higher
- Webcam

### Installation

#### Using `requirements.txt`
Run the following command to install all dependencies:
```bash
pip install -r requirements.txt
```

#### Without `requirements.txt`
If `requirements.txt` is not available, install the required libraries manually:
```bash
pip install opencv-python mediapipe numpy pycaw pyautogui psutil screen-brightness-control comtypes Pillow
```

### Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gesture-control
   ```
2. Run the application:
   ```bash
   python run.py
   ```

### Controls
- **Press 'q'**: Quit the application.
- **Press 'b'**: Toggle auto-brightness mode.
- **Press 'm'**: Toggle the face mask feature (smiling face overlay).
- **Show both palms with all fingers up**: Exit the application.
- **Index and ring finger gestures**: Scroll up and down.
- **Index finger gesture**: Control mouse pointer movement.
- **Thumb and index finger pinch**: Left-click.
- **Thumb and middle finger pinch**: Right-click.
- **Pinky finger gesture**: Open Spotify, YouTube, or WhatsApp (customizable).
- **Hand swipe gesture**: Play, pause, next track, or previous track.
- **Thumb and index finger distance**: Adjust system volume.
- **Left hand pinky finger gesture**: Toggle YouTube fullscreen mode.

### Troubleshooting
- Ensure your webcam is connected and functioning properly.
- If you encounter issues with library installation, ensure you are using Python 3.7 or later.
- For Windows compatibility, ensure all required libraries are installed and up-to-date.

### Project Holder
This project is developed and maintained by **Bhanu Prakash**.
