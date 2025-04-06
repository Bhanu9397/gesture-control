**Gesture Control System**

This project is a gesture control system that uses a webcam to detect hand gestures and perform various actions such as controlling media, adjusting brightness, scrolling, and interacting with applications like Spotify, YouTube, and WhatsApp.

## Features
- **Media Control**: Play, pause, next track, and previous track for both web and installed apps.
- **Brightness Adjustment**: Auto-brightness adjustment based on ambient light.
- **Mouse Control**: Move the mouse pointer smoothly and perform left/right clicks using gestures.
- **Scrolling**: Scroll up and down using index and ring finger gestures.
- **Application Control**: Open Spotify, YouTube, and WhatsApp using gestures.
- **Customizable UI**: Circular overlay with system information (CPU, Memory, FPS).
- **Settings Menu**: Change pinky gesture actions via a settings menu.
- **Exit Gesture**: Exit the application by showing both palms with all fingers up.

## Requirements
- Python 3.7 or higher
- Webcam

## Installation

### Using `requirements.txt`
Run the following command to install all dependencies:
```bash
pip install -r requirements.txt
```

### Without `requirements.txt`
If `requirements.txt` is not available, install the required libraries manually:
```bash
pip install opencv-python mediapipe numpy pycaw pyautogui psutil screen-brightness-control comtypes
```

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gesture-control
   ```
2. Run the application:
   ```bash
   python run.py
   ```
3. Use the following gestures:
   - **Pinky Gesture**: Open Spotify, YouTube, or WhatsApp.
   - **Mouse Control**: Move the mouse with the index finger and perform left/right clicks.
   - **Media Control**: Use gestures for play, pause, next, and previous track.
   - **Brightness Control**: Auto-adjust brightness or toggle it manually with the `b` key.
   - **Scrolling**: Use the index finger to scroll up and the ring finger to scroll down.
   - **Exit Gesture**: Show both palms with all fingers up to exit the application.

## Notes
- Ensure the `settings_icon.png` file is placed in the `icons` folder.
- The application uses the `Roboto-Regular.ttf` font for better text clarity. Place the font file in the project directory.

## License
This project is licensed under the MIT License.