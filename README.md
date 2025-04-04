**Gesture Control System**

This project is a **Gesture Control System** that allows users to control various system functionalities such as volume, brightness, media playback, and mouse movements using hand gestures. It uses **OpenCV**, **Mediapipe**, and other libraries to detect and process hand gestures in real-time.

## Features
- **Volume Control**: Adjust system volume using hand gestures.
- **Brightness Control**: Automatically adjust screen brightness or toggle manual control.
- **Media Control**: Play, pause, and navigate media tracks.
- **Mouse Control**: Move the mouse pointer and perform left/right clicks.
- **Battery and System Info**: Display battery percentage, CPU usage, and memory usage in a circular overlay.
- **Dynamic Battery Ring**: Changes color based on battery percentage.
- **Hand Presence Detection**: Displays a message when no hands are detected.
- **Logging**: Logs system events and errors for debugging.

## Requirements
Install the required Python libraries using the following command:
```bash
pip install -r requirements.txt
```

### Required Libraries
The following libraries are required and are listed in the `requirements.txt` file:
- `opencv-python`
- `mediapipe`
- `numpy`
- `pycaw`
- `pyautogui`
- `psutil`
- `screen-brightness-control`

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Bhanu9397/gesture-control.git
   cd gesture-control
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python run.py
   ```

## Usage
- **Press 'q'**: Quit the application.
- **Press 'b'**: Toggle auto-brightness mode.

## Troubleshooting
- Ensure your webcam is connected and functioning properly.
- If you encounter issues with library installation, ensure you are using Python 3.7 or later.
- For Windows compatibility, ensure all required libraries are installed and up-to-date.

## Project Holder
This project is developed and maintained by **Bhanu Prakash**.