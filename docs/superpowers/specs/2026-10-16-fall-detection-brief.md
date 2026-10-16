# Project Brief: Real-Time Fall Detection Using MediaPipe Pose

## 1. Project Overview

Build a lightweight Computer Vision application that detects potential falls in elderly people using a pretrained human pose estimation model.

The application uses **MediaPipe Pose Landmarker** to extract human body landmarks from webcam streams or video files. It then analyzes body posture and movement over time using rule-based logic to identify potential falls.

The project must be implemented without training or fine-tuning any machine learning models. The goal is to create a functional, real-time prototype within approximately two hours.

## 2. Objectives

- Detect human body landmarks using a pretrained pose estimation model.
- Estimate body orientation and hip movement from pose landmarks.
- Detect potential falls by combining posture, movement speed, and temporal persistence.
- Display pose landmarks and detection status directly on the video.
- Minimize obvious false alarms caused by normal activities such as sitting, bending, and lying down.
- Keep the implementation simple, modular, and easy to demonstrate.

## 3. Technology Stack

- **Python**: Main programming language.
- **MediaPipe Pose Landmarker**: Pretrained human pose estimation.
- **OpenCV**: Webcam/video input, image processing, and visualization.
- **NumPy**: Numerical calculations for angles, distances, and velocities.

Do not train or fine-tune any models. Use an official pretrained MediaPipe Pose Landmarker model compatible with the installed MediaPipe API.

## 4. System Pipeline

1. Input: Read frames from a webcam or a local video file.
2. Pose estimation: Run MediaPipe Pose Landmarker to extract body landmarks.
3. Landmark validation: Check landmark visibility or confidence before using coordinates.
4. Feature extraction:
   - Body angle relative to the vertical axis.
   - Hip position and vertical displacement.
   - Hip movement velocity between consecutive frames.
   - Duration of a near-horizontal body posture.
5. Fall detection: Apply temporal rules to combine the extracted features.
6. Visualization: Draw the skeleton, display the current status, and show a potential-fall alert when the conditions are satisfied.

## 5. Fall Detection Logic

Implement a simple rule-based state machine with the following states:

- **NORMAL**: No potential fall detected.
- **POSSIBLE_FALL**: A rapid downward movement or a sudden change in body orientation is detected.
- **FALL_ALERT**: The person becomes nearly horizontal after a rapid movement, and the posture persists for a configurable duration.
- **RECOVERY**: The person returns to an upright posture or the alert is reset according to the defined rules.

Use configurable thresholds rather than hardcoded values scattered throughout the code.

Important considerations:

- Normalize coordinates before calculating velocities and distances.
- Use timestamps or actual frame intervals when calculating movement speed.
- Apply light smoothing to reduce landmark jitter.
- Handle missing or low-confidence landmarks safely.
- Use a short temporal window to avoid classifying a single unusual frame as a fall.
- Make thresholds configurable so they can be tuned using test videos.

These rules are a baseline, not a medically validated fall detector. A person lying down intentionally may still trigger an alert.

## 6. User Interface

Implement a simple OpenCV-based interface. A separate web frontend is not required.

The interface should display:

- Live camera feed or video playback.
- Pose skeleton overlay.
- Current detection state.
- Body angle and hip movement indicators, if available.
- A visually distinct warning when a potential fall is detected.
- Basic instructions for starting and stopping the application.

If practical, support switching between webcam input and a local video file through command-line arguments or a simple configuration setting.

## 7. Suggested Project Structure

```text
fall-detection/
├── main.py
├── fall_detector.py
├── pose_estimator.py
├── requirements.txt
├── README.md
├── models/
│   └── pose_landmarker.task
└── test_videos/
```

Responsibilities:

- `main.py`: Application entry point, video input, main loop, and visualization.
- `pose_estimator.py`: MediaPipe initialization and landmark extraction.
- `fall_detector.py`: Feature calculation, temporal rules, and state transitions.
- `requirements.txt`: Python dependencies.
- `README.md`: Setup, model download instructions, usage, limitations, and troubleshooting.
- `models/`: Store the pretrained model locally if required.
- `test_videos/`: Optional local videos for testing.

The exact structure may be simplified if that helps finish the prototype faster.

## 8. Testing Requirements

Test the application using available videos or safe, controlled demonstrations of:

1. Standing still.
2. Walking.
3. Sitting down.
4. Bending to pick up an object.
5. Lying down intentionally.
6. Simulating a fall safely, without actually falling.
7. Remaining in a horizontal posture after a rapid downward movement.
8. Temporarily losing reliable pose landmarks.

Record which cases trigger alerts and which do not. If a labeled dataset is unavailable, report qualitative results rather than claiming formal model accuracy.

## 9. Implementation Constraints

- Complete a functional prototype within approximately two hours.
- Do not train or fine-tune any models.
- Avoid unnecessary dependencies and complicated architectures.
- Prefer a working end-to-end implementation over advanced features.
- Keep the code readable, modular, and easy to run.
- Do not build a database, authentication system, cloud backend, or separate frontend unless explicitly requested.
- Do not assume a particular operating system or hardware configuration without checking the environment.

## 10. Deliverables

The final deliverable must include:

1. Working Python source code.
2. Dependency list and setup instructions.
3. Instructions for downloading and loading the pretrained model.
4. A runnable webcam or video demonstration.
5. Visible pose landmarks and real-time fall detection status.
6. A short explanation of the detection algorithm and configurable thresholds.
7. Known limitations and basic testing results.

## Definition of Done

The project is complete when the application can read a webcam stream or video, display human pose landmarks, calculate movement-related features, and raise a potential-fall alert based on temporal rules without requiring any model training.
