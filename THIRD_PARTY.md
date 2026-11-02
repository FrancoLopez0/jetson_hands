# Model sources

The TensorRT backend uses MediaPipe models converted to ONNX by OpenCV Zoo,
under Apache License 2.0. Model downloads are pinned to OpenCV Zoo commit
`48b10fa505aeb51fbd127fc60a5c660040e823b7` and verified against the SHA256 hashes
published in that repository's Git LFS pointers.

- [Palm model, preprocessing reference and license](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/palm_detection_mediapipe)
- [Hand landmark model, preprocessing reference and license](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/handpose_estimation_mediapipe)
- [MediaPipe landmark-to-ROI graph](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/hand_landmark/hand_landmark_landmarks_to_roi.pbtxt)
- [MediaPipe tracking rectangle algorithm](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc)

The compact Python geometry implementation in this project uses the same
SSD anchor layout and palm/proximal-landmark tracking rectangle conventions.
No third-party application source is downloaded or executed by the setup script.
Downloaded ONNX models, generated engines and build products are excluded from
version control. Engines must be built for the deployment GPU/TensorRT version.
