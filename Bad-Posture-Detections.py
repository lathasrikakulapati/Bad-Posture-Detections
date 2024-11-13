
import cv2
import numpy as np
from PIL import Image
import tensorflow.lite as tflite
import gradio as gr
import math

# Define key point indices based on the PoseNet model output
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12

def load_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def process_image(interpreter, image, input_index):
    input_data = np.expand_dims(image, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    offset_data = np.squeeze(interpreter.get_tensor(output_details[1]['index']))

    points = []
    total_row, total_col, total_points = output_data.shape

    for k in range(total_points):
        max_score = output_data[0][0][k]
        max_row = 0
        max_col = 0
        for row in range(total_row):
            for col in range(total_col):
                if output_data[row][col][k] > max_score:
                    max_score = output_data[row][col][k]
                    max_row = row
                    max_col = col
        points.append((max_row, max_col))

    positions = []
    for idx, point in enumerate(points):
        pos_y, pos_x = point
        offset_x = offset_data[pos_y][pos_x][idx + 17]
        offset_y = offset_data[pos_y][pos_x][idx]
        positions.append((pos_x, pos_y, offset_x, offset_y))

    return positions

def display_result(positions, frame):
    size = 5
    color = (255, 0, 0)
    thickness = 3

    width = frame.shape[1]
    height = frame.shape[0]

    for pos in positions:
        pos_x, pos_y, offset_x, offset_y = pos
        x = int(pos_x / 8 * width + offset_x)
        y = int(pos_y / 8 * height + offset_y)
        cv2.circle(frame, (x, y), size, color, thickness)

    return frame

def calculate_angle(p1, p2):
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

def is_bad_posture(positions):
    # Extract key points
    left_shoulder = positions[LEFT_SHOULDER]
    right_shoulder = positions[RIGHT_SHOULDER]
    left_hip = positions[LEFT_HIP]
    right_hip = positions[RIGHT_HIP]

    # Calculate the midpoint of the shoulders and hips
    shoulder_midpoint = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    )
    hip_midpoint = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    )

    # Calculate the trunk flexion angle
    trunk_angle = calculate_angle(hip_midpoint, shoulder_midpoint)

    # Normalize the trunk angle to the range [0, 180]
    trunk_angle = abs(trunk_angle)
    if trunk_angle > 90:
        trunk_angle = 180 - trunk_angle

    # Classify the trunk flexion angle
    if trunk_angle <= 20:
        return "Low-risk posture"
    elif 20 < trunk_angle <= 60:
        return "Medium-risk posture"
    else:
        return "High-risk posture"

def predict_posture(image):
    model_path = 'data/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
    interpreter = load_model(model_path)

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    input_index = input_details[0]['index']

    frame = np.array(image)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image = Image.fromarray(frame_rgb).resize((width, height))
    positions = process_image(interpreter, np.array(resized_image), input_index)

    posture_risk = is_bad_posture(positions)

    if posture_risk != "Low-risk posture":
        cv2.putText(frame, posture_risk, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, posture_risk, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    result_image = display_result(positions, frame)
    return result_image, posture_risk

# Gradio interface
iface = gr.Interface(
    fn=predict_posture,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.Text()],
    title="Posture Detection",
    description="Upload an image to detect if the posture is low, medium, or high-risk."
)

if __name__ == "__main__":
    iface.launch()
