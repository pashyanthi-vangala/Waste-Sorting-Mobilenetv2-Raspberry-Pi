import subprocess
import cv2
import numpy as np
import tensorflow as tf
import gpiod
import atexit
import signal
import sys
from gpiod.line import Bias, Direction, Edge
import time
import lgpio


# Model and classification settings
MODEL_PATH = "/home/cosmos/waste_classifier.keras"
CLASS_NAMES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
RECYCLABLE_CLASSES = ["Cardboard", "Glass", "Metal", "Paper", "Plastic"]
INPUT_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 60.0


# GPIO settings
BUTTON_PIN = 17
SERVO_PIN = 18

BUTTON_CHIP_PATH = "/dev/gpiochip0"
SERVO_CHIP_NUM = 0

chip_handle = lgpio.gpiochip_open(SERVO_CHIP_NUM)

try:
    lgpio.gpio_claim_output(chip_handle, SERVO_PIN)
except Exception:
    pass


def angle_to_us(angle):
    return int(1500 + (angle / 90.0) * 1000)


def set_servo_angle(angle):
    pulse_width_us = angle_to_us(angle)
    lgpio.tx_servo(chip_handle, SERVO_PIN, pulse_width_us, 50)


def disable_servo():
    try:
        lgpio.tx_servo(chip_handle, SERVO_PIN, 0, 0)
    except Exception:
        pass


disable_servo()
CURRENT_ANGLE = 0


def move_servo_smoothly(target_angle, step_delay=0.015):
    global CURRENT_ANGLE
    if target_angle == CURRENT_ANGLE:
        return

    step = 1 if target_angle > CURRENT_ANGLE else -1

    for angle in range(CURRENT_ANGLE, target_angle + step, step):
        set_servo_angle(angle)
        time.sleep(step_delay)

    CURRENT_ANGLE = target_angle


BIN_ANGLES = {
    "Cardboard": -60,
    "Glass": -60,
    "Metal": -60,
    "Paper": -60,
    "Plastic": -60,
    "Trash": 60
}

NEUTRAL_ANGLE = 0


# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")
print(f"Input shape: {model.input_shape}")

preview_process = None


def stop_live_preview():
    global preview_process

    if preview_process is not None:
        preview_process.terminate()

        try:
            preview_process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            preview_process.kill()
            preview_process.wait()

        preview_process = None

    subprocess.run(
        ["pkill", "-9", "rpicam-vid"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL
    )

    time.sleep(0.3)


def start_live_preview():
    global preview_process

    stop_live_preview()

    cmd = [
        "rpicam-vid",
        "-t", "0",
        "--width", "640",
        "--height", "480",
        "--inline"
    ]

    preview_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


atexit.register(stop_live_preview)


def handle_exit_signal(sig, frame):
    print("\nStopping...")
    stop_live_preview()

    try:
        disable_servo()
        lgpio.gpiochip_close(chip_handle)
    except Exception:
        pass

    sys.exit(0)


signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)


# Camera functions
def capture_frame_to_memory():
    cmd = [
        "rpicam-still",
        "-o", "-",
        "-t", "200",
        "-n",
        "--width", "640",
        "--height", "480",
        "-e", "jpg"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True
    )

    image_array = np.frombuffer(
        result.stdout,
        dtype=np.uint8
    )

    return cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


def sort_with_servo(final_bin_destination):
    target_angle = BIN_ANGLES.get(
        final_bin_destination,
        NEUTRAL_ANGLE
    )

    WAIT_BEFORE_TURN = 2.0
    print(f"Waiting {WAIT_BEFORE_TURN}s...")
    time.sleep(WAIT_BEFORE_TURN)

    print(f"Moving servo to {target_angle}°...")
    move_servo_smoothly(target_angle)

    HOLD_DURATION = 2.0
    time.sleep(HOLD_DURATION)

    print("Moving servo back...")
    move_servo_smoothly(NEUTRAL_ANGLE)
    time.sleep(0.5)

    disable_servo()


# Classification and sorting
def capture_and_classify():
    print("\n--- Button pressed ---")
    print("Taking photo...")

    stop_live_preview()

    bgr_frame = None

    for attempt in range(2):
        try:
            bgr_frame = capture_frame_to_memory()
            break
        except subprocess.CalledProcessError:
            print("Capture failed. Trying again...")
            time.sleep(0.2)

    if bgr_frame is None:
        print("Could not take photo.")
        start_live_preview()
        return

    rgb_frame = cv2.cvtColor(
        bgr_frame,
        cv2.COLOR_BGR2RGB
    )

    resized_rgb = cv2.resize(
        rgb_frame,
        INPUT_SIZE,
        interpolation=cv2.INTER_AREA
    )

    input_data = np.expand_dims(
        np.array(resized_rgb, dtype=np.float32),
        axis=0
    )

    print("Classifying...")

    predictions = model(
        input_data,
        training=False
    ).numpy()[0]

    top_index = np.argmax(predictions)
    raw_label = CLASS_NAMES[top_index]
    confidence = predictions[top_index] * 100

    print(f"Prediction: {raw_label}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nClass probabilities:")

    for name, score in zip(CLASS_NAMES, predictions):
        print(f"  {name}: {score * 100:.1f}%")

    if confidence < CONFIDENCE_THRESHOLD:
        final_destination = "Trash"
        print(
            f"Confidence is below {CONFIDENCE_THRESHOLD}%. "
            f"Sending to Trash."
        )
    else:
        final_destination = raw_label

        if raw_label in RECYCLABLE_CLASSES:
            print(f"Sending to recycling: {raw_label}")
        else:
            print("Sending to Trash")

    print(f"Final bin: {final_destination}")

    sort_with_servo(final_destination)
    start_live_preview()


# Button setup
line_config = {
    BUTTON_PIN: gpiod.LineSettings(
        direction=Direction.INPUT,
        bias=Bias.PULL_UP,
        edge_detection=Edge.FALLING
    )
}

print("\nSystem ready.")
print("Live camera preview is active.")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}%")
print("Press the button on GPIO 17 to classify an item.")

start_live_preview()

try:
    with gpiod.request_lines(
        BUTTON_CHIP_PATH,
        consumer="waste_classifier",
        config=line_config
    ) as request:

        while True:
            if request.wait_edge_events(timeout=0.2):
                request.read_edge_events()
                capture_and_classify()
                time.sleep(0.8)

                while request.wait_edge_events(timeout=0.01):
                    request.read_edge_events()

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stop_live_preview()

    try:
        disable_servo()
        lgpio.gpiochip_close(chip_handle)
    except Exception:
        pass