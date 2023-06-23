import cv2
import pytesseract
import csv
import re
from datetime import datetime
import os
import sys
import json
import shutil
import time
import argparse
import textwrap
import logging

camera = 1
delay_time_sec = 10
coordinates_file = "rectangles_coordinates.json"
output_file = "data_captured.csv"

rectangles = []
current_rectangle = []
cropping = False
mouse_pos = (0, 0)

# Find the path to the Tesseract executable
tesseract_path = shutil.which("tesseract")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This function checks the Tesseract path and sets it
def check_tesseract_path():
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        logger.error(f"Could not find Tesseract executable at {pytesseract.pytesseract.tesseract_cmd}. Please check the path and try again.")
        raise Exception("Tesseract executable not found")

# Mouse click event for drawing and resizing rectangle
def click_and_crop(event, x, y, flags, param):
    global rectangles, current_rectangle, cropping, mouse_pos

    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        # If there is a rectangle and the click is inside it, start resizing
        if rectangles and rectangles[-1][0][0] < x < rectangles[-1][1][0] and rectangles[-1][0][1] < y < rectangles[-1][1][1]:
            cropping = False
            current_rectangle = list(rectangles[-1])
            rectangles = rectangles[:-1]
        else:
            current_rectangle = [(x, y)]
            cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # If resizing, update the second point of the rectangle
        if current_rectangle and not cropping:
            current_rectangle[1] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # If cropping, add the second point
        if cropping:
            current_rectangle.append((x, y))
            cropping = False
        rectangles.append(tuple(current_rectangle))


# Process the image and extract data from the region of interest
def process_image(frame, rect, debug=False, roi_id=0):
    roi = frame[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
    
    # Save original ROI image if debug is True
    if debug:
        cv2.imwrite(f'roi_{roi_id}_original.png', roi)

    # Resize the image
    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Save resized image if debug is True
    if debug:
        cv2.imwrite(f'roi_{roi_id}_resized.png', roi)
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Save grayscale image if debug is True
    if debug:
        cv2.imwrite(f'roi_{roi_id}_gray.png', gray)
    
    # Apply Gaussian blur to remove noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Save blurred image if debug is True
    if debug:
        cv2.imwrite(f'roi_{roi_id}_blurred.png', gray)
    
    # Threshold the image to get a binary image
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save threshold image if debug is True
    if debug:
        cv2.imwrite(f'roi_{roi_id}_threshold.png', threshold)
    
    # Set Tesseract config
    # custom_config = r'--oem 3 --psm 6 outputbase digits'
    custom_config = r'-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ., --psm 6'
    data = pytesseract.image_to_string(threshold, config=custom_config)

    return data

    

# Save the rectangles' coordinates to a file
def save_coordinates():
    with open(coordinates_file, "w") as file:
        json.dump(rectangles, file)

# Load the rectangles' coordinates from a file
def load_coordinates():
    global rectangles
    with open(coordinates_file, "r") as file:
        rectangles = json.load(file)


def convert_frequency(data):
    # Convert MHz to Hz
    data = re.sub(r'(\d+(\.\d+)?)(\s*)MHz', lambda match: str(float(match.group(1)) * 1e6), data)
    
    # Convert kHz to Hz
    data = re.sub(r'(\d+(\.\d+)?)(\s*)kHz', lambda match: str(float(match.group(1)) * 1e3), data)
    
    # Remove Hz
    data = re.sub(r'(\d+(\.\d+)?)(\s*)Hz', lambda match: match.group(1), data)

    return data


# Capture data from the camera and process the frames
def capture_data():
    while True:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            logger.error('Unable to open the camera')
            time.sleep(delay_time_sec)
            continue

        ret, frame = cap.read()
        if not ret:
            logger.error('Unable to read frame')
            cap.release()
            time.sleep(delay_time_sec)
            continue

        data_list = []
        for count, rect in enumerate(rectangles, 1):
            data = process_image(frame, rect, args.debug, roi_id=count)
            data = data.replace('\n', ' ').replace('\r', '')

            # Convert frequencies to Hz
            data = convert_frequency(data)

            logger.info(f'Data from ROI {count}: {data}')
            data_list.append(data)

        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.now()] + data_list)


        cap.release()
        time.sleep(delay_time_sec)

# Main function
def main():
    global rectangles, current_rectangle, cropping, mouse_pos

    if not os.path.exists(coordinates_file):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)

        cap = cv2.VideoCapture(camera)

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error('Unable to read frame')
                break

            # Display the drawn rectangles on the frame
            for rect in rectangles:
                cv2.rectangle(frame, tuple(rect[0]), tuple(rect[1]), (0, 255, 0), 2)
            if cropping and len(current_rectangle) == 1:
                cv2.rectangle(frame, tuple(current_rectangle[0]), mouse_pos, (0, 255, 0), 2)

            cv2.imshow("image", frame)
            key = cv2.waitKey(1) & 0xFF

            # Reset all rectangles if 'r' is pressed
            if key == ord("r"):
                rectangles = []

            # Break from the loop and save to file if 's' is pressed
            elif key == ord("s"):
                save_coordinates()
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        load_coordinates()

    capture_data()

if __name__ == "__main__":
    description = '''
    This script captures and recognizes digits
    of two screen regions and saves them in a CSV file.
    On the window selection screen:
    - Type 'r' to reset;
    - Type 's' to save;
    - Type 'q' to quit;
    '''

    parser = argparse.ArgumentParser(
        description=textwrap.dedent(description),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("-c", "--camera", type=int, default=1,
                        help="Camera source to use. Defaults to 1.")

    parser.add_argument("-d", "--delay", type=int, default=10,
                        help="Delay between frames in seconds. Defaults to 10.")

    parser.add_argument("-f", "--file", type=str, default="rectangles_coordinates.json",
                        help="Json file with coordinates. Defaults to \"rectangles_coordinates.json\".")

    parser.add_argument("-o", "--out", type=str, default="data_captured.csv",
                        help="Output file with csv. Defaults to \"data_captured.csv\".")

    parser.add_argument("-debug", "--debug", action='store_true',
                    help="Debug mode. If set, intermediate images will be saved during preprocessing.")


    args = parser.parse_args()

    # Show help when no parameters are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    camera = args.camera
    delay_time_sec = args.delay
    check_tesseract_path()
    main()
