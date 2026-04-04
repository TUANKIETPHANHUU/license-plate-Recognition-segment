import cv2
import numpy as np
from skimage import measure
from imutils import perspective
import imutils

from src.data_utils import order_points, convert2Square, draw_labels_and_boxes
from src.lp_detection.detect import detectNumberPlate
from src.char_classification.model import CNN_Model
from skimage.filters import threshold_local

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

LP_DETECTION_CFG = {
    "weight_path": "./src/weights/weight.h5",  # Make sure this path is correct for your CNN weights
    "classes_path": "./src/lp_detection/cfg/yolo.names",
    "config_path": "./src/lp_detection/cfg/yolov3-tiny.cfg"
}

CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5' # This should likely point to the CNN model weights

class E2E(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))
        # Note: Added 'weight_path' to init to match previous version structure, ensure consistency
        self.detectLP = detectNumberPlate(LP_DETECTION_CFG['classes_path'], LP_DETECTION_CFG['config_path'], LP_DETECTION_CFG['weight_path'])
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
        self.candidates = []

    def extractLP(self):
        coordinates = self.detectLP.detect(self.image)
        if len(coordinates) == 0:
            return  # Changed from ValueError to return to match previous version logic better

        for coordinate in coordinates:
            yield coordinate

    def predict(self, image):
        # Input image or frame
        self.image = image

        for coordinate in self.extractLP():     # detect license plate by yolov3
            self.candidates = []

            # convert (x_min, y_min, width, height) to coordinate(top left, top right, bottom left, bottom right)
            pts = order_points(coordinate)

            # crop number plate used by bird's eyes view transformation
            LpRegion = perspective.four_point_transform(self.image, pts)
            
            # =========================================================
            # Margin Cropping (Already included to remove edge noise)
            # =========================================================
            h_lp, w_lp = LpRegion.shape[:2]
            
            # Xén khoảng 6% - 8% từ các cạnh vào tâm để vứt bỏ viền đen và ốc vít
            margin_x = int(w_lp * 0.07)  # Xén 7% chiều rộng trái/phải
            margin_y = int(h_lp * 0.07)  # Xén 7% chiều cao trên/dưới
            
            # Simplified margin application. Ensure margins don't exceed dimensions.
            start_y = max(0, margin_y)
            end_y = min(h_lp, h_lp - margin_y)
            start_x = max(0, margin_x)
            end_x = min(w_lp, w_lp - margin_x)
            
            if start_y >= end_y or start_x >= end_x: # If cropped image is invalid, use original
                LpRegion = LpRegion 
            else:
                LpRegion = LpRegion[start_y:end_y, start_x:end_x]
            # =========================================================

            # segmentation
            self.segmentation(LpRegion)

            # recognize characters
            self.recognizeChar()

            # format and display license plate
            license_plate = self.format()

            # draw labels
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

        return self.image

    def segmentation(self, LpRegion):
        if LpRegion.size == 0: return # Handle empty regions after margin crop

        # apply thresh to extracted licences plate
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]

        # adaptive threshold
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255

        # convert black pixel of digits to white pixel
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)

        # connected components analysis
        labels = measure.label(thresh, connectivity=2, background=0)

        # loop over the unique components
        for label in np.unique(labels):
            # if this is background label, ignore it
            if label == 0:
                continue

            # init mask to store the location of the character candidates
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            # find contours from mask
            # Note: OpenCV 4+ findContours returns 2 values. For compatibility with older OpenCV, added third return variable `_` and check.
            contours_output = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_output) == 3: # OpenCV 3 or specific version
                _, contours, hierarchy = contours_output
            else: # OpenCV 4 or standard version
                contours, hierarchy = contours_output

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                # rule to determine characters
                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])

                if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                    # extract characters
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    # Check for grayscale vs color to handle dimensions correctly in reshape. segmentation gives grayscale.
                    if len(square_candidate.shape) == 2:
                        square_candidate = square_candidate.reshape((28, 28, 1))
                    elif len(square_candidate.shape) == 3:
                        square_candidate = square_candidate.reshape((28, 28, 1)) # Segmentation gives grayscale, so keep it single channel
                    
                    self.candidates.append((square_candidate, (y, x)))

    def recognizeChar(self):
        characters = []
        coordinates = []

        for char, coordinate in self.candidates:
            characters.append(char)
            coordinates.append(coordinate)

        if not characters:
            return

        characters = np.array(characters)
        result = self.recogChar.predict_on_batch(characters)
        
        # --- MODIFICATION: ADD CONFIDENCE THRESHOLD CHECK ---
        result_prob = np.max(result, axis=1) # Get the maximum probability (confidence) for each prediction
        result_idx = np.argmax(result, axis=1) # Get the corresponding class index

        confidence_threshold = 0.90 # Define a threshold (90%). Predictions below this will be ignored.

        self.candidates = []
        for i in range(len(result_idx)):
            if result_idx[i] == 31:    # if is background or noise, ignore it
                continue
            
            # --- NEW CHECK: Discard if confidence is low ---
            if result_prob[i] < confidence_threshold:
                continue # Discard unsure prediction even if top class is not background
                
            self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))
        # --- END MODIFICATION ---

    def format(self):
        if not self.candidates:
            return "Không đọc được"

        first_line = []
        second_line = []

        # Assuming measure.label provides rough row sorting, we will still improve robustness.
        # Check if first candidate has a valid Y coordinate.
        if not self.candidates or len(self.candidates[0][1]) == 0:
             # Basic handling if sorting assumption fails
             for candidate, coordinate in self.candidates:
                 first_line.append((candidate, coordinate[1])) # Stick everyone in first line for now
        else:
             baseline_y = self.candidates[0][1][0]
             for candidate, coordinate in self.candidates:
                # Use a more robust check for line height
                line_height_threshold = 30 # Threshold for what constitutes a different line, can be adjusted
                if coordinate[0] < baseline_y + line_height_threshold:
                    first_line.append((candidate, coordinate[1]))
                else:
                    second_line.append((candidate, coordinate[1]))

        def take_second(s):
            if len(s[1]) < 2: return 0 # Handle empty or malformed coordinates
            return s[1] # Use X-coordinate for left-to-right sorting within a line

        # Sort left to right within each line.
        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)

        # Build final string, handling cases where lines might be missing
        license_plate = ""
        if first_line:
            license_plate += "".join([str(ele[0]) for ele in first_line])
        if second_line:
            if license_plate: license_plate += "-" # Add separator if first line exists
            license_plate += "".join([str(ele[0]) for ele in second_line])

        return license_plate