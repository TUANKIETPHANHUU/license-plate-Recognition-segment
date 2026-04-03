import cv2
import numpy as np
from skimage import measure
from imutils import perspective
import imutils
from skimage.filters import threshold_local
import traceback  # THÊM THƯ VIỆN NÀY ĐỂ TRUY VẾT LỖI

from src.data_utils import order_points, convert2Square, draw_labels_and_boxes
from src.lp_detection.detect import detectNumberPlate
from src.char_classification.model import CNN_Model

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

LP_DETECTION_CFG = {
    "weight_path": "./src/weights/yolov3-tiny_15000.weights",
    "classes_path": "./src/lp_detection/cfg/yolo.names",
    "config_path": "./src/lp_detection/cfg/yolov3-tiny.cfg"
}

CHAR_CLASSIFICATION_WEIGHTS = './src/weights/weight.h5'

class E2E(object):
    def __init__(self):
        self.image = np.empty((28, 28, 1))
        self.detectLP = detectNumberPlate(LP_DETECTION_CFG['classes_path'], LP_DETECTION_CFG['config_path'], LP_DETECTION_CFG['weight_path'])
        self.recogChar = CNN_Model(trainable=False).model
        self.recogChar.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
        self.candidates = []

    def extractLP(self):
        coordinates = self.detectLP.detect(self.image)
        
        # 1. Bỏ qua nếu không có gì
        if coordinates is None or len(coordinates) == 0:
            return

        # 2. Cưỡng chế ép kiểu: Fix dứt điểm lỗi tách số lẻ (scalar)
        try:
            coords_array = np.array(coordinates)
            # Nếu nó là mảng 1 chiều (ví dụ: [x, y, w, h] hoặc tuple (x, y, w, h))
            if coords_array.ndim == 1 and len(coords_array) == 4:
                coordinates = [coordinates] # Gói nó vào thành 1 list 2D
        except Exception:
            pass

        # 3. Trả về toạ độ
        for coordinate in coordinates:
            # Lọc chốt chặn cuối: Nếu coordinate là 1 con số (int/float), bỏ qua ngay!
            if isinstance(coordinate, (int, float, np.integer, np.floating)):
                continue
            yield coordinate

    def predict(self, image):
        try:
            # Copy ảnh gốc để không ghi đè
            self.image = image.copy()

            for coordinate in self.extractLP():
                self.candidates = []

                # convert (x_min, y_min, width, height) to coordinate
                pts = order_points(coordinate)

                # crop number plate
                LpRegion = perspective.four_point_transform(self.image, pts)
                
                # segmentation
                self.segmentation(LpRegion)

                # Tránh lỗi mảng rỗng nếu ảnh mờ không tìm thấy chữ
                if len(self.candidates) == 0:
                    continue

                # recognize characters
                self.recognizeChar()

                # Tránh lỗi nếu lọc nhiễu xong không còn chữ nào
                if len(self.candidates) == 0:
                    continue

                # format and display license plate
                license_plate = self.format()

                # draw labels
                self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

            return self.image
            
        except Exception as e:
            # Bắt tận tay lỗi: Sinh ra mã traceback chi tiết dòng gây lỗi
            error_msg = f"Lỗi: {e}\nChi tiết:\n{traceback.format_exc()}"
            print(error_msg) # In ra terminal
            raise Exception(error_msg) # Gửi lỗi này ra ngoài app.py để hiển thị

    def segmentation(self, LpRegion):
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)

        labels = measure.label(thresh, connectivity=2, background=0)

        for label in np.unique(labels):
            if label == 0:
                continue

            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])

                if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.35 < heightRatio < 2.0:
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    self.candidates.append((square_candidate, (y, x)))

    def recognizeChar(self):
        if len(self.candidates) == 0:
            return

        characters = []
        coordinates = []

        for char, coordinate in self.candidates:
            characters.append(char)
            coordinates.append(coordinate)

        characters = np.array(characters)
        result = self.recogChar.predict_on_batch(characters)
        result_idx = np.argmax(result, axis=1)

        self.candidates = []
        for i in range(len(result_idx)):
            if result_idx[i] == 31: 
                continue
            self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    def format(self):
        first_line = []
        second_line = []

        for candidate, coordinate in self.candidates:
            if self.candidates[0][1][0] + 40 > coordinate[0]:
                first_line.append((candidate, coordinate[1]))
            else:
                second_line.append((candidate, coordinate[1]))

        def take_second(s):
            return s[1]

        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)

        if len(second_line) == 0: 
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:
            license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])

        return license_plate