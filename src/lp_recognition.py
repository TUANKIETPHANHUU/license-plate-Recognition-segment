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
        if len(coordinates) == 0:
            return 
        
        for coordinate in coordinates:
            yield coordinate

    def predict(self, image):
        self.image = image

        for coordinate in self.extractLP():
            self.candidates = []
            pts = order_points(coordinate)

            # Crop và xoay thẳng biển số
            LpRegion = perspective.four_point_transform(self.image, pts)
            
            # Phân đoạn ký tự
            self.segmentation(LpRegion)

            # Nhận diện từng ký tự
            self.recognizeChar()

            # Định dạng lại chuỗi biển số
            license_plate = self.format()

            # Vẽ kết quả lên ảnh gốc
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

        return self.image

    def segmentation(self, LpRegion):
        # CÁCH CHỮA BỆNH UNKNOWN Ô TÔ: Resize ảnh GỐC ngay từ đầu để đồng bộ tỷ lệ H/W
        LpRegion = imutils.resize(LpRegion, width=400)
        
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255

        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.medianBlur(thresh, 5)

        labels = measure.label(thresh, connectivity=2, background=0)

        for label in np.unique(labels):
            if label == 0:
                continue

            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            # CÁCH CHỮA BỆNH LẶP CHỮ: Dùng cv2.RETR_EXTERNAL để CHỈ LẤY VIỀN NGOÀI, bỏ qua viền tam giác bên trong các số 4, 8, 0...
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(LpRegion.shape[0])

                # Thu hẹp điều kiện để lọc rác tốt hơn
                if 0.15 < aspectRatio < 0.9 and solidity > 0.1 and 0.2 < heightRatio < 0.9:
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
        # CÁCH CHỮA BỆNH CHỮ LẠ (E, U...): Lấy độ tự tin (confidence)
        confidences = np.max(result, axis=1) 

        self.candidates = []
        for i in range(len(result_idx)):
            # NẾU LÀ BACKGROUND HOẶC ĐỘ TỰ TIN DƯỚI 75% -> LÀ RÁC -> BỎ QUA
            if result_idx[i] == 31 or confidences[i] < 0.75: 
                continue
            self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    def format(self):
        if len(self.candidates) == 0:
            return "Unknown"

        first_line = []
        second_line = []

        base_y = self.candidates[0][1][0]

        for candidate, coordinate in self.candidates:
            if base_y + 40 > coordinate[0]:
                first_line.append((candidate, coordinate[1]))
            else:
                second_line.append((candidate, coordinate[1]))

        def take_second(s):
            return s[1]

        # Hàm lọc trùng lặp tọa độ X (NMS tự chế)
        def filter_duplicates(line):
            if not line:
                return []
            line = sorted(line, key=take_second)
            res = [line[0]]
            for i in range(1, len(line)):
                # Nếu khoảng cách giữa 2 chữ cái gần nhau quá (< 15 pixel) thì tính là trùng -> Bỏ qua
                if abs(line[i][1] - res[-1][1]) > 15:
                    res.append(line[i])
            return res

        first_line = filter_duplicates(first_line)
        second_line = filter_duplicates(second_line)

        if len(second_line) == 0:
            license_plate = "".join([str(ele[0]) for ele in first_line])
        else:
            license_plate = "".join([str(ele[0]) for ele in first_line]) + "-" + "".join([str(ele[0]) for ele in second_line])

        return license_plate