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
            LpRegion = perspective.four_point_transform(self.image, pts)
            self.segmentation(LpRegion)
            self.recognizeChar()
            license_plate = self.format()
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)
        return self.image

    def segmentation(self, LpRegion):
        # Đồng bộ ảnh lên width=400px để các chỉ số phía sau chuẩn xác
        LpRegion = imutils.resize(LpRegion, width=400)
        img_h, img_w = LpRegion.shape[:2]
        
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.medianBlur(thresh, 5)

        # Xóa viền đen sát mép ảnh (Khắc phục lỗi nhận khung inox ra số 1-1)
        cv2.rectangle(thresh, (0, 0), (img_w - 1, img_h - 1), 0, 3)

        labels = measure.label(thresh, connectivity=2, background=0)
        char_candidates = []

        for label in np.unique(labels):
            if label == 0: continue
            
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)

                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(img_h)

                # Nới lỏng điều kiện chiều cao (0.2 -> 0.95) để cứu biển ô tô (Unknown)
                # Siết chặt aspectRatio để loại bỏ vết xước ngang/dọc
                if 0.1 < aspectRatio < 1.0 and solidity > 0.15 and 0.2 < heightRatio < 0.95 and h > 20:
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    
                    # LƯU TRỮ DẠNG DICTIONARY ĐỂ XỬ LÝ THÔNG MINH HƠN
                    char_candidates.append({
                        'img': square_candidate,
                        'x': x, 'y': y, 'w': w, 'h': h
                    })
        self.candidates = char_candidates

    def recognizeChar(self):
        if not self.candidates: return

        characters = np.array([c['img'] for c in self.candidates])
        result = self.recogChar.predict_on_batch(characters)
        
        result_idx = np.argmax(result, axis=1)
        confidences = np.max(result, axis=1)

        valid_candidates = []
        for i in range(len(result_idx)):
            # SIẾT CHẶT ĐỘ TỰ TIN LÊN 80% (0.80): Diệt sạch chữ E, U, Z, K sinh ra từ rác
            if result_idx[i] == 31 or confidences[i] < 0.80:
                continue
            
            c = self.candidates[i]
            c['char'] = ALPHA_DICT[result_idx[i]]
            c['conf'] = confidences[i]
            valid_candidates.append(c)

        self.candidates = valid_candidates

    def format(self):
        if not self.candidates: return "Unknown"

        # Phân chia 2 dòng động (An toàn tuyệt đối cho cả ô tô và xe máy)
        y_coords = [c['y'] for c in self.candidates]
        base_y = min(y_coords)
        y_threshold = base_y + 40 # Mốc 40px là lý tưởng cho ảnh đã resize 400px

        first_line = []
        second_line = []
        for c in self.candidates:
            if c['y'] < y_threshold:
                first_line.append(c)
            else:
                second_line.append(c)

        # THUẬT TOÁN NMS CHỐNG NHÂN BẢN KÝ TỰ (Chữa lỗi X4 -> X74, 2122 -> 2412122)
        def clean_line(line):
            if not line: return []
            # Sắp xếp từ trái qua phải theo tọa độ X
            line = sorted(line, key=lambda c: c['x'])
            res = []
            for current in line:
                if not res:
                    res.append(current)
                else:
                    prev = res[-1]
                    # KHI 2 KÝ TỰ DÍNH VÀO NHAU (Giao thoa > 50% chiều rộng)
                    if current['x'] < prev['x'] + prev['w'] * 0.5:
                        # CHỈ GIỮ LẠI KÝ TỰ CÓ ĐỘ TỰ TIN TỪ CNN CAO HƠN
                        if current['conf'] > prev['conf']:
                            res[-1] = current
                    else:
                        res.append(current)
            return res

        first_line = clean_line(first_line)
        second_line = clean_line(second_line)

        str_1 = "".join([c['char'] for c in first_line])
        str_2 = "".join([c['char'] for c in second_line])

        if not second_line:
            return str_1 # Ô tô 1 dòng
        return f"{str_1}-{str_2}" # Xe máy 2 dòng