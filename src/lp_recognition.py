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
        if len(coordinates) == 0: return
        for coordinate in coordinates:
            yield coordinate

    def predict(self, image):
        self.image = image
        for coordinate in self.extractLP():
            self.candidates = []
            pts = order_points(coordinate)
            LpRegion = perspective.four_point_transform(self.image, pts)
            
            # 1. CẮT VIỀN (MARGIN CROP): Loại bỏ hoa văn mép biển số
            h_lp, w_lp = LpRegion.shape[:2]
            mx, my = int(w_lp * 0.05), int(h_lp * 0.05)
            LpRegion = LpRegion[my:h_lp-my, mx:w_lp-mx]

            self.segmentation(LpRegion)
            self.recognizeChar()
            license_plate = self.format()
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)
        return self.image

    def segmentation(self, LpRegion):
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)

        labels = measure.label(thresh, connectivity=2, background=0)
        
        char_candidates = []
        for label in np.unique(labels):
            if label == 0: continue
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # 2. LỌC DIỆN TÍCH CỰC MẠNH: Loại bỏ dấu gạch ngang và đốm nhiễu
                # Chữ thật trên biển số sau khi resize lên 400px thường có diện tích > 600
                if area > 600: 
                    aspectRatio = w / float(h)
                    heightRatio = h / float(thresh.shape[0])
                    
                    # Chữ số thường có dáng đứng (w < h)
                    if 0.1 < aspectRatio < 0.85 and 0.4 < heightRatio < 0.95:
                        candidate = np.array(mask[y:y + h, x:x + w])
                        sq = convert2Square(candidate)
                        sq = cv2.resize(sq, (28, 28), cv2.INTER_AREA).reshape((28, 28, 1))
                        char_candidates.append((sq, (y, x)))
        
        self.candidates = char_candidates

    def recognizeChar(self):
        if not self.candidates: return
        chars = np.array([c[0] for c in self.candidates])
        preds = self.recogChar.predict_on_batch(chars)
        
        final_chars = []
        for i, prob in enumerate(preds):
            idx = np.argmax(prob)
            conf = np.max(prob)
            
            # 3. LỌC ĐỘ TIN CẬY (CONFIDENCE): Chỉ lấy nếu máy chắc chắn trên 85%
            if idx != 31 and conf > 0.85:
                final_chars.append((ALPHA_DICT[idx], self.candidates[i][1]))
        self.candidates = final_chars

    def format(self):
        if not self.candidates: return ""
        
        # Sắp xếp theo trục Y để chia dòng
        self.candidates.sort(key=lambda x: x[1][0])
        first_line, second_line = [], []
        
        # Ngưỡng chia dòng linh hoạt
        y_coords = [c[1][0] for c in self.candidates]
        mid_y = (max(y_coords) + min(y_coords)) / 2

        for char, pos in self.candidates:
            if pos[0] < mid_y: first_line.append((char, pos[1]))
            else: second_line.append((char, pos[1]))

        first_line.sort(key=lambda x: x[1])
        second_line.sort(key=lambda x: x[1])

        l1 = "".join([c[0] for c in first_line])
        l2 = "".join([c[0] for c in second_line])
        return f"{l1}-{l2}" if l2 else l1