import cv2
import numpy as np
from skimage import measure
from imutils import perspective
import imutils

from src.data_utils import order_points, convert2Square, draw_labels_and_boxes
from src.lp_detection.detect import detectNumberPlate
from src.char_classification.model import CNN_Model
from skimage.filters import threshold_local

# Từ điển ánh xạ kết quả dự đoán từ CNN
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

            # Bước 1: Cắt và xoay thẳng vùng biển số
            LpRegion = perspective.four_point_transform(self.image, pts)
            
            # Bước 2: Phân đoạn (Segmentation) tách từng chữ cái
            self.segmentation(LpRegion)

            # Bước 3: Nhận dạng (OCR) bằng CNN
            self.recognizeChar()

            # Bước 4: Sắp xếp và định dạng chuỗi
            license_plate = self.format()

            # Bước 5: Vẽ khung và kết quả lên ảnh gốc
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

        return self.image

    def segmentation(self, LpRegion):
        # 1. TỐI ƯU: Phóng to ảnh lên 400px NGAY TỪ ĐẦU để đồng bộ kích thước nét chữ
        LpRegion = imutils.resize(LpRegion, width=400)
        
        # Tiền xử lý ảnh để làm nổi bật ký tự
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        
        # 2. CHỐNG VỠ NÉT: Tăng block_size lên 85 (Vì ảnh đã to 400px)
        T = threshold_local(V, 85, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.medianBlur(thresh, 5)

        thresh_height = thresh.shape[0]

        # Tìm các vùng liên thông (Ký tự)
        labels = measure.label(thresh, connectivity=2, background=0)
        raw_candidates = []

        for label in np.unique(labels):
            if label == 0: continue

            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                heightRatio = h / float(thresh_height)

                # BỘ LỌC HÌNH HỌC (Siết chặt)
                if 0.15 < aspectRatio < 0.9 and solidity > 0.2 and 0.25 < heightRatio < 0.95:
                    raw_candidates.append((x, y, w, h, mask))
        
        # 3. LỌC BÓNG MA NÂNG CAO (Xóa khung nhỏ bị nuốt chửng bởi khung lớn)
        filtered_candidates = []
        for i, (x1, y1, w1, h1, mask1) in enumerate(raw_candidates):
            is_valid = True
            for j, (x2, y2, w2, h2, mask2) in enumerate(raw_candidates):
                if i != j:
                    # Kiểm tra xem khung 1 có nằm lọt thỏm trong khung 2 không
                    if x1 >= x2 and (x1 + w1) <= (x2 + w2) and y1 >= y2 and (y1 + h1) <= (y2 + h2):
                        is_valid = False
                        break
            if is_valid:
                filtered_candidates.append((x1, y1, w1, h1, mask1))

        # 4. Đóng gói kết quả cuối cùng chuyển cho CNN
        for (x, y, w, h, mask) in filtered_candidates:
            candidate = np.array(mask[y:y + h, x:x + w])
            square_candidate = convert2Square(candidate)
            square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
            square_candidate = square_candidate.reshape((28, 28, 1))
            
            self.candidates.append((square_candidate, (y, x)))

    def recognizeChar(self):
        if len(self.candidates) == 0: return

        characters = [c[0] for c in self.candidates]
        coordinates = [c[1] for c in self.candidates]

        characters = np.array(characters)
        
        # CNN dự đoán
        result = self.recogChar.predict_on_batch(characters)
        result_idx = np.argmax(result, axis=1)
        confidences = np.max(result, axis=1) 

        self.candidates = []
        for i in range(len(result_idx)):
            if result_idx[i] == 31: continue # Bỏ qua background
            
            # BỘ LỌC ĐỘ TỰ TIN: Dưới 75% tin cậy -> Bỏ qua
            if confidences[i] < 0.75:
                print(f"[INFO] Lọc nhiễu bởi CNN: {ALPHA_DICT[result_idx[i]]} - Tự tin: {confidences[i]:.2f}")
                continue

            self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    def format(self):
        if len(self.candidates) == 0: return "Unknown"

        # Lấy danh sách tọa độ Y để xác định dòng
        y_coords = [c[1][0] for c in self.candidates]
        min_y = min(y_coords)
        max_y = max(y_coords)

        first_line, second_line = [], []
        
        # Nhận diện biển 2 dòng
        is_two_lines = (max_y - min_y) > 35

        if is_two_lines:
            y_mean = sum(y_coords) / len(y_coords)
            for char, (y, x) in self.candidates:
                if y < y_mean: first_line.append((char, x))
                else: second_line.append((char, x))
        else:
            for char, (y, x) in self.candidates:
                first_line.append((char, x))

        # Sắp xếp các ký tự trong mỗi dòng theo trục X (trái qua phải)
        first_line.sort(key=lambda item: item[1])
        second_line.sort(key=lambda item: item[1])

        # Xóa bóng ma lặp chữ (Khoảng cách X > 25)
        def clean_line(line):
            if not line: return []
            res = [line[0]]
            for i in range(1, len(line)):
                if abs(line[i][1] - res[-1][1]) > 25: 
                    res.append(line[i])
            return res

        first_line = clean_line(first_line)
        second_line = clean_line(second_line)

        # Ghép chuỗi
        str_1 = "".join([c[0] for c in first_line])
        str_2 = "".join([c[0] for c in second_line])

        if len(second_line) == 0:
            return str_1
        else:
            return f"{str_1}-{str_2}"