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

            # Crop và xoay thẳng biển số
            LpRegion = perspective.four_point_transform(self.image, pts)
            
            # Phân đoạn ký tự
            self.segmentation(LpRegion)

            # Nhận diện từng ký tự
            self.recognizeChar()

            # Định dạng và sắp xếp lại chuỗi biển số (Đã tích hợp NMS để chống lặp chữ)
            license_plate = self.format()

            # Vẽ kết quả lên ảnh gốc
            self.image = draw_labels_and_boxes(self.image, license_plate, coordinate)

        return self.image

    def segmentation(self, LpRegion):
        # FIX LỖI "UNKNOWN" Ô TÔ: Phóng to ảnh GỐC lên 400px ngay từ đầu để đồng bộ kích thước nét chữ
        LpRegion = imutils.resize(LpRegion, width=400)
        
        # Tiền xử lý ảnh để tách chữ (Kênh màu V trong HSV tốt cho biển số)
        V = cv2.split(cv2.cvtColor(LpRegion, cv2.COLOR_BGR2HSV))[2]
        T = threshold_local(V, 15, offset=10, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        
        # Chuyển chữ đen sang trắng trên nền đen, medianBlur để mượt nét
        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.medianBlur(thresh, 5)

        # FIX LỖI QUY CHUẨN: Lấy chiều cao ảnh đã resize để làm mốc chia tỷ lệ
        thresh_height = thresh.shape[0]

        # Phân tích các thành phần liên thông (Connected Components)
        labels = measure.label(thresh, connectivity=2, background=0)

        for label in np.unique(labels):
            if label == 0: continue

            mask = np.zeros(thresh.shape, dtype="uint8")
            mask[labels == label] = 255

            # Tương thích OpenCV mọi phiên bản để lấy 2 giá trị Contours cuối
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(contour)

                aspectRatio = w / float(h)
                solidity = cv2.contourArea(contour) / float(w * h)
                
                # SỬA LỖI: Lấy h chia cho thresh_height đã resize. Ô tô sẽ không còn ra Unknown.
                heightRatio = h / float(thresh_height)

                # BỘ LỌC HÌNH HỌC SIẾT CHẶT: heightRatio > 0.35 để dọn sạch ốc vít. solidity > 0.15 để bỏ mẩu nhiễu vỡ.
                if 0.15 < aspectRatio < 0.95 and solidity > 0.15 and 0.35 < heightRatio < 0.95:
                    candidate = np.array(mask[y:y + h, x:x + w])
                    square_candidate = convert2Square(candidate)
                    square_candidate = cv2.resize(square_candidate, (28, 28), cv2.INTER_AREA)
                    square_candidate = square_candidate.reshape((28, 28, 1))
                    # Lưu (ảnh, (tọa độ y, tọa độ x))
                    self.candidates.append((square_candidate, (y, x)))

    def recognizeChar(self):
        # Chặn lỗi crash array rỗng
        if len(self.candidates) == 0: return

        characters = []
        coordinates = []
        for char, coordinate in self.candidates:
            characters.append(char)
            coordinates.append(coordinate)

        characters = np.array(characters)
        # Chạy CNN nhận diện đồng thời toàn bộ ký tự
        result = self.recogChar.predict_on_batch(characters)
        
        # result_idx chứa class (chữ/số), confidences chứa xác suất của class đó
        result_idx = np.argmax(result, axis=1)
        confidences = np.max(result, axis=1) 

        self.candidates = []
        for i in range(len(result_idx)):
            # Bỏ qua nhãn nền (background class 31)
            if result_idx[i] == 31: continue
            
            # FIX LỖI CHỮ LẠ (E, U...): Màng lọc độ tự tin. Nếu CNN "ngáo" đoán bừa ra class E, U nhưng xác suất thấp, thì xóa luôn.
            # Confidence Threshold: 0.75 là con số an toàn
            if confidences[i] < 0.75:
                # Debug in ra терминал để biết AI đang băn khoăn cái gì
                # print(f"Đã lọc: class {ALPHA_DICT[result_idx[i]]} với confidence thấp {confidences[i]:.2f}")
                continue

            self.candidates.append((ALPHA_DICT[result_idx[i]], coordinates[i]))

    def format(self):
        # Chặn lỗi crash nếu array rỗng
        if len(self.candidates) == 0: return "Unknown"

        first_line = []
        second_line = []

        # Tọa độ Y của ký tự đầu tiên để làm mốc phân dòng
        base_y = self.candidates[0][1][0]

        # Phân ký tự vào dòng 1 và dòng 2 (cho xe máy VN)
        # Con số Y + 40 pixel là mốc chia dòng khi ảnh đã resize 400px
        for char, coordinate in self.candidates:
            if base_y + 40 > coordinate[0]:
                first_line.append((char, coordinate[1]))
            else:
                second_line.append((char, coordinate[1]))

        def take_second(s):
            return s[1]

        # Sắp xếp các ký tự theo dòng từ trái qua phải (trục X)
        first_line = sorted(first_line, key=take_second)
        second_line = sorted(second_line, key=take_second)

        # FIX LỖI DƯ SỐ (Double Detection): Áp dụng NMS tự chế. 
        # Nếu 2 ký tự liền kề cách nhau quá gần theo trục X (< 20 pixel), thì xóa 1 cái.
        def clean_line(line):
            if not line: return []
            res = [line[0]]
            for i in range(1, len(line)):
                # Giữ khoảng cách tối thiểu X > 20 pixel
                if abs(line[i][1] - res[-1][1]) > 20: 
                    res.append(line[i])
            return res

        first_line = clean_line(first_line)
        second_line = clean_line(second_line)

        # Ghép ký tự thành chuỗi
        str_1 = "".join([str(ele[0]) for ele in first_line])
        str_2 = "".join([str(ele[0]) for ele in second_line])

        if len(second_line) == 0:
            return str_1 # Biển ô tô dài hoặc biển vuông VN
        else:
            return f"{str_1}-{str_2}" # Biển vuông VN có dấu gạch ngang