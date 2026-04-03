import src.data_utils as utils
import cv2
import numpy as np


class detectNumberPlate(object):
    def __init__(self, classes_path, config_path, weight_path, threshold=0.5):
        self.weight_path = weight_path
        self.cfg_path = config_path
        self.labels = utils.get_labels(classes_path)
        self.threshold = threshold

        # Load model YOLO
        self.model = cv2.dnn.readNet(model=self.weight_path, config=self.cfg_path)

    def detect(self, image):
        boxes = []
        classes_id = []
        confidences = []
        scale = 0.00392

        # Tiền xử lý ảnh đưa vào model
        blob = cv2.dnn.blobFromImage(image, scalefactor=scale, size=(416, 416), mean=(0, 0), swapRB=True, crop=False)
        height, width = image.shape[:2]

        # Đưa blob vào model
        self.model.setInput(blob)

        # Chạy forward để lấy kết quả (Hàm get_output_layers bạn đã sửa ở data_utils.py sẽ giúp chỗ này chạy mượt)
        outputs = self.model.forward(utils.get_output_layers(self.model))

        for output in outputs:
            for i in range(len(output)):
                scores = output[i][5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])

                if confidence > self.threshold:
                    # Tính toán tọa độ hộp bao (bounding boxes)
                    center_x = int(output[i][0] * width)
                    center_y = int(output[i][1] * height)

                    detected_width = int(output[i][2] * width)
                    detected_height = int(output[i][3] * height)

                    x_min = center_x - detected_width / 2
                    y_min = center_y - detected_height / 2

                    boxes.append([x_min, y_min, detected_width, detected_height])
                    classes_id.append(class_id)
                    confidences.append(confidence)

        # Áp dụng Non-Maximum Suppression (NMS) để lọc bớt các khung trùng nhau
        # Ở OpenCV mới, tham số phải là score_threshold thay vì chỉ threshold
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.threshold, nms_threshold=0.4)

        coordinates = []
        
        # Sửa lỗi IndexError: invalid index to scalar variable tại đây
        if len(indices) > 0:
            # Ép kiểu indices về mảng 1 chiều (flatten) bất kể phiên bản OpenCV nào
            if hasattr(indices, "flatten"):
                indices = indices.flatten()

            for i in indices:
                # Giờ đây i đã là một con số, không dùng i[0] nữa
                index = i 
                x_min, y_min, w, h = boxes[index]
                
                x_min = round(x_min)
                y_min = round(y_min)

                coordinates.append((x_min, y_min, w, h))

        return coordinates