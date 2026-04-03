import numpy as np
import cv2

def get_digits_data(path):
    """
    Load digit data from .npy file, shuffle and return as list.
    """
    data = np.load(path, allow_pickle=True)
    np.random.shuffle(data)
    data_train = [d for d in data]

    print("-------------DONE------------")
    print('The number of train digits data: ', len(data_train))

    return data_train


def get_alphas_data(path):
    """
    Load alphabet data from .npy file, shuffle and return as list.
    """
    data = np.load(path, allow_pickle=True)
    np.random.shuffle(data)
    data_train = [d for d in data]

    print("-------------DONE------------")
    print('The number of train alphas data: ', len(data_train))

    return data_train


def get_labels(path):
    """
    Load labels from a text file, one label per line.
    """
    with open(path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def draw_labels_and_boxes(image, label, boxes):
    """
    Draw bounding box and label text on image.
    """
    x_min = round(boxes[0])
    y_min = round(boxes[1])
    x_max = round(boxes[0] + boxes[2])
    y_max = round(boxes[1] + boxes[3])

    # Vẽ khung hình chữ nhật quanh biển số
    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
    
    # Viết chữ label lên ảnh
    image = cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0, color=(0, 0, 255), thickness=2)
    return image


def get_output_layers(model):
    """
    Lấy tên các layer đầu ra của mô hình YOLO.
    Sửa lỗi "IndexError: invalid index to scalar variable" cho mọi bản OpenCV.
    """
    layers_name = model.getLayerNames()
    layer_indices = model.getUnconnectedOutLayers()

    # Xử lý trường hợp layer_indices là số đơn lẻ (scalar) hoặc mảng 2D
    if hasattr(layer_indices, "flatten"):
        layer_indices = layer_indices.flatten()

    # OpenCV cũ (3.x) dùng 1-based index, OpenCV mới (4.x+) dùng 0-based index
    # Đoạn code này tự động kiểm tra và lấy đúng tên layer
    output_layers = []
    for i in layer_indices:
        try:
            # Thử theo kiểu 1-based (cũ)
            output_layers.append(layers_name[i - 1])
        except:
            # Nếu lỗi thì lấy trực tiếp theo 0-based (mới)
            output_layers.append(layers_name[i])

    return output_layers


def order_points(coordinates):
    """
    Convert x, y, width, height to 4 corner points in order:
    top-left, top-right, bottom-left, bottom-right
    """
    rect = np.zeros((4, 2), dtype="float32")
    x_min, y_min, width, height = coordinates

    rect[0] = [round(x_min), round(y_min)]
    rect[1] = [round(x_min + width), round(y_min)]
    rect[2] = [round(x_min), round(y_min + height)]
    rect[3] = [round(x_min + width), round(y_min + height)]

    return rect


def convert2Square(image):
    """
    Resize non-square image to square by padding with zeros.
    """
    img_h, img_w = image.shape[:2]

    if img_h > img_w:
        diff = img_h - img_w
        x1 = np.zeros((img_h, diff // 2, 3), dtype=image.dtype)
        x2 = np.zeros((img_h, diff - diff // 2, 3), dtype=image.dtype)
        # Kiểm tra nếu ảnh xám thì bỏ chiều thứ 3
        if len(image.shape) == 2:
            x1 = np.zeros((img_h, diff // 2), dtype=image.dtype)
            x2 = np.zeros((img_h, diff - diff // 2), dtype=image.dtype)
        squared_image = np.concatenate((x1, image, x2), axis=1)
    elif img_w > img_h:
        diff = img_w - img_h
        x1 = np.zeros((diff // 2, img_w, 3), dtype=image.dtype)
        x2 = np.zeros((diff - diff // 2, img_w, 3), dtype=image.dtype)
        if len(image.shape) == 2:
            x1 = np.zeros((diff // 2, img_w), dtype=image.dtype)
            x2 = np.zeros((diff - diff // 2, img_w), dtype=image.dtype)
        squared_image = np.concatenate((x1, image, x2), axis=0)
    else:
        squared_image = image

    return squared_image