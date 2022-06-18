
import face_recognition
import numpy as np
import cv2

#旋转
def rotate_bound(image, angle):
    #中间点的位置
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    #进行仿射变换 cv2.warpAffine(变换的图像, 变化矩阵, (变化后的长与宽))
    return cv2.warpAffine(image, M, (nW, nH)), M


def get_random_color():
    return randrange(0, 255, 1), randrange(10, 255, 1), randrange(10, 255, 1)


LABELS = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
          'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
COLORS = [get_random_color() for _ in LABELS]

#人脸的特征点
def get_landmarks(img):
    module = hub.Module(name="face_landmark_localization")
    result = module.keypoint_detection(images=[img])
    if len(result) == 0:
        return None
    landmarks = result[0]['data'][0]
    return landmarks

#人脸倾斜角度
def get_face_rectangle(img):
    face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")
    result = face_detector.face_detection(images=[img])
    if len(result) == 0:
        return None, None, None, None
    x1 = int(result[0]['data'][0]['left'])
    y1 = int(result[0]['data'][0]['top'])
    x2 = int(result[0]['data'][0]['right'])
    y2 = int(result[0]['data'][0]['bottom'])
    return x1, y1, x2 - x1, y2 - y1


def face_landmarks(face_image, location_of_face=None):
    landmarks = get_landmarks(face_image)
    if landmarks is None:
        return None
    landmarks_as_tuples = [[(int(p[0]), int(p[1])) for p in landmarks]]
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] +
                      [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]


def get_bound_box(points):
    points = np.array(points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    return min_x, min_y, width, height


def check_if_mouth_open(img, points):
    middle_points = []
    middle_points.append(points["top_lip"][6])
    middle_points.append(points["top_lip"][7])
    middle_points.append(points["top_lip"][8])

    middle_points.append(points["bottom_lip"][len(points["bottom_lip"]) - 2])
    middle_points.append(points["bottom_lip"][len(points["bottom_lip"]) - 3])
    middle_points.append(points["bottom_lip"][len(points["bottom_lip"]) - 4])

    min_x, min_y, w, h = get_bound_box(middle_points)
    # cv2.rectangle(img, (min_x, min_y), (min_x + w, min_y + h), (255, 0, 0), 2)
    x, y, w_bottom_lip, h_bottom_lip = face_part(points, "bottom_lip")

    # cv2.rectangle(img, (x, y), (x + w_bottom_lip, y + h_bottom_lip), (255, 255, 0), 2)

    if h > h_bottom_lip / 2:
        return True
    return False


def face_part(points, part):
    assert part in LABELS, "face_part should be in [" + ','.join(LABELS) + ']'
    x, y, w, h = get_bound_box(points[part])
    return x, y, w, h


def calculate_angle(point1, point2):
    x1, x2, y1, y2 = point1[0], point2[0], point1[1], point2[1]
    return 180 / math.pi * math.atan((float(y2 - y1)) / (x2 - x1))


def get_top_left(img, sticker, landmarks, base_center, ratio, face_part, point_order, extra=[0, 0]):
    # check_if_mouth_open(img, landmarks[0])
    angle = calculate_angle(landmarks[0]['left_eyebrow'][0], landmarks[0]['right_eyebrow'][-1])
    nose_tip_center = base_center
    rotated, M = rotate_bound(sticker, angle)
    tip_center_rotate = np.dot(M, np.array([[nose_tip_center[0]], [nose_tip_center[1]], [1]]))
    sticker_h, sticker_w, _ = rotated.shape
    x, y, w, h = get_face_rectangle(img)
    if x is None:
        return None, None, None
    dv = w / sticker_w * ratio
    distance_x, distance_y = int(tip_center_rotate[0] * dv), int(tip_center_rotate[1] * dv)
    rotated = cv2.resize(rotated, (0, 0), fx=dv, fy=dv)
    if len(point_order) == 2:
        y_top_left = (landmarks[0][face_part[0]][point_order[0]][1] + landmarks[0][face_part[1]][point_order[1]][1]) // 2 - distance_y - extra[1]
        x_top_left = (landmarks[0][face_part[0]][point_order[0]][0] + landmarks[0][face_part[1]][point_order[1]][0]) // 2 - distance_x - extra[0]
    else:
        y_top_left = landmarks[0][face_part[0]][point_order[0]][1] - distance_y
        x_top_left = landmarks[0][face_part[0]][point_order[0]][0] - distance_x
    return y_top_left, x_top_left, rotated


def add_sticker(img, sticker_name, base_center, ratio, face_part, point_order, extra=[0, 0]):
    sticker = cv2.imread(sticker_name, -1)
    landmarks = face_landmarks(img)
    if landmarks is None:
        return img
    y_top_left, x_top_left, rotated = get_top_left(img, sticker, landmarks, base_center, ratio, face_part, point_order, extra)
    if y_top_left is None:
        return img
    sticker_h, sticker_w, _ = rotated.shape
    start = 0
    if y_top_left < 0:
        sticker_h = sticker_h + y_top_left
        start = -y_top_left
        y_top_left = 0

    for chanel in range(3):
        img[y_top_left:y_top_left + sticker_h, x_top_left:x_top_left + sticker_w, chanel] = \
            rotated[start:, :, chanel] * (rotated[start:, :, 3] / 255.0) + \
            img[y_top_left:y_top_left + sticker_h, x_top_left:x_top_left + sticker_w, chanel] \
            * (1.0 - rotated[start:, :, 3] / 255.0)

    return img


def add_sticker_ears(img):
    sticker_img = f'stickers/ears.png'
    center = [133, 150]
    face_part = ['left_eyebrow', 'right_eyebrow']
    point_order = [2, 2]
    img = add_sticker(img, sticker_img, center, 1.5, face_part, point_order)
    return img


def add_sticker_flowers(img):
    sticker_img = f'stickers/flowers.png'
    center = [251, 252]
    face_part = ['left_eyebrow', 'right_eyebrow']
    point_order = [2, 2]
    img = add_sticker(img, sticker_img, center, 1.5, face_part, point_order)
    return img


def add_sticker_hat(img):
    sticker_img = f'stickers/pirate.png'
    center = [300, 300]
    face_part = ['left_eyebrow', 'right_eyebrow']
    point_order = [2, 2]
    extra = [0, 25]
    img = add_sticker(img, sticker_img, center, 1.3, face_part, point_order, extra)
    return img


def add_sticker_mask(img):
    sticker_img = f'stickers/smallmask.png'
    center = [150, 80]
    face_part = ['left_eye', 'right_eye']
    point_order = [0, 3]
    img = add_sticker(img, sticker_img, center, 1.0, face_part, point_order)
    return img


def add_sticker_glasses(img):
    sticker_img = f'stickers/glasses.png'
    center = [360, 270]
    face_part = ['left_eye', 'right_eye']
    point_order = [0, 3]
    img = add_sticker(img, sticker_img, center, 1.0, face_part, point_order)
    return img


def add_sticker_ear_and_nose(img, sticker_name):
    stickers = {'cat': 'cat', 'mouse': 'mouse'}
    nose_center = {'cat': [180, 400], 'mouse': [208, 313]}
    sticker_img = f'stickers/{stickers[sticker_name]}.png'
    face_part = ['nose_tip']
    point_order = [2]
    img = add_sticker(img, sticker_img, nose_center[sticker_name], 1.0, face_part, point_order)
    return img



image_file = 'faces/01.jpg'
image = cv2.imread(image_file)
image = add_sticker_cat(image)
cv2.imshow('cat', image)
cv2.waitKey()