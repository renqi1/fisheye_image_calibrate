import os.path

import cv2
import numpy as np
from tqdm import tqdm


def imshow(image, name=""):
    cv2.imshow(name, image)
    cv2.waitKey(0)


def caculate_radius(image, show=True):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]

    _, binary = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
    if show:
        imshow(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if show:
        imshow(closing)

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_edge = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # epsilon = 0.001*cv2.arcLength(contour_edge, True)
    # contour_edge = cv2.approxPolyDP(contour_edge, epsilon, True)

    valid_inds = (contour_edge[:, 0, 0] > 3) & (contour_edge[:, 0, 0] < w - 3) & (contour_edge[:, 0, 1] > 3) & (
            contour_edge[:, 0, 1] < h - 3)
    contour_edge_valid = contour_edge[valid_inds, :]

    center, radius = cv2.minEnclosingCircle(contour_edge_valid)

    if show:
        draw_img = image.copy()
        res = cv2.drawContours(draw_img, [contour_edge], -1, (0, 0, 255), 2)
        imshow(res)
        draw_img = image.copy()
        cv2.circle(draw_img, (int(center[0]), int(center[1])), int(radius), (255, 255, 255), 2)
        res = cv2.drawContours(draw_img, [contour_edge_valid], -1, (0, 0, 255), 2)
        imshow(res)

    return center, radius


def get_useful_area(image, center, radius, show=False):
    h, w = image.shape[:2]
    cx, cy, r = int(center[0]), int(center[1]), int(radius)

    new_image = np.zeros([2 * r, 2 * r, 3], dtype=np.uint8)
    new_image[r - cy:r - cy + h, r - cx:r - cx + w, :] = image
    if show:
        imshow(new_image)

    mask = np.zeros_like(new_image, dtype=np.uint8)
    mask = cv2.circle(mask, (r, r), int(r), (1, 1, 1), -1)

    image_useful = new_image * mask
    if show:
        draw_img = image_useful.copy()
        cv2.circle(draw_img, (r, r), int(r), (255, 255, 255), 2)
        imshow(draw_img)

    return image_useful


def calibrate(image, method='longitude', show=False, useful_area=True, save_npx=False, load_npx=False, save_img=True):
    assert method in ('longitude', 'latitude', 'unfold')
    if useful_area:
        c, r = caculate_radius(image, show=show)
        image = get_useful_area(image, c, r, show=show)
    if image.shape[0] != image.shape[1]:
        raise ValueError('Image width isn\'t equal to height!')
    exist_npx = os.path.exists('./npy/mapx_{}.npy'.format(method))
    if load_npx and exist_npx:
        print('loading map...')
        mapx = np.load('./npy/mapx_{}.npy'.format(method))
        mapy = np.load('./npy/mapy_{}.npy'.format(method))
    elif method == 'longitude':
        print('longitude calibrate...')
        R = image.shape[0] // 2
        mapx = np.zeros([2 * R, 2 * R], dtype=np.float32)
        mapy = np.zeros([2 * R, 2 * R], dtype=np.float32)
        for j in tqdm(range(mapx.shape[0])):
            for i in range(mapx.shape[1]):
                mapx[j, i] = (i - R) / R * (R ** 2 - (j - R) ** 2) ** 0.5 + R
                mapy[j, i] = j
    elif method == 'latitude':
        print("latitude calibrate...")
        R = image.shape[0] // 2
        mapx = np.zeros([2 * R, 2 * R], dtype=np.float32)
        mapy = np.zeros([2 * R, 2 * R], dtype=np.float32)
        for j in tqdm(range(mapx.shape[0])):
            for i in range(mapx.shape[1]):
                mapx[j, i] = i
                mapy[j, i] = (j - R) / R * (R ** 2 - (i - R) ** 2) ** 0.5 + R
    elif method == 'unfold':
        print("unfold calibrate")
        R = image.shape[0] // 2
        W = int(2 * np.pi * R)
        H = R
        mapx = np.zeros([H, W], dtype=np.float32)
        mapy = np.zeros([H, W], dtype=np.float32)
        for j in tqdm(range(mapx.shape[0])):
            for i in range(mapx.shape[1]):
                angle = i / W * np.pi * 2
                radius = H - j
                mapx[j, i] = R + np.sin(angle) * radius
                mapy[j, i] = R - np.cos(angle) * radius

    if save_npx and (not exist_npx or not load_npx):
        np.save('./npy/mapx_{}.npy'.format(method), mapx)
        np.save('./npy/mapy_{}.npy'.format(method), mapy)
        print("save map successfully !")

    image_remap = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if show:
        imshow(image_remap)

    if save_img:
        cv2.imwrite('./result/{}.jpg'.format(method), image_remap)
        print("save remapped image successfully !")

    return image_remap


def calibrate_improve(image, method='longitude', k=1.0, show=False, save_npx=False, load_npx=False, save_img=True, k1=0.7, k2=0.2):
    assert method in ('longitude', 'latitude', 'union')
    h, w = image.shape[:2]

    if method == 'union':
        exist_npx = os.path.exists('./npy/mapx_{}_k{}_{}.npy'.format(method, k1, k2))
    else:
        exist_npx = os.path.exists('./npy/mapx_{}_k{}.npy'.format(method, k))
    if load_npx and exist_npx:
        print('loading map...')
        if method == 'union':
            mapx = np.load('./npy/mapx_{}_k{}_{}.npy'.format(method, k1, k2))
            mapy = np.load('./npy/mapy_{}_k{}_{}.npy'.format(method, k1, k2))
        else:
            mapx = np.load('./npy/mapx_{}_k{}.npy'.format(method, k))
            mapy = np.load('./npy/mapy_{}_k{}.npy'.format(method, k))
    elif method == 'longitude':
        print('improved longitude calibrate...')
        mapx = np.zeros([h, w], dtype=np.float32)
        mapy = np.zeros([h, w], dtype=np.float32)
        for j in tqdm(range(mapx.shape[0])):
            for i in range(mapx.shape[1]):
                mapx[j, i] = ((2 * i - w) / w) * ((w / 2) ** 2 - k*(j - h / 2) ** 2) ** 0.5 + w / 2
                mapy[j, i] = j
    elif method == 'latitude':
        print('improved latitude calibrate...')
        mapx = np.zeros([h, w], dtype=np.float32)
        mapy = np.zeros([h, w], dtype=np.float32)
        for j in tqdm(range(mapx.shape[0])):
            for i in range(mapx.shape[1]):
                mapx[j, i] = i
                mapy[j, i] = ((2 * j - h) / h) * ((h / 2) ** 2 - k*(i - w / 2) ** 2) ** 0.5 + h / 2

    elif method == 'union':
        print('union calibrate...')
        mapx = np.zeros([h, w], dtype=np.float32)
        mapy = np.zeros([h, w], dtype=np.float32)
        for j in tqdm(range(mapx.shape[0])):
            for i in range(mapx.shape[1]):
                mapx[j, i] = ((2 * i - w) / w) * ((w / 2) ** 2 - k1*(j - h / 2) ** 2) ** 0.5 + w / 2
                mapy[j, i] = ((2 * j - h) / h) * ((h / 2) ** 2 - k2*(i - w / 2) ** 2) ** 0.5 + h / 2
        k = '{}_{}'.format(k1, k2)

    if save_npx and (not exist_npx or not load_npx):
        np.save('./npy/mapx_{}_k{}.npy'.format(method, k), mapx)
        np.save('./npy/mapy_{}_k{}.npy'.format(method, k), mapy)
        print("save map successfully !")

    image_remap = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if show:
        imshow(image_remap)

    if save_img:
        cv2.imwrite('./result/{}_k{}.jpg'.format(method, k), image_remap)
        print("save remapped image successfully !")

    return image_remap

def calibrate_longitude_latitude(image, k1, k2, reverse=False, show=False, save_img=True):
    if reverse:
        remap1 = calibrate_improve(image, method='latitude', k=k2, show=show, load_npx=True, save_npx=True, save_img=False)
        remap2 = calibrate_improve(remap1, method='longitude', k=k1, show=show, load_npx=True, save_npx=True, save_img=False)
    else:
        remap1 = calibrate_improve(image, method='longitude', k=k1, show=show, load_npx=True, save_npx=True, save_img=False)
        remap2 = calibrate_improve(remap1, method='latitude', k=k2, show=show, load_npx=True, save_npx=True, save_img=False)
    if save_img:
        if reverse:
            save_path = './result/latitude_k{}_longitude_k{}.jpg'.format(k2, k1)
        else:
            save_path = './result/longitude_k{}_latitude_k{}.jpg'.format(k1, k2)
        cv2.imwrite(save_path, remap2)
        print("save remapped image successfully !")
    return remap2

if __name__ == "__main__":
    image = cv2.imread('woodscape/FV/00000_FV.png')

    # --------------------------------------------------------有效区域提取--------------------------------------------
    # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    # c, r = caculate_radius(image, show=True)
    # image_useful = get_useful_area(image, c, r, show=True)

    # --------------------------------------------------------经纬校正和横向展开--------------------------------------------
    # calibrate(image, method='longitude', load_npx=True, save_npx=True)
    # calibrate(image, method='latitude', load_npx=True, save_npx=True)
    # calibrate(image, method='unfold', load_npx=True, save_npx=True)

    # --------------------------------------------------------改善经度校正--------------------------------------------
    # calibrate_improve(image, method='longitude', k=1.75, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='longitude', k=1, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='longitude', k=0.9, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='longitude', k=0.8, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='longitude', k=0.7, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='longitude', k=0.6, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='longitude', k=0.5, load_npx=True, save_npx=True)

    # --------------------------------------------------------改善纬度校正--------------------------------------------
    # calibrate_improve(image, method='latitude', k=0.56, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='latitude', k=0.5, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='latitude', k=0.4, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='latitude', k=0.3, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='latitude', k=0.2, load_npx=True, save_npx=True)
    # calibrate_improve(image, method='latitude', k=0.1, load_npx=True, save_npx=True)

    # --------------------------------------------------------双纬度校正--------------------------------------------
    # calibrate_longitude_latitude(image, k1=0.8, k2=0.4, reverse=False, show=False)
    # calibrate_improve(image, method='union', k1=0.7, k2=0.2, load_npx=True, save_npx=True)


