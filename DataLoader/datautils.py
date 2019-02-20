#encoding:utf-8
import os
import torch
import glob as gb
import numpy as np
import cv2
import csv

from DataLoader.generate_label import Genetate_shrinkly_poly

def get_images(root):
    '''
    get images's path and name
    '''
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(gb.glob(os.path.join(root, '*.{}'.format(ext))))
    name = []
    for i in range(len(files)):
        name.append(files[i].split('/')[-1])
    return files, name


def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype = np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype = np.float32), np.array(text_tags, dtype = np.bool)


def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(im, polys, tags, crop_background = False, max_tries = 50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    h_array = np.zeros((h + pad_h * 2), dtype = np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype = np.int32)
    for poly in polys:
        poly = np.round(poly, decimals = 0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size = 2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size = 2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        # if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
        if xmax - xmin < 0.1 * w or ymax - ymin < 0.1 * h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis = 1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


# def shrink_poly(poly, r,shrink_ratio):
#     '''
#     fit a poly inside the origin poly, maybe bugs here...
#     used for generate the score map
#     :param poly: the text poly
#     :param r: r in the paper
#     :return: the shrinked poly
#     '''
#     # shrink ratio
#     R = shrink_ratio
#     # find the longer pair
#     if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
#             np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
#         # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
#         ## p0, p1
#         theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
#         poly[0][0] += R * r[0] * np.cos(theta)
#         poly[0][1] += R * r[0] * np.sin(theta)
#         poly[1][0] -= R * r[1] * np.cos(theta)
#         poly[1][1] -= R * r[1] * np.sin(theta)
#         ## p2, p3
#         theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
#         poly[3][0] += R * r[3] * np.cos(theta)
#         poly[3][1] += R * r[3] * np.sin(theta)
#         poly[2][0] -= R * r[2] * np.cos(theta)
#         poly[2][1] -= R * r[2] * np.sin(theta)
#         ## p0, p3
#         theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
#         poly[0][0] += R * r[0] * np.sin(theta)
#         poly[0][1] += R * r[0] * np.cos(theta)
#         poly[3][0] -= R * r[3] * np.sin(theta)
#         poly[3][1] -= R * r[3] * np.cos(theta)
#         ## p1, p2
#         theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
#         poly[1][0] += R * r[1] * np.sin(theta)
#         poly[1][1] += R * r[1] * np.cos(theta)
#         poly[2][0] -= R * r[2] * np.sin(theta)
#         poly[2][1] -= R * r[2] * np.cos(theta)
#     else:
#         ## p0, p3
#         # print poly
#         theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
#         poly[0][0] += R * r[0] * np.sin(theta)
#         poly[0][1] += R * r[0] * np.cos(theta)
#         poly[3][0] -= R * r[3] * np.sin(theta)
#         poly[3][1] -= R * r[3] * np.cos(theta)
#         ## p1, p2
#         theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
#         poly[1][0] += R * r[1] * np.sin(theta)
#         poly[1][1] += R * r[1] * np.cos(theta)
#         poly[2][0] -= R * r[2] * np.sin(theta)
#         poly[2][1] -= R * r[2] * np.cos(theta)
#         ## p0, p1
#         theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
#         poly[0][0] += R * r[0] * np.cos(theta)
#         poly[0][1] += R * r[0] * np.sin(theta)
#         poly[1][0] -= R * r[1] * np.cos(theta)
#         poly[1][1] -= R * r[1] * np.sin(theta)
#         ## p2, p3
#         theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
#         poly[3][0] += R * r[3] * np.cos(theta)
#         poly[3][1] += R * r[3] * np.sin(theta)
#         poly[2][0] -= R * r[2] * np.cos(theta)
#         poly[2][1] -= R * r[2] * np.sin(theta)
#     return poly
#
#
# def point_dist_to_line(p1, p2, p3):
#     # compute the distance from p3 to p1-p2
#     return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
#
#
# def fit_line(p1, p2):
#     # fit a line ax+by+c = 0
#     if p1[0] == p1[1]:
#         return [1., 0., -p1[0]]
#     else:
#         [k, b] = np.polyfit(p1, p2, deg = 1)
#         return [k, -1., b]
#
#
# def line_cross_point(line1, line2):
#     # line1 0= ax+by+c, compute the cross point of line1 and line2
#     if line1[0] != 0 and line1[0] == line2[0]:
#         print('Cross point does not exist')
#         return None
#     if line1[0] == 0 and line2[0] == 0:
#         print('Cross point does not exist')
#         return None
#     if line1[1] == 0:
#         x = -line1[2]
#         y = line2[0] * x + line2[2]
#     elif line2[1] == 0:
#         x = -line2[2]
#         y = line1[0] * x + line1[2]
#     else:
#         k1, _, b1 = line1
#         k2, _, b2 = line2
#         x = -(b1 - b2) / (k1 - k2)
#         y = k1 * x + b1
#     return np.array([x, y], dtype = np.float32)
#
#
# def line_verticle(line, point):
#     # get the verticle line from line across point
#     if line[1] == 0:
#         verticle = [0, -1, point[1]]
#     else:
#         if line[0] == 0:
#             verticle = [1, 0, -point[0]]
#         else:
#             verticle = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
#     return verticle
#
#
# def rectangle_from_parallelogram(poly):
#     '''
#     fit a rectangle from a parallelogram
#     :param poly:
#     :return:
#     '''
#     p0, p1, p2, p3 = poly
#     angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
#     if angle_p0 < 0.5 * np.pi:
#         if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
#             # p0 and p2
#             ## p0
#             p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
#             p2p3_verticle = line_verticle(p2p3, p0)
#
#             new_p3 = line_cross_point(p2p3, p2p3_verticle)
#             ## p2
#             p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
#             p0p1_verticle = line_verticle(p0p1, p2)
#
#             new_p1 = line_cross_point(p0p1, p0p1_verticle)
#             return np.array([p0, new_p1, p2, new_p3], dtype = np.float32)
#         else:
#             p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
#             p1p2_verticle = line_verticle(p1p2, p0)
#
#             new_p1 = line_cross_point(p1p2, p1p2_verticle)
#             p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
#             p0p3_verticle = line_verticle(p0p3, p2)
#
#             new_p3 = line_cross_point(p0p3, p0p3_verticle)
#             return np.array([p0, new_p1, p2, new_p3], dtype = np.float32)
#     else:
#         if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
#             # p1 and p3
#             ## p1
#             p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
#             p2p3_verticle = line_verticle(p2p3, p1)
#
#             new_p2 = line_cross_point(p2p3, p2p3_verticle)
#             ## p3
#             p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
#             p0p1_verticle = line_verticle(p0p1, p3)
#
#             new_p0 = line_cross_point(p0p1, p0p1_verticle)
#             return np.array([new_p0, p1, new_p2, p3], dtype = np.float32)
#         else:
#             p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
#             p0p3_verticle = line_verticle(p0p3, p1)
#
#             new_p0 = line_cross_point(p0p3, p0p3_verticle)
#             p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
#             p1p2_verticle = line_verticle(p1p2, p3)
#
#             new_p2 = line_cross_point(p1p2, p1p2_verticle)
#             return np.array([new_p0, p1, new_p2, p3], dtype = np.float32)
#
#
# def sort_rectangle(poly):
#     # sort the four coordinates of the polygon, points in poly should be sorted clockwise
#     # First find the lowest point
#     p_lowest = np.argmax(poly[:, 1])
#     if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
#         # 底边平行于X轴, 那么p0为左上角
#         p0_index = np.argmin(np.sum(poly, axis = 1))
#         p1_index = (p0_index + 1) % 4
#         p2_index = (p0_index + 2) % 4
#         p3_index = (p0_index + 3) % 4
#         return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
#     else:
#         # 找到最低点右边的点
#         p_lowest_right = (p_lowest - 1) % 4
#         p_lowest_left = (p_lowest + 1) % 4
#         angle = np.arctan(
#             -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
#         # assert angle > 0
#         if angle <= 0:
#             print(angle, poly[p_lowest], poly[p_lowest_right])
#         if angle / np.pi * 180 > 45:
#             # 这个点为p2
#             p2_index = p_lowest
#             p1_index = (p2_index - 1) % 4
#             p0_index = (p2_index - 2) % 4
#             p3_index = (p2_index + 1) % 4
#             return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
#         else:
#             # 这个点为p3
#             p3_index = p_lowest
#             p0_index = (p3_index + 1) % 4
#             p1_index = (p3_index + 2) % 4
#             p2_index = (p3_index + 3) % 4
#             return poly[[p0_index, p1_index, p2_index, p3_index]], angle
#
#
# def restore_rectangle_rbox(origin, geometry):
#     d = geometry[:, :4]
#     angle = geometry[:, 4]
#     # for angle > 0
#     origin_0 = origin[angle >= 0]
#     d_0 = d[angle >= 0]
#     angle_0 = angle[angle >= 0]
#     if origin_0.shape[0] > 0:
#         p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
#                       d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
#                       d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
#                       np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
#                       d_0[:, 3], -d_0[:, 2]])
#         p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2
#
#         rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
#         rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis = 1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2
#
#         rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
#         rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis = 1).reshape(-1, 2, 5).transpose((0, 2, 1))
#
#         p_rotate_x = np.sum(rotate_matrix_x * p, axis = 2)[:, :, np.newaxis]  # N*5*1
#         p_rotate_y = np.sum(rotate_matrix_y * p, axis = 2)[:, :, np.newaxis]  # N*5*1
#
#         p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis = 2)  # N*5*2
#
#         p3_in_origin = origin_0 - p_rotate[:, 4, :]
#         new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
#         new_p1 = p_rotate[:, 1, :] + p3_in_origin
#         new_p2 = p_rotate[:, 2, :] + p3_in_origin
#         new_p3 = p_rotate[:, 3, :] + p3_in_origin
#
#         new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
#                                   new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis = 1)  # N*4*2
#     else:
#         new_p_0 = np.zeros((0, 4, 2))
#     # for angle < 0
#     origin_1 = origin[angle < 0]
#     d_1 = d[angle < 0]
#     angle_1 = angle[angle < 0]
#     if origin_1.shape[0] > 0:
#         p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
#                       np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
#                       np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
#                       -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
#                       -d_1[:, 1], -d_1[:, 2]])
#         p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2
#
#         rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
#         rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis = 1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2
#
#         rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
#         rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis = 1).reshape(-1, 2, 5).transpose((0, 2, 1))
#
#         p_rotate_x = np.sum(rotate_matrix_x * p, axis = 2)[:, :, np.newaxis]  # N*5*1
#         p_rotate_y = np.sum(rotate_matrix_y * p, axis = 2)[:, :, np.newaxis]  # N*5*1
#
#         p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis = 2)  # N*5*2
#
#         p3_in_origin = origin_1 - p_rotate[:, 4, :]
#         new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
#         new_p1 = p_rotate[:, 1, :] + p3_in_origin
#         new_p2 = p_rotate[:, 2, :] + p3_in_origin
#         new_p3 = p_rotate[:, 3, :] + p3_in_origin
#
#         new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
#                                   new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis = 1)  # N*4*2
#     else:
#         new_p_1 = np.zeros((0, 4, 2))
#     return np.concatenate([new_p_0, new_p_1])
#
#
# def restore_rectangle(origin, geometry):
#     return restore_rectangle_rbox(origin, geometry)


def generate_rbox(im_size, polys, tags):
    num_feature = 6
    h, w = im_size
    score_map = np.zeros((num_feature,h, w), dtype = np.uint8)
    train_mask = np.ones((h,w),dtype = np.uint8)
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        if(poly_tag[1]):
            poly1 = np.array([poly], np.int)
            cv2.fillPoly(train_mask,poly1,0)
        for i in range(num_feature):
            shrinked_poly = Genetate_shrinkly_poly(poly.copy(),m=0.5,n=num_feature,i=i)
            cv2.fillPoly(score_map[i], shrinked_poly, 1)
    return score_map,train_mask

def collate_fn(batch):
    img, score_map,train_mask ,show_images= zip(*batch)
    bs = len(score_map)
    images = []
    score_maps = []
    show_imgs = []
    for i in range(bs):
        if img[i] is not None:
            a = torch.from_numpy(img[i])
            a = a.permute(2, 0, 1)
            images.append(a)
            a1 = torch.from_numpy(show_images[i])
            a1 = a1.permute(2, 0, 1)
            show_imgs.append(a1)
            b = torch.from_numpy(score_map[i])
            score_maps.append(b)
    images = torch.stack(images, 0)
    show_imgs = torch.stack(show_imgs,0)
    score_maps = torch.stack(score_maps, 0)
    train_masks = np.array(train_mask)

    return images, score_maps,train_masks,show_imgs

