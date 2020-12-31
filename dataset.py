import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import os
import time
import pickle


def read_image(path, color=False):
    if color:
        image = np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), dtype=np.float64)
    else:
        image = np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), dtype=np.float64)
    return image


def show_image(image, title=None, cmap=None, textbox=None):
    if cmap is not None:
        plt.imshow(np.uint8(image),cmap=cmap)
    else:
        plt.imshow(np.uint8(image))
    if title is not None:
        plt.title(title)
    # if textbox is not None:
    #     plt.text(textbox[0], textbox[1], str(textbox[2]))
    plt.axis('off')
    plt.show()


def image_to_standard(img):
    return (img - np.mean(img))/np.std(img)


def calc_integral_image(img):
    return np.cumsum(np.cumsum(img, axis=1), axis=0)


def calc_rec(integral_img, x, y, width, height):
    x_point_sum, y_point_sum = x+width-1, y+height-1

    if x_point_sum-width < 0:
        empty_left = True
    else:
        empty_left = False

    if y_point_sum-height < 0:
        empty_top = True
    else:
        empty_top = False

    d = integral_img[x_point_sum, y_point_sum]
    if not empty_top:
        c = integral_img[x_point_sum, y_point_sum-height]
    else:
        c = 0
    if not empty_left:
        b = integral_img[x_point_sum-width, y_point_sum]
    else:
        b = 0
    if not empty_left and not empty_top:
        a = integral_img[x_point_sum-width, y_point_sum-height]
    else:
        a = 0
    return d - c - b + a


def calc_primitive(image, dict_boxes=False,types=(1,2,3,4,5)):
    idx_boxes = {}
    features = []
    integral_img = calc_integral_image(image)
    all_x = [x for x in range(1, image.shape[1] + 1)]
    all_y = [y for y in range(1, image.shape[0] + 1)]
    global_s = 0
    if 1 in types:
        s = 0
        for x, y in itertools.product(all_x, all_y):
            all_w = [w for w in range(1, (image.shape[1] - x + 1) // 2 + 1)]
            all_h = [h for h in range(1, image.shape[0] - y + 2)]
            for w, h in itertools.product(all_w, all_h):
                features.append(
                    calc_rec(integral_img, x - 1, y - 1, w, h) - calc_rec(integral_img, x + w - 1, y - 1, w, h))
                s += 1

                idx_boxes[global_s] = (1, x, y, w, h)
                global_s += 1
        # print(f'Features type 1: {s}')
    if 2 in types:
        s = 0
        for x, y in itertools.product(all_x, all_y):
            all_h = [h for h in range(1, (image.shape[1] - y + 1) // 2 + 1)]
            all_w = [w for w in range(1, image.shape[0] - x + 2)]
            for w, h in itertools.product(all_w, all_h):
                features.append(
                    calc_rec(integral_img, x - 1, y - 1, w, h) - calc_rec(integral_img, x - 1, y + h - 1, w, h))
                s += 1
                idx_boxes[global_s] = (2, x, y, w, h)
                global_s += 1
        # print(f'Features type 2: {s}')

    if 3 in types:
        s = 0
        for x, y in itertools.product(all_x, all_y):
            all_w = [w for w in range(1, (image.shape[1] - x + 1) // 3 + 1)]
            all_h = [h for h in range(1, image.shape[0] - y + 2)]
            for w, h in itertools.product(all_w, all_h):
                features.append(
                    calc_rec(integral_img, x - 1, y - 1, w, h) + calc_rec(integral_img, x + 2*w - 1, y - 1, w, h)
                    -calc_rec(integral_img, x + w - 1, y-1, w, h))
                s += 1
                idx_boxes[global_s] = (3, x, y, w, h)
                global_s += 1
        # print(f'Features type 3: {s}')

    if 4 in types:
        s = 0
        for x, y in itertools.product(all_x, all_y):
            all_h = [h for h in range(1, (image.shape[1] - y + 1) // 3 + 1)]
            all_w = [w for w in range(1, image.shape[0] - x + 2)]
            for w, h in itertools.product(all_w, all_h):
                features.append(
                    calc_rec(integral_img, x - 1, y - 1, w, h) + calc_rec(integral_img, x - 1, y + 2*h - 1, w, h)-
                - calc_rec(integral_img, x - 1, y + h - 1, w, h))
                s += 1
                idx_boxes[global_s] = (4, x, y, w, h)
                global_s += 1
        # print(f'Features type 4: {s}')

    if 5 in types:
        s = 0
        for x, y in itertools.product(all_x, all_y):
            all_w = [w for w in range(1, (image.shape[1] - x + 1) // 2 + 1)]
            all_h = [h for h in range(1, (image.shape[0] - y + 1) // 2 + 1)]
            for w, h in itertools.product(all_w, all_h):
                features.append(
                    calc_rec(integral_img, x - 1, y - 1, w, h) + calc_rec(integral_img, x + w - 1, y + h - 1, w, h) -
                calc_rec(integral_img, x + w - 1, y - 1, w, h) - calc_rec(integral_img, x - 1, y + h - 1, w, h))
                s += 1
                idx_boxes[global_s] = (5, x, y, w, h)
                global_s += 1
        # print(f'Features type 5: {s}')
    if dict_boxes:
        return features, idx_boxes
    else:
        return features


def calc_primitive_elect(image, elect_idxs, idx_boxes):
    features = np.zeros(len(idx_boxes))
    integral_img = calc_integral_image(image)
    for idx in elect_idxs:
        type_primitive, x, y, w, h = idx_boxes[idx]
        if type_primitive == 1:
            features[idx] = calc_rec(integral_img, x - 1, y - 1, w, h) - calc_rec(integral_img, x + w - 1, y - 1, w, h)
        elif type_primitive == 2:
            features[idx] = calc_rec(integral_img, x - 1, y - 1, w, h) - calc_rec(integral_img, x - 1, y + h - 1, w, h)
        elif type_primitive == 3:
            features[idx] = calc_rec(integral_img, x - 1, y - 1, w, h) + calc_rec(integral_img, x + 2*w - 1, y - 1, w, h) \
                            - calc_rec(integral_img, x + w - 1, y-1, w, h)
        elif type_primitive == 4:
            features[idx] = calc_rec(integral_img, x - 1, y - 1, w, h) + calc_rec(integral_img, x - 1, y + 2*h - 1, w, h)\
                            - calc_rec(integral_img, x - 1, y + h - 1, w, h)
        else:
            features[idx] = calc_rec(integral_img, x - 1, y - 1, w, h) + calc_rec(integral_img, x + w - 1, y + h - 1, w, h) \
                            -calc_rec(integral_img, x + w - 1, y - 1, w, h) - calc_rec(integral_img, x - 1, y + h - 1, w, h)
    return features



def splitter(features, labels, percent_valid=10, balanced=True, with_idxs=False):
    np.random.seed(10)
    if balanced:
        idxs_dataset = np.arange(len(labels))
        idx_class_1 = np.where(labels == 1)[0]
        idx_class_0 = np.where(labels == -1)[0]
        size_dataset = len(features)
        size_half_valid = int((size_dataset * percent_valid/100) / 2)
        valid_idx_class_1 = np.random.choice(idx_class_1, size_half_valid, replace=False)
        valid_idx_class_0 = np.random.choice(idx_class_0, size_half_valid, replace=False)
        mask = np.zeros(len(labels), dtype=bool)
        mask[valid_idx_class_1] = True
        mask[valid_idx_class_0] = True
        train_idxs = idxs_dataset[~mask]
        valid_idxs = idxs_dataset[mask]
        if with_idxs:
            return features[train_idxs], features[valid_idxs], labels[train_idxs], labels[valid_idxs], train_idxs, valid_idxs
        else:
            return features[train_idxs], features[valid_idxs], labels[train_idxs], labels[valid_idxs]
    else:
        len_dataset = len(features)
        indexes = np.arange(len_dataset)
        nb_class_1 = len(np.where(labels==1)[0])
        nb_class_0 = len(np.where(labels==-1)[0])
        np.random.shuffle(indexes)
        portion = int(len_dataset * (100 - percent_valid) / 100)
        return features[indexes][:portion], labels[indexes][:portion], features[indexes][portion:], labels[indexes][
                                                                                                    portion:]


def read_dataset(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def processing_dataset(dataset, with_idxs=False):
    np.random.seed(10)
    indexes = np.arange(dataset.shape[0])
    np.random.shuffle(indexes)
    if with_idxs:
        return dataset[indexes, :-1], dataset[indexes, -1], with_idxs
    else:
        return dataset[indexes, :-1], dataset[indexes, -1]


def delete_nan(X,y):
    indexes_nan = np.any(np.isnan(X), axis=1)
    X = X[~indexes_nan]
    y = y[~indexes_nan]
    return X,y


def create_dict_idx_and_box():
    empty_picture = np.zeros((19, 19))
    _, dict_idx_and_box = calc_primitive(empty_picture, True)
    return dict_idx_and_box


def create_files(input_path, output_path, class_type):
    paths_class = os.listdir(input_path)
    if class_type == -1:
        output_path = os.path.join(output_path, 'nonfaces_')
    elif class_type == 1:
        output_path = os.path.join(output_path, 'faces_')
    start = time.time()
    print('Starting creating haar-like features')
    for i, name in enumerate(paths_class):
        path_file = os.path.join(input_path, name)
        image = read_image(path_file)
        image = image_to_standard(image)
        features = calc_primitive(image)

        with open(output_path+f'{i}.pickle', 'wb') as f:
            pickle.dump(np.array(features + [class_type]), f)
    print(f'End: {(time.time() - start)/60} min')


def merge_samples(path, output_path):
    paths_sample = os.listdir(path)
    dataset = None
    for i, path_sample in enumerate(paths_sample):
        full_path = os.path.join(path, path_sample)
        sample = read_dataset(full_path)
        if i == 0:
            dataset = sample
        else:
            dataset = np.concatenate((dataset, sample), axis=0)
        if i % 100 == 0:
            print(f'{i}/{len(paths_sample)} sample')
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)



if __name__ == '__main__':
    path_class_1 = 'faces\\train\\face'
    path_class_0 = 'faces\\train\\non-face'


    create_files(path_class_0, 'data/train', -1)
    create_files(path_class_1, 'data/train', 1)

    merge_samples('data/train', 'data/train_samples.pickle')

    # start = time.time()
    # for i, name in enumerate(class_0):
    #     path_file = os.path.join(path_class_0, name)
    #     image = read_image(path_file)
    #     image = image_to_standard(image)
    #     features = calc_primitive(image)
    #
    #     with open(f'data\\train\\nonfaces_{i}.pickle', 'wb') as f:
    #         pickle.dump(np.array(features+[-1]),f)
    # print(time.time() - start)
    #
    # start = time.time()
    # for i, name in enumerate(class_1):
    #     path_file = os.path.join(path_class_1, name)
    #     image = read_image(path_file)
    #     image = image_to_standard(image)
    #     features = calc_primitive(image)
    #     with open(f'data\\train\\faces_{i}.pickle', 'wb') as f:
    #         pickle.dump(np.array(features+[1]),f)
    # print(time.time() - start)
    # with open('train_faces.pickle', 'wb') as f:
    #     pickle.dump(dataset,f)
    # with open('first_img.pickle', 'wb') as f:
    #     pickle.dump(dataset,f)
    # s = time.time()
    # a = read_dataset('test_nonfaces1.pickle')
    # print(time.time()-s)
    # b = read_dataset('faces/train_faces.pickle')
    # with open('faces/train.pickle', 'wb') as f:
    #     pickle.dump(np.concatenate((a, b), axis=0), f)









