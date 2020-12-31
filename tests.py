from random import sample
from cascade_adaboost import *


def find_face(path_img, loading_cascade, load_boxes_path=None, save_path=None):
    img_gray = read_image(path_img)
    img_color = read_image(path_img, color=True)
    current_shape = img_gray.shape
    orig_shape = img_gray.shape
    if load_boxes_path is None:
        cascade = CascadeAdaBoost(0.5, 0.995, 0.001, 0.9)
        cascade.load_adaboosts(loading_cascade)
        idxs_features = cascade.idxs_feature_stumps
        idx_boxes = create_dict_idx_and_box()
        scale_factor = 1
        selected_boxes = []
        scaling_img = img_gray
        while current_shape[0] > 19 or current_shape[1] > 19:
            print(f'Scale factor: {scale_factor}, Shape:{current_shape}')
            scaling_img = cv2.resize(scaling_img, current_shape[::-1])
            # show_image(scaling_img,cmap='gray')
            windows = sliding_window(scaling_img, stride=3)
            for window, box in windows:
                std_window = image_to_standard(window)
                features = np.array(calc_primitive_elect(std_window,idxs_features, idx_boxes))[np.newaxis,:]
                pred, prob = cascade.predict(features)
                if pred == 1 and prob > 0.5:
                    scaling_box = list(box)
                    scaling_box[2] = int(box[2]*scale_factor) if int(box[2]*scale_factor) < orig_shape[1] else orig_shape[1]
                    scaling_box[3] = int(box[3]*scale_factor) if int(box[3]*scale_factor) < orig_shape[0] else orig_shape[0]
                    selected_boxes.append((box, prob[0], scale_factor))

            scale_factor *= 1.25
            current_shape = (int(current_shape[0]//1.25),int(current_shape[1]//1.25))
        cut_selected_boxes = []
        for i, box in enumerate(selected_boxes):
            scaling_box = list(box[0])
            scale_factor = box[2]
            scaling_box[0] = int(scaling_box[0]*scale_factor)
            scaling_box[1] = int(scaling_box[1]*scale_factor)
            scaling_box[2] = scaling_box[0] + int((box[0][2] - box[0][0]) * scale_factor) if int(
                box[0][2] * scale_factor) < \
                                                                                        orig_shape[1] else \
            orig_shape[1]
            scaling_box[3] = scaling_box[1] + int((box[0][3] - box[0][1]) * scale_factor) if int(
                box[0][3] * scale_factor) < \
                                                                                        orig_shape[0] else \
            orig_shape[0]

            cut_selected_boxes.append((scaling_box, box[1], box[2]))
        clear_boxes = nms_ineffective(cut_selected_boxes, img_color)
        if save_path is not None:
            save('boxes', save_path, clear_boxes)
    else:
        clear_boxes = load('boxes', load_boxes_path)

    show_image_with_boxes(img_color, clear_boxes)


def test_cascade_on_train_part_dataset(loading_dataset, loading_cascade):
    dataset = read_dataset(loading_dataset)
    X, y = processing_dataset(dataset)
    indexes_nan = np.any(np.isnan(X), axis=1)
    X = X[~indexes_nan]
    y = y[~indexes_nan]
    type_preds = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    cascade = CascadeAdaBoost(0.5, 0.995, 0.001, 0.9)
    cascade.load_adaboosts(loading_cascade)

    for t in range(len(X)):
        label = y[t]
        y_pred = cascade.predict(X[t:t+1])
        if y_pred[0] == 1 and label == 1:
            type_preds['TP'] += 1
        elif y_pred[0] == -1 and label == 1:
            type_preds['FN'] += 1
        elif y_pred[0] == 1 and label == -1:
            type_preds['FP'] += 1
        else:
            type_preds['TN'] += 1
    print('Train dataset')
    eval_metrics(type_pred=type_preds, printed=True)


def test_cascade_on_test_part_dataset(loading_dataset, loading_cascade, balanced=False):
    path_test_class_1 = os.path.join(loading_dataset, 'face')
    path_test_class_0 = os.path.join(loading_dataset, 'non-face')
    type_preds = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    cascade = CascadeAdaBoost(0.5, 0.995, 0.001, 0.9)
    cascade.load_adaboosts(loading_cascade)
    nb_img = 0

    for path_test in [path_test_class_1, path_test_class_0]:
        if path_test == path_test_class_0:
            if balanced:
                paths_img = sample(os.listdir(path_test), 472)
            else:
                paths_img = os.listdir(path_test)
        else:
            paths_img = os.listdir(path_test)
        for i, path_img in enumerate(paths_img):
            picture = read_dataset(os.path.join(path_test, path_img))
            features, label = picture[:-1][np.newaxis,:], picture[-1]
            y_pred, prob = cascade.predict(features)
            if y_pred == 1 and label == 1:
                type_preds['TP'] += 1
            elif y_pred == -1 and label == 1:
                type_preds['FN'] += 1
            elif y_pred == 1 and label == -1:
                type_preds['FP'] += 1
            else:
                type_preds['TN'] += 1
            nb_img += 1
            if nb_img % 500 == 0:
                print(f'Processed pictures: {nb_img}/24045')
                print(type_preds)
    print('Test dataset')
    eval_metrics(type_pred=type_preds, printed=True)



if __name__ == "__main__":
    # test_cascade_on_train_part_dataset(path_train_dataset,'ab_cascade')
    print('---------------------------------------------------------')
    # test_cascade_on_test_part_dataset(path_test_dataset, 'ab_cascade')
    
    
    # find_face('imgs/test_group_1.png', 'ab_cascade', save_path='test_group_1')
    # find_face('imgs/test_group_2.jpeg','ab_cascade',save_path='test_group_2')
    # find_face('imgs/test_group_3.jpg', 'ab_cascade', save_path='test_group_3')
    # find_face('imgs/test_group_4.jpg', 'ab_cascade', save_path='test_group_4')
    # find_face('imgs/test_group_5.jpg', 'ab_cascade', save_path='test_group_5')
    # find_face('imgs/test_group_6.jpeg', 'ab_cascade', save_path='test_group_6')
    # find_face('imgs/test_group_8.jpg', 'ab_cascade', save_path='test_group_8')
    # find_face('imgs/test_group_9.jpg', 'ab_cascade', save_path='test_group_9')
    
    find_face('imgs/test_group_1.png', 'ab_cascade', load_boxes_path='test_group_1')
    find_face('imgs/test_group_2.jpeg','ab_cascade',load_boxes_path='test_group_2')
    find_face('imgs/test_group_3.jpg', 'ab_cascade', load_boxes_path='test_group_3')
    find_face('imgs/test_group_4.jpg', 'ab_cascade', load_boxes_path='test_group_4')
    find_face('imgs/test_group_5.jpg', 'ab_cascade', load_boxes_path='test_group_5')
    find_face('imgs/test_group_6.jpeg', 'ab_cascade', load_boxes_path='test_group_6')
    find_face('imgs/test_group_8.jpg', 'ab_cascade', load_boxes_path='test_group_8')
    find_face('imgs/test_group_9.jpg', 'ab_cascade', load_boxes_path='test_group_9')
