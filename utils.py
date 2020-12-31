from dataset import *
import matplotlib.pyplot as plt


def calc_type_predict(y_pred, y_true):
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    TN = np.sum(np.logical_and(y_pred == -1, y_true == -1))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == -1))
    FN = np.sum(np.logical_and(y_pred == -1, y_true == 1))
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN}





def precision(y_pred, y_true, type_pred=None):
    if type_pred is None:
        type_pred = calc_type_predict(y_pred, y_true)

    return type_pred['TP'] / (type_pred['TP'] + type_pred['FP'])


def recall(y_pred, y_true, type_pred=None):
    if type_pred is None:
        type_pred = calc_type_predict(y_pred, y_true)

    return type_pred['TP'] / (type_pred['TP'] + type_pred['FN'])


def accuracy(y_pred, y_true, type_pred=None):
    if type_pred is None:
        type_pred = calc_type_predict(y_pred, y_true)
    return (type_pred['TP'] + type_pred['TN']) / (type_pred['TP'] + type_pred['TN'] + type_pred['FP'] + type_pred['FN'])


def false_positive_rate(y_pred, y_true, type_pred=None):
    if type_pred is None:
        type_pred = calc_type_predict(y_pred, y_true)
    return type_pred['FP']/(type_pred['FP']+type_pred['TN'])


def true_positive_rate(y_pred, y_true, type_pred=None):
    return recall(y_pred,y_true, type_pred)


def detection_rate(y_pred, y_true, type_pred=None):
    return recall(y_pred, y_true, type_pred)


def eval_metrics(y_pred=None, y_true=None, type_pred=None,printed=False):

    precision_ = precision(y_pred,y_true, type_pred)
    recall_ = recall(y_pred,y_true, type_pred)
    fpr = false_positive_rate(y_pred, y_true, type_pred)
    accuracy_ = accuracy(y_pred,y_true, type_pred)
    if printed:
        print(f'Precision: {precision_}\n Recall (True positive rate): {recall_}\n Accuracy {accuracy_}\n '
              f'False positive rate: {fpr}')
    return precision_, recall_, accuracy_, fpr


def save(path, filename, obj):
    with open(os.path.join(path, f'{filename}.pickle'), 'wb') as f:
        pickle.dump(obj, f)


def load(path, filename):

    with open(os.path.join(path, f'{filename}.pickle'), 'rb') as f:
        return pickle.load(f)


def IoU(prediction_box, ground_truth_box, image):
    pred = np.zeros(image.shape[:2], dtype=bool)
    pred[prediction_box[1]:prediction_box[3],
        prediction_box[0]:prediction_box[2]] = True

    gt = np.zeros(image.shape[:2], dtype=bool)
    gt[ground_truth_box[1]:ground_truth_box[3],
    ground_truth_box[0]: ground_truth_box[2]] = True

    intersection = pred * gt
    union = pred + gt

    iou = intersection.sum() / union.sum()
    return iou


def sliding_window(image, size=(19, 19), stride=1):
    for x in range(0, image.shape[1]-size[1], stride):
        for y in range(0, image.shape[0]-size[0], stride):
            yield image[y:y+size[0], x:x+size[1]], (x,y,x+size[0],y+size[1])


def predict_picture(path, adboost):
    features = read_dataset(path)
    pred, prob = adboost.test(features[:-1][np.newaxis, :], adboost.thr)
    return pred, features[-1]

def nms_ineffective(boxes, image):
    i = 0
    j = 1
    selected_boxes = boxes.copy()
    selected_boxes.sort(key=lambda x: x[1], reverse=True)
    print('Starting NMS')
    while i < len(selected_boxes):
        while j < len(selected_boxes):
            img_c = image.copy()
            for bx in [selected_boxes[i][0],selected_boxes[j][0]]:
                img_c[bx[1]:bx[3], bx[0]] = [255, 255, 0]
                img_c[bx[1], bx[0]:bx[2]] = [255, 255, 0]
                img_c[bx[1]:bx[3], bx[2] - 1] = [255, 255, 0]
                img_c[bx[3] - 1, bx[0]:bx[2]] = [255, 255, 0]


            iou = IoU(selected_boxes[i][0], selected_boxes[j][0], image)
            if iou >= 0.5:
                del selected_boxes[j]
            else:
                j += 1
        i += 1
        j = i + 1
        # print(i, len(selected_boxes))
    return selected_boxes


def show_image_with_boxes(img, boxes):
    for i in range(len(boxes)):
        box = boxes[i][0]
        img[box[1]:box[3], box[0]] = [255, 255, 0]
        img[box[1], box[0]:box[2]] = [255, 255, 0]
        img[box[1]:box[3], box[2] - 1] = [255, 255, 0]
        img[box[3] - 1, box[0]:box[2]] = [255, 255, 0]
        plt.text(box[0], box[1], f'{np.round(boxes[i][1],2)} ({np.round(boxes[i][2],2)})', fontsize=5, color='black',
                 bbox={'facecolor':'yellow','edgecolor': 'yellow', 'boxstyle': 'round'})
    plt.title(f'Nums of boxes:{len(boxes)}')
    plt.imshow(np.uint8(img))
    plt.axis('off')
    plt.show()
