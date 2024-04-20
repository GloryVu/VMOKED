from dataset import SmokeDetectionDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from model.OurModel.object_detection import Detector


smkd_dataset = SmokeDetectionDataset('data/dataset_v1/test','data/dataset_v1/labels',sequence_length=9)
test_loader = DataLoader(dataset=smkd_dataset, batch_size=1, pin_memory=True, shuffle=True)


detector = Detector('./model/OurModel/models/best_ds_nms.trt','./model/OurModel/models/sky_seg.trt')
detector.set_params(2,1, 0.9, 0.00, 0.00)

def dict_of_label(y):
    return [{
            'label': yy[0].item(),
            'box': [yy[1][0].item(), 
                    yy[1][1].item(), 
                    yy[1][2].item(), 
                    yy[1][3].item()],
            'conf': 1.0
            } for yy in y]


def iou(box1, box2):
        # Extract coordinates of the boxes
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    
    # Calculate the intersection coordinates
    x_left = max(xmin1, xmin2)
    y_top = max(ymin1, ymin2)
    x_right = min(xmax1, xmax2)
    y_bottom = min(ymax1, ymax2)
    
    # Calculate the intersection area
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # Calculate the areas of the bounding boxes
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def calculate_detection_accuracy(predicted_boxes, gt_boxes, iou_threshold=0.5):
    """
    input: predicted_boxes, gt_boxes, iou_threshold
    return tp, tn, fp, fn
    """
    total_gt_objects = len(gt_boxes)
    total_predicted_objects = len(predicted_boxes)
    tp = tn = fp = fn = 0
    if total_gt_objects==0 & total_predicted_objects==0:
        tn = 1
        return tp, tn, fp, fn
    if total_gt_objects==0 & total_predicted_objects!=0:
        fp = total_predicted_objects
        return tp, tn, fp, fn
    if total_gt_objects!=0 & total_predicted_objects==0:
        fn = total_gt_objects

    match_matrix = np.zeros((total_predicted_objects,total_gt_objects))
    for pred_idx in range(total_predicted_objects):
        for gt_idx in range(total_gt_objects): 
            if iou(predicted_boxes[pred_idx],gt_boxes[gt_idx]) > iou_threshold:
                match_matrix[pred_idx][gt_idx] = 1
    gt_sum = np.sum(match_matrix, 0)
    tp = np.sum(gt_sum)
    fn = np.sum(gt_sum==0)
    fp = np.sum(np.sum(match_matrix,1)==0)
    return tp, tn, fp, fn

conf_threshold = 0.5

# predicted_boxes
loop = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
total_tp = total_tn = total_fp = total_fn = 0
epsilon = 0.0000001
for x, y in test_loader:
    x = np.array([i.numpy().squeeze(0) for i in x])
    if(len(y)!=0):
        y = dict_of_label(y)
    
    preds = detector.detect(x)
    if preds:
        print(preds)
    gt_boxes = list(map(lambda yy: yy['box'],y))
    pred_boxes = list(map(lambda yy: yy['box'],preds))
    tp, tn, fp, fn = calculate_detection_accuracy(pred_boxes, gt_boxes)
    if fp !=0 or fn !=0:
        print()
    total_tp+=tp
    total_tn+=tn
    total_fp+=fp
    total_fn+=fn
    precision = total_tp/(total_tp+total_fp+epsilon)
    recall = total_tp/(total_tp+total_fn+epsilon)
    f1 = 2*precision*recall/(precision+recall+epsilon)
    acc = (total_tp+total_tn)/(total_tp+total_tn+total_fn+total_fp)
    loop.set_postfix(precision=f'{precision:.4f}', recall=f'{recall:.4f}', f1=f'{f1:.4f}', acc=f'{acc:.4f}')
    loop.update(1)

#  print(f'accuracy: {round(num_correct / num_samples, 4)}')