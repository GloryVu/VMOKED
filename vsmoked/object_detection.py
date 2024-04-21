import cv2
from .detector import TensorRtDetector
from .segmentor import TensorRtSegmentor
import numpy as np
import time
width = 640
# Required: Object detection model input height (default: shown below)
height = 640
# Optional: Label name modifications. These are merged into the standard labelmap.
input_tensor = "nchw"
input_pixel_format = "rgb"

bbox_motion_resize_w = 75
bbox_motion_resize_h = 75
label_map={
    0: 'smoke'
}
def intersection(box_a, box_b):
    return (
        max(box_a[0], box_b[0]),
        max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]),
        min(box_a[3], box_b[3]),
    )

def intersection_over_union(box_a, box_b):
    # determine the (x, y)-coordinates of the intersection rectangle
    intersect = intersection(box_a, box_b)

    # compute the area of intersection rectangle
    inter_area = max(0, intersect[2] - intersect[0] + 1) * max(
        0, intersect[3] - intersect[1] + 1
    )

    if inter_area == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # return the intersection over union value
    return iou

def similarity(img1,img2):
        
    if(img1.shape!=img2.shape):
        img1 = cv2.resize(img1,[img2.shape[1],img2.shape[0]])
    # logger.info(f' image {img1}{img2}')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # logger.info(f'{gray1}')
    diff = cv2.absdiff(gray1,gray2)
    # logger.info(f'difference error: {absdiff}')
    similarity = 1 - diff.sum() / float(img1.shape[0]*img1.shape[1]*255)
    return similarity
    
def motion_measure(input_motion):
    sim = 0
    origin_h, orgin_w,_ = input_motion[0].shape
    input_motion = [cv2.resize(im,(bbox_motion_resize_h,bbox_motion_resize_w))
                    for im in input_motion]
    for i in range(len(input_motion)-1):
        # logger.info(f'diff mmb {similarity(input_motion[i],input_motion[i+1])}')
        sim += similarity(input_motion[i],input_motion[i+1])
    # logger.log(input_motion)
    return (1-sim/(len(input_motion)-1))*(bbox_motion_resize_h*bbox_motion_resize_w)/(origin_h*orgin_w)

def get_bbox(detection,h,w):
    return [max(0,int(detection[2]*w//1)),
            max(0,int(detection[3]*h//1)),
            min(w-1,int(detection[4]*w//1)),
            min(h-1,int(detection[5]*h//1))]
    
def find_matchest_box(box,predet,h,w,detection):
    max_iou = 0
    matchest_box = []
    for det in predet:
        if det[1] == 0:
            continue
        if(label_map[detection[0]]!='smoke'):
            continue
        prebox = get_bbox(det,h,w)
        iou = intersection_over_union(box,prebox)
        if (iou > max_iou):
            max_iou = iou
            matchest_box = prebox
    return max_iou,matchest_box
    
def on_sky(box,mask,threshold=0.8):
    percent = 1.0*mask[box[1]:box[3]][box[0]:box[2]].sum()/((box[3]-box[1])*(box[2]-box[0]))
    return percent >= threshold

    # frame_queue =[]
    # output_queue =[]
class Detector:
    def __init__(self,detector_path,segmentor_path,windowsize = 7, step = 3, 
                 preset_similarity= 0.9, curbbox_motion_threshold = 0.02,
                 bboxes_motion_threshold = 0.02, use_SGS_block=True) -> None:
        self.detector = TensorRtDetector(detector_path)
        self.segmentor = TensorRtSegmentor(segmentor_path)
        self.windowsize = windowsize
        self.step = step
        self.preset_similarity =preset_similarity
        self.curbbox_motion_threshold = curbbox_motion_threshold
        self.bboxes_motion_threshold= bboxes_motion_threshold
        self.use_SGS_block=use_SGS_block

    def set_params(self,windowsize, step, preset_similarity, curbbox_motion_threshold,
                        bboxes_motion_threshold):
        self.windowsize = windowsize
        self.step = step
        self.preset_similarity=preset_similarity
        self.curbbox_motion_threshold = curbbox_motion_threshold
        self.bboxes_motion_threshold = bboxes_motion_threshold
    def detect(self, frames, conf=0.4):
        
        input_frame = frames[-1]
        
        detections = [self.detector.detect_raw(np.expand_dims(np.transpose(frame, (2, 0, 1)),0)) for frame in frames]
        # frames =[np.transpose(frame, (1, 2, 0)) for frame in frames]
        detection = detections[-1]
        # We measure latency from here because in product we cache previous predict
        # this is only simulate for convention
        total_time = 0
        start = time.time()
        if(similarity(frames[0],frames[self.windowsize-1]) < self.preset_similarity):
            detection = [[0.0]*8]*20
        else:
        # remove no motion objects
            self.detector.detect_raw(np.expand_dims(np.transpose(frames[0], (2, 0, 1)),0)) 
            for i, pred in enumerate(detection):
                if pred[1] < conf:
                    detection[i] = [0.0]*8
                    continue
                if(label_map[int(pred[0])]!='smoke'):
                    detection[i] = [0.0]*8
                    continue
                h = input_frame.shape[0]
                w = input_frame.shape[1]
                
                box = get_bbox(pred,h,w)
                image = frames[-1]
                # image = cv2.resize(image,[320,320])
                total_time+= time.time()- start
                # this also cached
                if self.use_SGS_block:
                    if on_sky(box, self.segmentor.detect_raw(np.expand_dims(np.transpose(image, (2, 0, 1)),0))):
                        detection[i] = [0.0]*8
                        continue
                start = time.time()
                input_motion = [a[box[1]:box[3],box[0]:box[2],:].copy() for a in frames[::self.step]]
                # logger.info(f'{input_motion}')
                # logger.info(f'inputmotion shape : {input_motion[0].shape}')
                # logger.info(f'box : {box}')
                # for ii in range(len(input_motion)):
                #     input_motion[ii] = input_motion[ii][box[1]:box[3],box[0]:box[2],:].copy()
                # logger.info(f'inputmotion strimed shape : {input_motion[0].shape}')
                mm = motion_measure(input_motion)
                # logger.info(f'motion_measure : {mm} {detector_config.model.self.curbbox_motion_threshold}')
                detection[i][6] = mm
                if mm <self.curbbox_motion_threshold:
                    detection[i] = [0.0]*8
                    print('motionless mm')
                    continue
                # check previous detection
                input_motion_box = [frames[-1][box[1]:box[3],box[0]:box[2],:].copy()]
                for j in range(0,self.windowsize-1,self.step):
                    predet = detections[self.windowsize-2-j]
                    iou, matchest_box = find_matchest_box(box,predet,h,w,pred)
                    # logger.info(f'{iou} {matchest_box}')
                    if(iou < 0.1):
                        break
                    # current_frame = 
                    input_motion_box.append(
                    frames[self.windowsize-2-j][matchest_box[1]:matchest_box[3],matchest_box[0]:matchest_box[2],:].copy())
                
                if(len(input_motion_box)<= self.windowsize//(2*self.step)):
                    # detection[i] = [0.0]*8
                    continue
                # logger.info(f'inputmotionbox shape : {input_motion_box}')
                # logger.info(f'inputmotionbox shape : {len(input_motion_box)}')
                mmb = motion_measure(input_motion_box)
                # logger.info(f'motion_measure_box : {mmb} {detector_config.model.self.bboxes_motion_threshold}')
                
                if(mmb<self.bboxes_motion_threshold):
                    detection[i] = [0.0]*8
                    print('motionless mmb')
                    continue
                
                detection[i][7] = mmb
        detection =[{'label': det[0],
                    'box': det[2:6].tolist(),
                    'conf': det[1],
                    "mm": det[6],
                    "mmb":det[7]
                    } for det in detection if sum(det) != 0]
    
        return time.time()-start+total_time,detection

    def cal_motion(self,frames,detections, conf=0.4):
        input_frame = frames[-1]
        # detections = [self.detector.detect_raw(np.expand_dims(np.transpose(frame, (2, 0, 1)),0)) for frame in frames]
        # frames =[np.transpose(frame, (1, 2, 0)) for frame in frames]
        detection = detections[-1]
        # preset_similarity = similarity(frames[0],frames[self.windowsize-1])
        # if(preset_similarity >= preset_similarity):
        # remove no motion objects

        for i, pred in enumerate(detection):
            h = input_frame.shape[0]
            w = input_frame.shape[1]
            
            box = get_bbox(pred,h,w)
            image = frames[-1]
            # image = cv2.resize(image,[320,320])
            input_motion = [a[box[1]:box[3],box[0]:box[2],:].copy() for a in frames[::self.step]]
            mm = motion_measure(input_motion)
            detection[i][6] = mm
            # check previous detection
            input_motion_box = [frames[-1][box[1]:box[3],box[0]:box[2],:].copy()]
            for j in range(0,self.windowsize-1,self.step):
                predet = detections[self.windowsize-2-j]
                iou, matchest_box = find_matchest_box(box,predet,h,w,pred)
                # logger.info(f'{iou} {matchest_box}')
                if(iou < 0.1):
                    break
                # current_frame = 
                input_motion_box.append(
                frames[self.windowsize-2-j][matchest_box[1]:matchest_box[3],matchest_box[0]:matchest_box[2],:].copy())
            
            if(len(input_motion_box)<= self.windowsize//(2*self.step)):
                # detection[i] = [0.0]*8
                continue
            # logger.info(f'inputmotionbox shape : {input_motion_box}')
            # logger.info(f'inputmotionbox shape : {len(input_motion_box)}')
            mmb = motion_measure(input_motion_box)
            # logger.info(f'motion_measure_box : {mmb} {detector_config.model.self.bboxes_motion_threshold}')
            
            if(mmb<self.bboxes_motion_threshold):
                detection[i] = [0.0]*8
                continue
            
            detection[i][7] = mmb
        detection =[{'label': det[0],
                     'box': det[2:6].tolist(),
                     'conf': det[1],
                     "mm": det[6],
                     "mmb":det[7]
                     } for det in detection if sum(det) != 0]
        return detection