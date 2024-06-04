import cv2 as cv
import numpy as np
import torch
import yaml
from ultralytics import YOLO, checks
from yolo.functions import getModel
from ultralytics.engine.results import Results
import os

checks()

def crop_square(img, size, interpolation=cv.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv.resize(crop_img, (size, size), interpolation=interpolation)

    return resized

def crop_and_resize(img, w, h):
    im_h, im_w, channels = img.shape
    res_aspect_ratio = w/h
    input_aspect_ratio = im_w/im_h

    if input_aspect_ratio > res_aspect_ratio:
        im_w_r = int(input_aspect_ratio*h)
        im_h_r = h
        img = cv.resize(img, (im_w_r , im_h_r))
        x1 = int((im_w_r - w)/2)
        x2 = x1 + w
        img = img[:, x1:x2, :]
    if input_aspect_ratio < res_aspect_ratio:
        im_w_r = w
        im_h_r = int(w/input_aspect_ratio)
        img = cv.resize(img, (im_w_r , im_h_r))
        y1 = int((im_h_r - h)/2)
        y2 = y1 + h
        img = img[y1:y2, :, :]
    if input_aspect_ratio == res_aspect_ratio:
        img = cv.resize(img, (w, h, 3))

    return img

def apply_mask(frame, mask):
    """Apply binary mask to frame, return in-place masked image."""
    return cv.bitwise_and(frame,frame,mask=mask)


async def yoloTest(modelNum: int | None):
    modelPath =  getModel(model_name=modelNum, find='weights/best_web_model') if modelNum is not None else getModel(find='weights/best_web_model')
    modelData = yaml.safe_load(open(os.path.join(modelPath, '/args.yaml')))
    modelName = modelData['name']
    modelSize = modelData['imgsz']
    
    if not isinstance(modelSize, list) and not isinstance(modelSize, int):
        raise ValueError(f'Não foi possivel determinar o tamanho do modelo: {modelPath}')

    # Load the exported model
    model = YOLO(model=os.path.join(modelPath, 'weights/best_web_model'), task='segment', verbose=True)
    

    # Run inference
    # half Float16 invez de Float32
    # augment: Melhor a robustez da deteção a custo de tempo
    # agnostic_nms: Mescle caixas
    # retina_masks: Melhora a qualidade das máscaras
    results: list[Results] = model.predict('images', imgsz=modelSize, boxes=True, visualize=True, augment=True, agnostic_nms=True)
    
    if not os.path.exists('output'):
        os.mkdir('output')

    # Process results generator
    for result in results:
        imageName = str(result.path).split('/').pop().split('.')[0] # path/image.png => image.png => image
        image = cv.imread(result.path)
        seg_classes = list(result.names.values())

        shape = result.orig_shape
        masks = result.masks.data
        boxes = result.boxes.data

        clss = boxes[:, 5]

        #EXTRACT A SINGLE MASK WITH ALL THE CLASSES
        obj_indices = torch.where(clss != -1)
        obj_masks = masks[obj_indices]
        obj_mask = torch.any(obj_masks, dim=0).int() * 255
        cv.imwrite(f'output/{modelName}-{imageName}-all-masks-{modelSize}.png', obj_mask.cpu().numpy())
        
        oldMask = cv.imread(f'output/{modelName}-{imageName}-all-masks-{modelSize}.png')
        newMask = crop_and_resize(oldMask, shape[1], shape[0])
        cv.imwrite(f'output/{modelName}-{imageName}-all-masks-{modelSize}.png', newMask)
        
        mask = cv.imread(f'output/{modelName}-{imageName}-all-masks-{modelSize}.png',0)
        
        apllyMask = apply_mask(image, mask)
        cv.imwrite(f'output/{modelName}-{imageName}-all-masks-{modelSize}.png', apllyMask)

        #MASK OF ALL INSTANCES OF A CLASS
        for i, seg_class in enumerate(seg_classes):

            obj_indices = torch.where(clss == i)
            obj_masks = masks[obj_indices]
            obj_mask = torch.any(obj_masks, dim=0).int() * 255

            cv.imwrite(str(f'output/{modelName}-{imageName}-{seg_class}s-{modelSize}.png'), obj_mask.cpu().numpy())

            #MASK FOR EACH INSTANCE OF A CLASS
            for i, obj_index in enumerate(obj_indices[0].numpy()):
                obj_masks = masks[torch.tensor([obj_index])]
                obj_mask = torch.any(obj_masks, dim=0).int() * 255
                cv.imwrite(str(f'output/{modelName}-{imageName}-{seg_class}_{i}-{modelSize}.png'), obj_mask.cpu().numpy())
    # cv.imwrite('mascaraAplicada.png', applyMask.cpu().numpy())