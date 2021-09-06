import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json

from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants
import matplotlib.pyplot as plt



def bbox_from_openpose(openpose_file, rescale=1.2, detection_thresh=0.2):
    """Get center and scale for bounding box from openpose detections."""
    with open(openpose_file, 'r') as f:
        keypoints = json.load(f)['people'][0]['pose_keypoints_2d']
    keypoints = np.reshape(np.array(keypoints), (-1,3))
    valid = keypoints[:,-1] > detection_thresh
    valid_keypoints = keypoints[valid][:,:-1]
    center = valid_keypoints.mean(axis=0)
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)).max()
    # adjust bounding box tightness
    scale = bbox_size / 200.0
    scale *= rescale
    return center, scale

def bbox_from_json(bbox_file):
    """Get center and scale of bounding box from bounding box annotations.
    The expected format is [top_left(x), top_left(y), width, height].
    """
    with open(bbox_file, 'r') as f:
        bbox = np.array(json.load(f)['bbox']).astype(np.float32)
    ul_corner = bbox[:2]
    center = ul_corner + 0.5 * bbox[2:]
    width = max(bbox[2], bbox[3])
    scale = width / 200.0
    # make sure the bounding box is rectangular
    return center, scale

def process_image(img_file, bbox_file, openpose_file, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    img = cv2.imread(img_file)[:,:,::-1].copy() # PyTorch does not support negative stride at the moment
    if bbox_file is None and openpose_file is None:
        # Assume that the person is centerered in the image
        height = img.shape[0]
        width = img.shape[1]
        center = np.array([width // 2, height // 2])
        scale = max(height, width) / 200
    else:
        if bbox_file is not None:
            center, scale = bbox_from_json(bbox_file)
        elif openpose_file is not None:
            center, scale = bbox_from_openpose(openpose_file)
    img = crop(img, center, scale, (input_res, input_res))
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2,0,1)
    norm_img = normalize_img(img.clone())[None]
    return img, norm_img





def build_mesh(model_checkpoint, photo, device, gender):    

    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                gender=str(gender),
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)


    # Preprocess input image and generate predictions
    img, norm_img = process_image(photo, None, None, input_res=constants.IMG_RES)
    with torch.no_grad():
        pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
        # print(pred_rotmat.shape, pred_betas.shape, pred_camera.shape, pred_rotmat[:,0].unsqueeze(1).shape)
        # set return_full_pose to True
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False, return_full_pose=True)
        pred_vertices = pred_output.vertices

        original_vertices = pred_output.vertices.cpu().numpy().squeeze()
        pred_orient = pred_output.global_orient #.cpu().numpy().squeeze()
        pred_pose = pred_output.body_pose #.cpu().numpy().squeeze()
        pred_joints = pred_output.joints.cpu().numpy().squeeze()
        pred_betas = pred_output.betas #.cpu().numpy().squeeze()
        pred_fullpose = pred_output.full_pose #.cpu().numpy().squeeze()
      
         

    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices =  original_vertices #pred_vertices[0].cpu().numpy()
    img = img.permute(1,2,0).cpu().numpy()
    # torch.save(pred_vertices, 'sample_tensor.pt')
    
    # Render parametric shape
    img_shape = renderer(pred_vertices, camera_translation, img)
    # torch.save(img_shape, 'sample_front.pt')

    
    outfile = photo.split('.')[0]
    outfile = '/content/gdrive/MyDrive/Clothing/output/' + outfile
    # Save reconstructions
    cv2.imwrite(outfile + '_mesh.png', 255 * img_shape[:,:,::-1])
    

    return pred_rotmat, pred_betas, pred_camera, original_vertices, pred_orient, pred_pose, pred_joints, pred_betas, pred_fullpose

      