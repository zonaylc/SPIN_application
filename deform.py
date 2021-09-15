from demo import *

# set the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# set the SPIN model with the configuration of the device
null_smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(device)

# reset the pose to the rest pose(SMPL)
def rest_pose(original_vertices, pred_betas, pred_camera): 
  output = null_smpl(v_template=original_vertices ,betas=pred_betas, transl=pred_camera)
  out = output.vertices.cpu().detach().numpy().squeeze()
  joint = output.joints.cpu().detach().numpy().squeeze()
  return out, joint


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



# reconstruct a 3d human body mesh from a single image and reset the posture to the rest pose in SMPL model
def reconstruct(img):
    pred_rotmat, pred_betas, pred_camera, original_vertices, pred_orient, pred_pose, pred_joints, pred_betas, pred_fullpose = build_mesh(model_checkpoint='data/model_checkpoint.pt', photo=img, device=device, gender = gender)
    out, joint = rest_pose(original_vertices, pred_betas, pred_camera)

    # visualize the result
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_axis_off()
    ax.scatter(out[:,0], -out[:,2], out[:,1], c="palevioletred",s=0.1) 
    ax.scatter(original_vertices[:,0], -original_vertices[:,2], original_vertices[:,1], c="palevioletred",s=0.1)
    plt.xlim(-1,3)
    plt.ylim(-3,1)