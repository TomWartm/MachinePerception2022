import pdb
import sys
sys.path.append('.')
import pdb
import torch
import torch.nn as nn
from .head import SMPLHead
import torchvision
from ..utils.geometry import rot6d_to_rotmat, rotmat_to_rot6d
from .hmr import create_hmr
from hps_core.utils.train_utils import load_pretrained_model, set_seed, add_init_smpl_params_to_dict


class Model4(nn.Module):
    """very small model, but use the idea in hmr"""
    def __init__(self):
        self.inplanes = 64
        super(Model4, self).__init__()
        # pdb.set_trace()
        new_dict = add_init_smpl_params_to_dict({})
        
        self.register_buffer('init_pose', new_dict['model.head.init_pose'])
        self.register_buffer('init_shape', new_dict['model.head.init_shape'])
        self.register_buffer('init_cam', new_dict['model.head.init_cam'])
        
        npose = 24 * 6

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3, bias=True)
        
        self.fnn = nn.Linear(128*8*8, 192)
        self.avg_pool = nn.AdaptiveAvgPool2d(8)
        self.decpose = nn.Linear(192, npose)
        self.decshape = nn.Linear(192, 10)
        self.deccam = nn.Linear(192, 3)


        self.head = SomeHead()

        # SMPLHead takes estimated pose, shape, cam parameters as input
        # and returns the 3D mesh vertices, 3D/2D joints as output
        self.smpl = SMPLHead(
            focal_length=5000.,
            img_res=img_res
        )
    
    def forward(self, features):    
        batch_size = images.shape[0]

        features = self.conv1(images)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.bn1(features)
        
        features = self.conv2(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.bn2(features)
        
        features = self.conv3(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.bn3(features)
        
        features = self.conv4(features)
        features = self.relu(features)
        features = self.maxpool(features)
        
        
        features = features.reshape(batch_size, -1)

        features = self.fnn(features)

        pred_pose = self.init_pose
        pred_shape = self.init_shape
        pred_cam = self.init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam


        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        smpl_output = self.smpl(
            rotmat=pred_rotmat,
            shape=pred_shape,
            cam=pred_cam,
            normalize_joints2d=True,
        )

        # pdb.set_trace()
        output = {
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose_6d': pred_pose,
        }
        smpl_output.update(output)

        # smpl output keys and valus:
        #         "smpl_vertices": torch.Size([batch_size, 6890, 3]), -> 3D mesh vertices
        #         "smpl_joints3d": torch.Size([batch_size, 49, 3]), -> 3D joints
        #         "smpl_joints2d": torch.Size([batch_size, 49, 2]), -> 2D joints
        #         "pred_cam_t": torch.Size([batch_size, 3]), -> camera translation [x,y,z]
        #         "pred_pose": torch.Size([batch_size, 24, 3, 3]), -> SMPL pose params in rotation matrix form
        #         "pred_cam": torch.Size([batch_size, 3]), -> weak perspective camera [s,tx,ty]
        #         "pred_shape": torch.Size([batch_size, 10]), -> SMPL shape (betas) params
        #         "pred_pose_6d": torch.Size([batch_size, 144]), -> SMPL pose params in 6D rotation form

        return smpl_output



def yaz(input_size):
    summary(Model4(), input_size)
yaz((3, 128, 128))

