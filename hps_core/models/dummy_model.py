import pdb
import torch
import torch.nn as nn
from .head import SMPLHead
import torchvision
from ..utils.geometry import rot6d_to_rotmat, rotmat_to_rot6d
from .hmr import create_hmr
from hps_core.utils.train_utils import load_pretrained_model, set_seed, add_init_smpl_params_to_dict
class SomeHead(nn.Module):
    def __init__(self):
        super(SomeHead, self).__init__()
        new_dict = add_init_smpl_params_to_dict({})
        
        self.register_buffer('init_pose', new_dict['model.head.init_pose'])
        self.register_buffer('init_shape', new_dict['model.head.init_shape'])
        self.register_buffer('init_cam', new_dict['model.head.init_cam'])

    def forward(self, x):
        return x

class Model1(nn.Module):
    def __init__(
            self,
            img_res=224,
    ):
        self.inplanes = 64
        super(Model1, self).__init__()
        # pdb.set_trace()

        npose = 24 * 6

        self.resnet = torchvision.models.resnet50(pretrained=True)

        # self.avg_pool = nn.AdaptiveAvgPool2d(8)
        num_neurons = 1000
        self.decpose = nn.Linear(num_neurons, npose)
        self.decshape = nn.Linear(num_neurons, 10)
        self.deccam = nn.Linear(num_neurons, 3)

        self.head = SomeHead()

        # SMPLHead takes estimated pose, shape, cam parameters as input
        # and returns the 3D mesh vertices, 3D/2D joints as output
        self.smpl = SMPLHead(
            focal_length=5000.,
            img_res=img_res
        )

    def forward(self, x):
        # pdb.set_trace()
        batch_size = x.shape[0]
                
        # features = self.avg_pool(x).reshape(batch_size, -1)
        features = self.resnet(x)

        pred_pose = self.decpose(features)
        pred_shape = self.decshape(features)
        pred_cam = self.deccam(features)

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



class Model2(nn.Module):
    def __init__(
            self,
            img_res=224,
    ):
        self.inplanes = 64
        super(Model2, self).__init__()
        npose = 24 * 6

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fnn = nn.Linear(128*8*8, 192)
        self.avg_pool = nn.AdaptiveAvgPool2d(8)
        self.decpose = nn.Linear(192, npose)
        self.decshape = nn.Linear(192, 10)
        self.deccam = nn.Linear(192, 3)


        self.smpl = SMPLHead(
            focal_length=5000.,
            img_res=img_res
        )

    def forward(self, images):
        batch_size = images.shape[0]

        features = self.conv1(images)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.bn1(features)
        features = self.conv2(features)
        features = self.relu(features)
        features = self.maxpool(features)
        
        
        features = features.reshape(batch_size, -1)

        features = self.fnn(features)

        pred_pose = self.decpose(features)
        pred_shape = self.decshape(features)
        pred_cam = self.deccam(features)

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



class Model3(nn.Module):
    def __init__(
            self,
            img_res=224,
    ):
        self.inplanes = 64
        super(Model3, self).__init__()
        # pdb.set_trace()

        self.hmr = create_hmr()

    def forward(self, x):
        # pdb.set_trace()
        batch_size = x.shape[0]
                
        # features = self.avg_pool(x).reshape(batch_size, -1)
        # features = self.resnet(x)

        pred_pose, pred_shape, pred_cam = self.hmr(x)

        # pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)

        # pred_pose = self.decpose(features)
        # pred_shape = self.decshape(features)
        # pred_cam = self.deccam(features)

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        smpl_output = self.smpl(
            rotmat=pred_rotmat,
            shape=pred_shape,
            cam=pred_cam,
            normalize_joints2d=True,
        )

        # # pdb.set_trace()
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

DummyModel = Model3