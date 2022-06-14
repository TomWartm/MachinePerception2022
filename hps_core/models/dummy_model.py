import pdb
import torch
import torch.nn as nn
from hps_core.models.head import SMPLHead
import torchvision
from hps_core.utils.geometry import rot6d_to_rotmat, rotmat_to_rot6d
from hps_core.models.hmr import create_hmr
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

        self.head = SomeHead()

        self.smpl = SMPLHead(
            focal_length=5000.,
            img_res=img_res
        )


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

class Model4(nn.Module):
    """very small model, but use the idea in hmr"""
    def __init__(self, img_res):
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
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3, bias=True)
        
        self.fnn1 = nn.Linear(256 + npose + 13, 256)
        self.drop1 = nn.Dropout()
        self.fnn2 = nn.Linear(256, 192)
        self.drop2 = nn.Dropout()
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
    
    def forward(self, images):    
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
        
        
        features = features.reshape(batch_size, -1)

        pred_pose = self.init_pose.expand(batch_size, -1)
        pred_shape = self.init_shape.expand(batch_size, -1)
        pred_cam = self.init_cam.expand(batch_size, -1)
        for i in range(3):
            xc = torch.cat([features, pred_pose, pred_shape, pred_cam],1)
            # pdb.set_trace()
            xc = self.fnn1(xc)
            xc = self.drop1(xc)
            xc = self.relu(xc)
            xc = self.fnn2(xc)

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


class Model5(nn.Module):
    """resnet 50 + hmr"""

    def __init__(self, img_res):
        self.inplanes = 64
        super(Model5, self).__init__()
        # pdb.set_trace()
        new_dict = add_init_smpl_params_to_dict({})
        
        self.register_buffer('init_pose', new_dict['model.head.init_pose'])
        self.register_buffer('init_shape', new_dict['model.head.init_shape'])
        self.register_buffer('init_cam', new_dict['model.head.init_cam'])
        
        npose = 24 * 6

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        for param in self.resnet.parameters():
            param.requires_grad = False
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=True)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3, bias=True)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.conv4 = nn.Conv2d(256, 512, kernel_size=7, stride=2, padding=3, bias=True)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=7, stride=2, padding=3, bias=True)
        # self.bn3 = nn.BatchNorm2d(128)
        
        self.fnn1 = nn.Linear(2048 + npose + 13, 256)
        self.drop1 = nn.Dropout()
        self.fnn2 = nn.Linear(256, 192)
        self.drop2 = nn.Dropout()
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
    
    def forward(self, images):    
        batch_size = images.shape[0]

        # features = self.conv1(images)
        # features = self.relu(features)
        # features = self.maxpool(features)
        # features = self.bn1(features)
        
        # features = self.conv2(features)
        # features = self.relu(features)
        # features = self.maxpool(features)
        # features = self.bn2(features)
        
        # features = self.conv3(features)
        # features = self.relu(features)
        # features = self.maxpool(features)
        # features = self.bn3(features)
        
        # features = self.conv4(features)

        features = self.resnet(images)
        
        # return features 
        features = features.reshape(batch_size, -1)

        pred_pose = self.init_pose.expand(batch_size, -1)
        pred_shape = self.init_shape.expand(batch_size, -1)
        pred_cam = self.init_cam.expand(batch_size, -1)
        for i in range(3):
            xc = torch.cat([features, pred_pose, pred_shape, pred_cam],1)
            # pdb.set_trace()
            xc = self.fnn1(xc)
            xc = self.drop1(xc)
            xc = self.relu(xc)
            xc = self.fnn2(xc)

            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        # return pred_pose, pred_shape, pred_cam

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



DummyModel = Model5