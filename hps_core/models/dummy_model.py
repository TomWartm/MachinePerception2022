import pdb
import torch
import torch.nn as nn
from .head import SMPLHead

from ..utils.geometry import rot6d_to_rotmat, rotmat_to_rot6d

class Bottleneck(nn.Module):
    """
    Redefinition of Bottleneck residual block
    Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return 

class DummyModel(nn.Module):
    def __init__(
            self,
            img_res=224,
    ):
        self.inplanes = 64
        super(DummyModel, self).__init__()
        # pdb.set_trace()

        # bu ne demek ya?
        npose = 24 * 6

        # this is a quite dumb model
        # it does avg pooling on the input image
        # and apply seperate mlps to regress SMPL
        # pose, shape, and cam params
        # mkocabas deneyelim..
        block = Bottleneck
        layers = [3, 4, 6, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)


        # self.avg_pool = nn.AdaptiveAvgPool2d(8)
        # self.decpose = nn.Linear(192, npose)
        # self.decshape = nn.Linear(192, 10)
        # self.deccam = nn.Linear(192, 3)

        # SMPLHead takes estimated pose, shape, cam parameters as input
        # and returns the 3D mesh vertices, 3D/2D joints as output
        self.smpl = SMPLHead(
            focal_length=5000.,
            img_res=img_res
        )

    def forward(self, images):
        batch_size = images.shape[0]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        # features = self.avg_pool(images).reshape(batch_size, -1)

        # pred_pose = self.decpose(features)
        # pred_shape = self.decshape(features)
        # pred_cam = self.deccam(features)

        pred_pose = self.init_pose.expand(batch_size, -1)
        init_shape = self.init_shape.expand(batch_size, -1)
        pred_cam = self.init_cam.expand(batch_size, -1)
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
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

        pdb.set_trace()
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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
