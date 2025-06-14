import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class PoseExpNet(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False):
        super(PoseExpNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)
        
        self.globalpool = nn.AvgPool2d([1,4])
        self.decouphead_r = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 2, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 3*self.nb_ref_imgs, 1, 1)
        )
        self.decouphead_t = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 2, 1),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, 3*self.nb_ref_imgs, 1, 1)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)

        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        
        flag_decoup = False
        if flag_decoup :
            out_conv7 = self.globalpool(out_conv7)
            pose_r = self.decouphead_r(out_conv7) # b c h w
            pose_t = self.decouphead_t(out_conv7)
            pose_r = pose_r.mean(3).mean(2)  # [2, 6]
            pose_t = pose_t.mean(3).mean(2)
            pose_r_1 = pose_r[:,:3]
            pose_r_2 = pose_r[:,3:]
            pose_t_1 = pose_t[:,:3]
            pose_t_2 = pose_t[:,3:]
            pose_1 = torch.cat((pose_t_1,pose_r_1),1)
            pose_2 = torch.cat((pose_t_2,pose_r_2),1)
            pose = torch.cat((pose_1,pose_2),1)
            
        else:
            pose = self.pose_pred(out_conv7)
            pose = pose.mean(3).mean(2)

        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)
        return pose


if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoseExpNet().to(device)
    summary(model, (9, 416, 128))