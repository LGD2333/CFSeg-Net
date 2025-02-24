"""
ACC_try1, run in acc2_2

hnc1上无原始x
hnc上k==2
注释hnc1 pointwise
tiao k  小到大

FLOPs:  26.99644384 G
Params: 14.759762 M

add conv between two block,incoder

"""
import os

import torch
import torchvision
from matplotlib import pyplot as plt


# def save_image(tensor, epoch, batch_idx, phase):
#     img_dir = os.path.join('saved_images', phase)
#     if not os.path.exists(img_dir):
#         os.makedirs(img_dir)
#
#     grid = torchvision.utils.make_grid(tensor.cpu(), normalize=True)
#     # 计算所有通道的平均值并进行显示
#     avg_image = grid.mean(dim=0)  # 沿着通道维度取平均
#     plt.imshow(avg_image.permute(1, 2, 0))  # permute 为 HWC 形式
#     plt.title(f'{phase} Epoch {epoch} Batch {batch_idx}')
#     plt.savefig(os.path.join(img_dir, f'epoch_{epoch}_batch_{batch_idx}.png'))
#     plt.close()



class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)

    def forward(self, inp):
        batch_size, num_channels, H, W = inp.size()

        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.sigmoid(self.fc2(out))

        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))

        out = self.bn(out)
        out = self.act(out)

        return out


class HANCLayer(torch.nn.Module):
    """
    Implements Hierarchical Aggregation of Neighborhood Context operation
    试下删掉k，k一直等于2
    """

    def __init__(self, in_chnl, out_chnl, ):
        """
        Initialization

        Args:
            in_chnl (int): number of input channels
            out_chnl (int): number of output channels
            k (int): value of k in HANC
        """

        super(HANCLayer, self).__init__()

        self.cnv = torch.nn.Conv2d(3 * in_chnl, out_chnl, kernel_size=(1, 1))
        self.cnv1 = torch.nn.Conv2d(in_chnl, out_chnl, kernel_size=(1, 1))
        self.act = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm2d(out_chnl)

    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        x = inp

        # 2*2与4*4，进行更改，都用max,在2_1
        if H > 14:
            x = torch.cat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                ],
                dim=1,
            )


            # x = x.view(batch_size, num_channels * 5, H, W)
            x = self.act(self.bn(self.cnv(x)))

        else:
            x = self.act(self.bn(self.cnv1(x)))

        return x


class HANCLayer1(torch.nn.Module):
    """
    目前空洞卷积没有使用group，可以试试
    试下两个空洞对比三个空洞的参数量,要cancat，倾向于两个
    """

    def __init__(self, in_chnl, out_chnl, k):
        """
        Initialization

        Args:
            in_chnl (int): number of input channels
            out_chnl (int): number of output channels
            k (int): value of k in HANC，  用来变换空洞系数
        """
        super(HANCLayer1, self).__init__()

        self.k = k

        # self.cnv = torch.nn.Conv2d(in_chnl, out_chnl, kernel_size=(1, 1))
        self.act = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm2d(out_chnl)

        self.cnv1 = torch.nn.Conv2d(int(in_chnl / 2), int(in_chnl / 2), kernel_size=3, dilation=k, padding=k,
                                    groups=int(in_chnl / 2))  # group去掉
        self.cnv2 = torch.nn.Conv2d(int(in_chnl / 2), int(in_chnl / 2), kernel_size=3, dilation=k + 2, padding=k + 2,
                                    groups=int(in_chnl / 2))  # 改 k
        self.bn1 = torch.nn.BatchNorm2d(int(in_chnl / 2))

    def forward(self, inp):
        batch_size, num_channels, H, W = inp.size()

        x = inp

        # x进行通道分割
        x1, x2 = torch.split(x, [int(num_channels / 2), int(num_channels / 2)], dim=1)

        x = torch.cat(
            [
                self.cnv1(x1),
                self.cnv2(x2),
            ],
            dim=1,
        )
        x = self.channel_shuffle(x, 2)  # 注释
        x = self.act(self.bn(x))

        return x

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        x = x.view(batch_size, groups, channels_per_group, height, width)

        x = x.transpose(1, 2).contiguous()

        x = x.view(batch_size, -1, height, width)

        return x


class Conv(torch.nn.Module):

    def __init__(self, in_chnl, out_chnl, ):

        super(Conv, self).__init__()
        self.act = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm2d(out_chnl)
        self.cnv = torch.nn.Conv2d(in_chnl, out_chnl, kernel_size=3, padding=1,)  # group去掉

    def forward(self, x):
        x = self.cnv(x)
        x = self.act(self.bn(x))

        return x


class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
            self,
            num_in_filters,
            num_out_filters,
            kernel_size,
            stride=(1, 1),
            activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.sqe(self.activation(x))


class Conv2d_channel(torch.nn.Module):
    """
    2D pointwise Convolutional layers
    """

    def __init__(self, num_in_filters, num_out_filters):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.sqe(self.activation(x))


class HANCBlock(torch.nn.Module):
    """
    Encapsulates HANC block
    """

    def __init__(self, n_filts, out_channels, k=3, inv_fctr=3):
        """
        Initialization

        Args:
            n_filts (int): number of filters
            out_channels (int): number of output channel
            activation (str, optional): activation function. Defaults to 'LeakyReLU'.
            k (int, optional): k in HANC. Defaults to 1.
            inv_fctr (int, optional): inv_fctr in HANC. Defaults to 4.
        """

        super().__init__()

        self.conv1 = torch.nn.Conv2d(n_filts, n_filts * inv_fctr, kernel_size=1)
        self.norm1 = torch.nn.BatchNorm2d(n_filts * inv_fctr)

        self.hnc1 = HANCLayer1(n_filts * inv_fctr, n_filts * inv_fctr, k)
        # self.norm = torch.nn.BatchNorm2d(n_filts)

        self.conv2 = torch.nn.Conv2d(
            n_filts * inv_fctr,
            n_filts * inv_fctr,
            kernel_size=3,
            padding=1,
            groups=n_filts * inv_fctr,
        )
        self.norm2 = torch.nn.BatchNorm2d(n_filts * inv_fctr)

        self.hnc = HANCLayer(n_filts * inv_fctr, n_filts)
        self.norm = torch.nn.BatchNorm2d(n_filts)

        self.conv3 = torch.nn.Conv2d(n_filts, out_channels, kernel_size=1)
        self.norm3 = torch.nn.BatchNorm2d(out_channels)

        self.sqe = ChannelSELayer(out_channels)

        self.activation = torch.nn.LeakyReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.hnc1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.hnc(x)

        x = self.norm(x + inp)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.sqe(x)

        return x


class ResPath(torch.nn.Module):
    """
    Implements ResPath-like modified skip connection

    """

    def __init__(self, in_chnls, n_lvl):
        """
        Initialization

        Args:
            in_chnls (int): number of input channels
            n_lvl (int): number of blocks or levels
        """

        super(ResPath, self).__init__()

        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.sqes = torch.nn.ModuleList([])

        self.bn = torch.nn.BatchNorm2d(in_chnls)
        self.act = torch.nn.LeakyReLU()
        self.sqe = torch.nn.BatchNorm2d(in_chnls)

        for i in range(n_lvl):
            self.convs.append(
                torch.nn.Conv2d(in_chnls, in_chnls, kernel_size=(3, 3), padding=1)
            )
            self.bns.append(torch.nn.BatchNorm2d(in_chnls))
            self.sqes.append(ChannelSELayer(in_chnls))

    def forward(self, x):

        for i in range(len(self.convs)):
            x = x + self.sqes[i](self.act(self.bns[i](self.convs[i](x))))

        return self.sqe(self.act(self.bn(x)))


class ACC2_2(torch.nn.Module):
    """
    ACC-UNet model
    """

    def __init__(self, n_channels, n_classes, n_filts=32):
        """
        Initialization

        Args:
            n_channels (int): number of channels of the input image.
            n_classes (int): number of output classes
            n_filts (int, optional): multiplier of the number of filters throughout the model.
                                     Increase this to make the model wider.
                                     Decrease this to make the model ligher.
                                     Defaults to 32.
        """

        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.pool = torch.nn.MaxPool2d(2)

        self.cnv11 = HANCBlock(n_channels, n_filts, k=2, inv_fctr=2)
        self.dia1 = Conv(n_filts, n_filts)
        self.cnv12 = HANCBlock(n_filts, n_filts, k=2, inv_fctr=3)

        self.cnv21 = HANCBlock(n_filts, n_filts * 2, k=2, inv_fctr=3)
        self.dia2 = Conv(n_filts * 2, n_filts * 2)
        self.cnv22 = HANCBlock(n_filts * 2, n_filts * 2, k=3, inv_fctr=3)

        self.cnv31 = HANCBlock(n_filts * 2, n_filts * 4, k=3, inv_fctr=4)
        self.dia3 = Conv(n_filts * 4, n_filts * 4)
        self.cnv32 = HANCBlock(n_filts * 4, n_filts * 4, k=4, inv_fctr=4)

        self.cnv41 = HANCBlock(n_filts * 4, n_filts * 8, k=4, inv_fctr=3)
        self.dia4 = Conv(n_filts * 8, n_filts * 8)
        self.cnv42 = HANCBlock(n_filts * 8, n_filts * 8, k=5, inv_fctr=3)

        self.cnv51 = HANCBlock(n_filts * 8, n_filts * 16, k=2, inv_fctr=3)
        self.dia5 = Conv(n_filts * 16, n_filts * 16)
        self.cnv52 = HANCBlock(n_filts * 16, n_filts * 16, k=2, inv_fctr=3)

        self.rspth1 = ResPath(n_filts, 4)
        self.rspth2 = ResPath(n_filts * 2, 3)
        self.rspth3 = ResPath(n_filts * 4, 2)
        self.rspth4 = ResPath(n_filts * 8, 1)

        self.up6 = torch.nn.ConvTranspose2d(n_filts * 16, n_filts * 8, kernel_size=(2, 2), stride=2)
        self.cnv61 = HANCBlock(n_filts * 8 + n_filts * 8, n_filts * 8, k=2, inv_fctr=3)
        # self.dia6 = HANCLayer1(n_filts * 8, n_filts * 8, k=2)
        self.cnv62 = HANCBlock(n_filts * 8, n_filts * 8, k=2, inv_fctr=3)

        self.up7 = torch.nn.ConvTranspose2d(n_filts * 8, n_filts * 4, kernel_size=(2, 2), stride=2)
        self.cnv71 = HANCBlock(n_filts * 4 + n_filts * 4, n_filts * 4, k=2, inv_fctr=3)
        # self.dia7 = HANCLayer1(n_filts * 4, n_filts * 4, k=2)
        self.cnv72 = HANCBlock(n_filts * 4, n_filts * 4, k=2, inv_fctr=3)  # inv_fctr=34?

        self.up8 = torch.nn.ConvTranspose2d(n_filts * 4, n_filts * 2, kernel_size=(2, 2), stride=2)
        self.cnv81 = HANCBlock(n_filts * 2 + n_filts * 2, n_filts * 2, k=2, inv_fctr=3)
        # self.dia8 = HANCLayer1(n_filts * 2, n_filts * 2, k=2)
        self.cnv82 = HANCBlock(n_filts * 2, n_filts * 2, k=2, inv_fctr=3)

        self.up9 = torch.nn.ConvTranspose2d(n_filts * 2, n_filts, kernel_size=(2, 2), stride=2)
        self.cnv91 = HANCBlock(n_filts + n_filts, n_filts, k=2, inv_fctr=3)
        # self.dia9 = HANCLayer1(n_filts, n_filts, k=2)
        self.cnv92 = HANCBlock(n_filts, n_filts, k=2, inv_fctr=3)

        if n_classes == 1:
            self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=(1, 1))
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Conv2d(n_filts, n_classes + 1, kernel_size=(1, 1))
            self.last_activation = None

    def forward(self, x):

        x1 = x

        x2 = self.cnv11(x1)
        x2 = self.dia1(x2)
        x2 = self.cnv12(x2)

        x2p = self.pool(x2)

        x3 = self.cnv21(x2p)
        x3 = self.dia2(x3)
        x3 = self.cnv22(x3)

        x3p = self.pool(x3)

        x4 = self.cnv31(x3p)
        x4 = self.dia3(x4)
        x4 = self.cnv32(x4)

        x4p = self.pool(x4)

        x5 = self.cnv41(x4p)
        x5 = self.dia4(x5)
        x5 = self.cnv42(x5)

        x5p = self.pool(x5)

        x6 = self.cnv51(x5p)
        x6 = self.dia5(x6)
        x6 = self.cnv52(x6)

        x2 = self.rspth1(x2)
        x3 = self.rspth2(x3)
        x4 = self.rspth3(x4)
        x5 = self.rspth4(x5)

        x7 = self.up6(x6)
        x7 = self.cnv61(torch.cat([x7, x5], dim=1))
        # x7 = self.dia6(x7)
        x7 = self.cnv62(x7)

        x8 = self.up7(x7)
        x8 = self.cnv71(torch.cat([x8, x4], dim=1))
        # x8 = self.dia7(x8)
        x8 = self.cnv72(x8)

        x9 = self.up8(x8)
        x9 = self.cnv81(torch.cat([x9, x3], dim=1))
        # x9 = self.dia8(x9)
        x9 = self.cnv82(x9)

        x10 = self.up9(x9)
        x10 = self.cnv91(torch.cat([x10, x2], dim=1))
        # x10 = self.dia9(x10)
        x10 = self.cnv92(x10)

        if self.last_activation is not None:
            logits = self.last_activation(self.out(x10))

        else:
            logits = self.out(x10)

        return logits