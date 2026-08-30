import torch
from torchvision.models import resnet18
import torchvision
import typing

class FcNet(torch.nn.Module):
    """Simple fully connected network with customizable topology, activations, and regularization."""
    def __init__(
        self,
        dim_output: typing.Optional[int] = None,
        num_hidden_units: typing.Union[typing.List[int], typing.Tuple[int, ...]] = (40, 40),
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_layernorm: bool = False
    ) -> None:
        super(FcNet, self).__init__()

        self.dim_output = dim_output
        self.num_hidden_units = list(num_hidden_units)
        self.activation = activation
        self.dropout_rate = float(dropout_rate)
        self.use_layernorm = bool(use_layernorm)

        self.fc_net = self.construct_network()

    def _get_activation(self) -> torch.nn.Module:
        act = str(self.activation).strip().lower()
        if act in ("leaky_relu", "leakyReLU", "leaky"):
            return torch.nn.LeakyReLU()
        elif act == "elu":
            return torch.nn.ELU()
        elif act == "gelu":
            return torch.nn.GELU()
        elif act == "tanh":
            return torch.nn.Tanh()
        return torch.nn.ReLU()

    def construct_network(self):
        net = torch.nn.Sequential()
        if len(self.num_hidden_units) == 0:
            net.add_module(
                name='classifier',
                module=torch.nn.LazyLinear(out_features=self.dim_output) if self.dim_output is not None else torch.nn.Identity()
            )
            return net

        # Layer 0
        l0_modules = [torch.nn.LazyLinear(out_features=self.num_hidden_units[0])]
        if self.use_layernorm:
            l0_modules.append(torch.nn.LayerNorm(self.num_hidden_units[0]))
        l0_modules.append(self._get_activation())
        if self.dropout_rate > 0.0:
            l0_modules.append(torch.nn.Dropout(p=self.dropout_rate))
        net.add_module(name='layer0', module=torch.nn.Sequential(*l0_modules))

        # Subsequent hidden layers
        for i in range(1, len(self.num_hidden_units)):
            li_modules = [torch.nn.Linear(in_features=self.num_hidden_units[i - 1], out_features=self.num_hidden_units[i])]
            if self.use_layernorm:
                li_modules.append(torch.nn.LayerNorm(self.num_hidden_units[i]))
            li_modules.append(self._get_activation())
            if self.dropout_rate > 0.0:
                li_modules.append(torch.nn.Dropout(p=self.dropout_rate))
            net.add_module(name='layer{0:d}'.format(i), module=torch.nn.Sequential(*li_modules))
        
        # Classifier output layer
        net.add_module(
            name='classifier',
            module=torch.nn.Linear(in_features=self.num_hidden_units[-1], out_features=self.dim_output) if self.dim_output is not None \
                else torch.nn.Identity()
        )

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_net(x)

class CNN(torch.nn.Module):
    """A simple convolutional module networks
    """
    def __init__(self, dim_output: typing.Optional[int] = None, bn_affine: bool = False, stride_flag: bool = True) -> None:
        """Initialize an instance

        Args:
            dim_output: the number of classes at the output. If None,
                the last fully-connected layer will be excluded.
            image_size: a 3-d tuple consisting of (nc, iH, iW)

        """
        super(CNN, self).__init__()

        self.dim_output = dim_output
        self.kernel_size = (3, 3)
        if stride_flag:
            self.stride = 2
        else:
            self.stride = 1
        self.padding = 1
        self.num_channels = (32, 32, 32, 32)
        self.bn_affine = bn_affine
        self.stride_flag = stride_flag
        self.cnn = self.construct_network()
    
    def construct_network(self) -> torch.nn.Module:
        """Construct the network

        """
        net = torch.nn.Sequential()
        temp = torch.nn.Sequential(
            torch.nn.LazyConv2d(
                out_channels=self.num_channels[0],
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=not self.bn_affine
            ),
            torch.nn.BatchNorm2d(
                num_features=self.num_channels[0],
                momentum=1,
                track_running_stats=False,
                affine=self.bn_affine
            ),
            torch.nn.ReLU()
        )
        if not self.stride_flag:
            temp.add_module(name="max-pool-0", module=torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        net.add_module(name='layer0', module=temp)

        for i in range(1, len(self.num_channels)):
            temp = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.num_channels[i - 1],
                    out_channels=self.num_channels[i],
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=not self.bn_affine
                ),
                torch.nn.BatchNorm2d(
                    num_features=self.num_channels[i],
                    momentum=1,
                    track_running_stats=False,
                    affine=self.bn_affine
                ),
                torch.nn.ReLU()
            )
            if not self.stride_flag:
                temp.add_module(name="max-pool-{0:d}".format(i), module=torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

            net.add_module(
                name='layer{0:d}'.format(i),
                module=temp
            )

        net.add_module(name='Flatten', module=torch.nn.Flatten(start_dim=1, end_dim=-1))

        if self.dim_output is None:
            clf = torch.nn.Identity()
        else:
            clf = torch.nn.LazyLinear(out_features=self.dim_output)

        net.add_module(name='classifier', module=clf)

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""
        return self.cnn(x)

class ResNet10(torch.nn.Module):
    def __init__(self, dim_output: typing.Optional[int] = None, bn_affine: bool = False, **kwargs) -> None:
        """Initialize an instance of Resnet-10

        Args:
            dim_output: the number of classes at the output. If None, the last fully-connected layer will be excluded.
        """
        super().__init__()

        self.dim_output = dim_output
        self.bn_affine = bn_affine
        self.dropout_prob = kwargs["dropout_prob"]

        self.net = self.modified_resnet10()

    def modified_resnet10(self) -> torch.nn.Module:
        """Create an instance of Resnet-10 with the batch-norm layer modified
        """
        # initialize a Resnet-10 instance
        try:
            net = torchvision.models.resnet._resnet(
                weights=None,
                progress=False,
                block=torchvision.models.resnet.BasicBlock,
                layers=[1, 1, 1, 1]
            )
        except TypeError:
            net = torchvision.models.resnet._resnet(
                arch="resnet10",
                block=torchvision.models.resnet.BasicBlock,
                layers=[1, 1, 1, 1],
                pretrained=False,
                progress=False
            )

        # the first layer will be a lazy convolutional layer with any input channels
        net.conv1 = torch.nn.LazyConv2d(
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=not self.bn_affine
        )

        # modify batch-norm layer to have momentum 1 and no tracking statistics
        net.bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer1[0].bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer1[0].bn2 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer2[0].bn1 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[0].bn2 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[0].downsample[1] = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer3[0].bn1 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[0].bn2 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[0].downsample[1] = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer4[0].bn1 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[0].bn2 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[0].downsample[1] = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)

        # last layer
        if self.dim_output is not None:
            net.fc = torch.nn.LazyLinear(out_features=self.dim_output)
        else:
            net.fc = torch.nn.Identity()

        # add dropout-2d after layers 1, 2, and 3
        net.maxpool.add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))

        net.layer1[0].add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))
        # net.layer1[1].add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))
        net.layer1.add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))

        net.layer2[0].add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))
        # net.layer2[1].add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))
        net.layer2.add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))

        net.layer3[0].add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))
        # net.layer3[1].add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))
        net.layer3.add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))

        net.layer4[0].add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))
        # net.layer4[1].add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))
        net.layer4.add_module(name='dropout2d', module=torch.nn.Dropout2d(p=self.dropout_prob))

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.net(x)

class ResNet18(torch.nn.Module):
    """A modified version of ResNet-18 that suits meta-learning"""
    def __init__(self, dim_output: typing.Optional[int] = None, bn_affine: bool = False, **kwargs) -> None:
        """
        Args:
            dim_output: the number of classes at the output. If None,
                the last fully-connected layer will be excluded.
        """
        super(ResNet18, self).__init__()

        # self.input_channel = input_channel
        self.dim_output = dim_output
        self.bn_affine = bn_affine
        self.dropout_prob = kwargs["dropout_prob"]

        self.net = self.modified_resnet18()

    def modified_resnet18(self):
        """
        """
        try:
            net = resnet18(weights=None)
        except TypeError:
            net = resnet18(pretrained=False)

        # modify the resnet to suit the data
        net.conv1 = torch.nn.LazyConv2d(
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=not self.bn_affine
        )

        # update batch norm for meta-learning by setting momentum to 1 and no track_running_stats
        net.bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer1[0].bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer1[0].bn2 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer1[1].bn1 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer1[1].bn2 = torch.nn.BatchNorm2d(64, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer2[0].bn1 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[0].bn2 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[0].downsample[1] = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer2[1].bn1 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer2[1].bn2 = torch.nn.BatchNorm2d(128, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer3[0].bn1 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[0].bn2 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[0].downsample[1] = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer3[1].bn1 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer3[1].bn2 = torch.nn.BatchNorm2d(256, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer4[0].bn1 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[0].bn2 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[0].downsample[1] = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)

        net.layer4[1].bn1 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)
        net.layer4[1].bn2 = torch.nn.BatchNorm2d(512, momentum=1, track_running_stats=False, affine=self.bn_affine)

        # last layer
        if self.dim_output is not None:
            net.fc = torch.nn.LazyLinear(out_features=self.dim_output)
        else:
            net.fc = torch.nn.Identity()
        
        # add dropout-2d after layers 1, 2, and 3
        net.layer1.add_module(name="dropout2d", module=torch.nn.Dropout2d(p=self.dropout_prob))
        net.layer2.add_module(name="dropout2d", module=torch.nn.Dropout2d(p=self.dropout_prob))
        net.layer3.add_module(name="dropout2d", module=torch.nn.Dropout2d(p=self.dropout_prob))
        net.layer4.add_module(name="dropout2d", module=torch.nn.Dropout2d(p=self.dropout_prob))

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.net(x)

class MiniCNN(torch.nn.Module):
    def __init__(self, dim_output: typing.Optional[int] = None, bn_affine: bool = False) -> None:
        """Initialize an instance

        Args:
            dim_output: the number of classes at the output. If None,
                the last fully-connected layer will be excluded.
            image_size: a 3-d tuple consisting of (nc, iH, iW)

        """
        super(MiniCNN, self).__init__()

        self.dim_output = dim_output
        self.kernel_size = (3, 3)
        self.stride = (2, 2)
        self.padding = (1, 1)
        self.num_channels = (32, 32, 32, 32)
        self.bn_affine = bn_affine
        self.cnn = self.construct_network()
    
    def construct_network(self) -> torch.nn.Module:
        """Construct the network

        """
        net = torch.nn.Sequential(
            torch.nn.LazyConv2d(
                out_channels=4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            torch.nn.BatchNorm2d(
                num_features=4,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            torch.nn.BatchNorm2d(
                num_features=8,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            torch.nn.BatchNorm2d(
                num_features=16,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),

            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True
            ),
            torch.nn.BatchNorm2d(
                num_features=32,
                momentum=1.,
                affine=False,
                track_running_stats=False
            ),
            torch.nn.ReLU(),

            torch.nn.Flatten(start_dim=1, end_dim=-1)
        )
        if self.dim_output is None:
            clf = torch.nn.Identity()
        else:
            clf = torch.nn.LazyLinear(out_features=self.dim_output)

        net.add_module(name='classifier', module=clf)

        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward"""
        return self.cnn(x)


class LogisticRegression(torch.nn.Module):
    def __init__(self, dim_output: typing.Optional[int] = None, bias: bool = True, **kwargs) -> None:
        super(LogisticRegression, self).__init__()

        self.dim_output = dim_output
        self.bias = bias

        self.net = self.construct_network()

    def construct_network(self) -> torch.nn.Module:
        net = torch.nn.Sequential()
        net.add_module(name='flatten', module=torch.nn.Flatten(
            start_dim=1, end_dim=-1))
        net.add_module(
            name='classifier',
            module=torch.nn.LazyLinear(out_features=self.dim_output, bias=self.bias) if self.dim_output is not None
            else torch.nn.Identity()
        )
        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
