from torch import nn

class ConvAdapter(nn.Module):
    """
    Small residual conv adapter.
    - zero-init the final up conv so it behaves like identity at init.
    - use_bn default is False to avoid changing backbone BN running stats.
    """
    def __init__(self, channels, reduction=16, kernel_size=3, use_bn=False):
        super().__init__()
        mid = max(4, channels // reduction)
        self.down = nn.Conv2d(channels, mid, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        padding = kernel_size // 2
        # use a cheap spatial conv; could be depthwise for smaller param count
        self.conv = nn.Conv2d(mid, mid, kernel_size=kernel_size, padding=padding, groups=1, bias=False)
        self.up = nn.Conv2d(mid, channels, kernel_size=1, bias=True)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(channels)

        # zero-init up so adapter starts near-identity
        nn.init.constant_(self.up.weight, 0.0)
        if self.up.bias is not None:
            nn.init.constant_(self.up.bias, 0.0)

    def forward(self, x):
        residual = x
        y = self.down(x)
        y = self.act(y)
        y = self.conv(y)
        y = self.act(y)
        y = self.up(y)
        if self.use_bn:
            y = self.bn(y)
        return residual + y