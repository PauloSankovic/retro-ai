class CnnStructure:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = padding
