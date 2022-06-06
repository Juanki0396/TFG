
from .base_options import BaseOptions


class CycleGanOptions(BaseOptions):

    def __init__(self):
        super().__init__()

    def read_parameters(self):

        super().read_parameters()

        self.parser.add_argument("--net_G", type=str, default="resnet", help="Set the network for the generator: resnet  | unet")
        self.parser.add_argument("--net_D", type=str, default="patch", help="Set the network for the generator: patch  | pixel")
        self.parser.add_argument("--learning_rate", type=float, default=0.0002, help="Set the initial learning rate")
        self.parser.add_argument("--beta", type=float, default=0.99, help="Set beta parameter for Adam optimizer")
        self.parser.add_argument("--epochs_constant", type=int, default=100, help="Set the number of epochs with constant lr")
        self.parser.add_argument("--epochs_decay", type=int, default=100, help="Set the number of epochs in which the lr deacays to 0")
        self.parser.add_argument("--lr_constant", action="store_true", help="Delete the lr scheduler")
        # Data options
        self.parser.add_argument("--image_size", nargs=2, type=int, default=(64, 64), help="Select the image size")
        self.parser.add_argument("--num_threads", type=int, default=5, help="Select the number of threads dedicated to data IO")
        self.parser.add_argument("--batch_size", type=int, default=16, help="Select the batch size for training")
