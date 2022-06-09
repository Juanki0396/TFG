
from .base_options import BaseOptions


class ClassifierOptions(BaseOptions):

    def __init__(self):
        super().__init__()

    def read_parameters(self):
        super().read_parameters()

        # Model Options
        self.parser.add_argument("--network", type=str, default="resnet", help="Choose the network architecture: resnet | ")
        self.parser.add_argument("--n_classes", type=int, default=1, help="Set the number of labels to learn")
        self.parser.add_argument("--learning_rate", type=float, default=1e-3, help="Set classifier learning rate")
        self.parser.add_argument("--threshold", type=float, default=0.5, help="Set the threshold to determine a predictions as positive")
        self.parser.add_argument("--loss_function", type=str, default="BinaryCrossEntropy",
                                 help="Set default loss function: BinaryCrossEntropy | CrossEntropy")
        self.parser.add_argument("--metric", type=str, default="Accuracy", help="Set default metric: Accuracy | ")
        self.parser.add_argument("--epochs", type=int, default=10, help="Set the number of epochs to train")
        # Data options
        self.parser.add_argument("--image_size", nargs=2, type=int, default=(64, 64), help="Select the image size")
        self.parser.add_argument("--num_threads", type=int, default=2, help="Select the number of threads dedicated to data IO")
        self.parser.add_argument("--batch_size", type=int, default=16, help="Select the batch size for training")
