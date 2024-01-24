from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of the test examples')
        parser.add_argument('--results_dir', type=str, default='/data/code/inpainting/can/paris', help='saves results here')
        parser.add_argument('--how_many', type=int, default=1, help='how many test images to run')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')

        self.isTrain = False

        return parser
