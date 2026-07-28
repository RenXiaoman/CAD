from Options.BaseOptions import BaseOptions


class Options_AWUNet_AHCDU(BaseOptions):
    """Training options for AW_UNet + BoundaryDoULoss3D on AHCDU."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=40, type=int, help='# threads for loading data')

        parser.add_argument('--datapath', type=str, default='dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU', help='path of the data')
        parser.add_argument('--task_name', type=str, default='SegGland_AWUNet_AHCDU', help='the current task name')

        parser.add_argument('--batch_size', type=int, default=4, help='input train batch size')
        parser.add_argument('--resume', type=str, default='False', help='False, True, or checkpoint path')
        parser.add_argument('--num_res_units', type=int, default=2, choices=[2, 4], help='AW_UNet residual units')
        parser.set_defaults(lr=1e-4, gpu_ids='2', epoch=400)
        self.isTrain = True
        return parser


class Options_AWUNet_PICAI(BaseOptions):
    """Training options for AW_UNet + BoundaryDoULoss3D on PICAI."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=20, type=int, help='# threads for loading data')

        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI', help='path of the data')
        parser.add_argument('--task_name', type=str, default='SegGland_AWUNet_PICAI', help='the current task name')

        parser.add_argument('--batch_size', type=int, default=4, help='input train batch size')
        parser.add_argument('--resume', type=str, default='False', help='False, True, or checkpoint path')
        parser.add_argument('--num_res_units', type=int, default=2, choices=[2, 4], help='AW_UNet residual units')
        parser.set_defaults(lr=1e-4, gpu_ids='3', epoch=400)
        self.isTrain = True
        return parser


class Options_AWUNet_FullPICAI(BaseOptions):
    """Training options for AW_UNet on the full PI-CAI split (Dataset141)."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_FullPICAI', help='models are saved here')
        parser.add_argument('--num_threads', default=20, type=int, help='# threads for loading data')

        parser.add_argument(
            '--datapath',
            type=str,
            default='dataset/PI-CAI/nnUNet_raw/Dataset141_FullPICAI',
            help='path of the data',
        )
        parser.add_argument(
            '--task_name',
            type=str,
            default='SegGland_AWUNet_FullPICAI',
            help='the current task name',
        )

        parser.add_argument('--batch_size', type=int, default=2, help='input train batch size')
        parser.add_argument('--resume', type=str, default='True', help='False, True, or checkpoint path')
        parser.add_argument('--num_res_units', type=int, default=2, choices=[2, 4], help='AW_UNet residual units')
        parser.set_defaults(lr=1e-4, gpu_ids='0', epoch=60, model_save_fre=30)
        self.isTrain = True
        return parser
