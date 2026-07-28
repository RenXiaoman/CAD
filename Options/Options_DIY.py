from Options.BaseOptions import BaseOptions


class Options_DIY_AHCDU(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=30, type=int, help='# threads for loading data')
        
        # parser.add_argument('--datapath', type=str, default='dataset/PI-CAI/PI-CAI', help='path of the data')
        # parser.add_argument('--task_name', type=str, default='SegTumor_UNET_PI-CAI', help='the current task name')
        
        
        parser.add_argument('--datapath', type=str, default='dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU', help='path of the data')
        parser.add_argument('--task_name', type=str, default='SegGland_DIY_AHCDU_Wave_Attentaion', help='the current task name')
        
        
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=10, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=True, help='resume training from checkpoint')
        parser.set_defaults(lr=1e-4, gpu_ids='1', epoch=500)
        self.isTrain = True
        return parser
    
    
class Options_DIY_PICAI(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=50, type=int, help='# threads for loading data')
        
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI', help='path of the data')
        parser.add_argument('--task_name', type=str, default='SegGland_DIY_PICAI_Wave_Attentaion', help='the current task name')
        
        
        # parser.add_argument('--datapath', type=str, default='dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU', help='path of the data')
        # parser.add_argument('--task_name', type=str, default='SegGland_DIY_AHCDU_NoAugment', help='the current task name')
        
        
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--batch_size', type=int, default=10, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=True, help='resume training from checkpoint')
        parser.set_defaults(lr=1e-4, gpu_ids='3', epoch=500)
        self.isTrain = True
        return parser


class Options_DIY_Bound_AHCDU(BaseOptions):
    """Training options for DIY with boundary auxiliary supervision on AHCDU."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=30, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU', help='path of the data')
        
       
        parser.add_argument('--task_name', type=str, default='SegGland_DIY_AHCDU_Wave_MSAGAttention_Bound_0.3', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--boundary_weight', type=float, default=0.3, help='weight for boundary auxiliary loss')
        parser.add_argument('--boundary_radius', type=int, default=1, help='3D morphology radius for boundary target')
        parser.add_argument('--batch_size', type=int, default=6, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=False, help='resume training from checkpoint')
        parser.set_defaults(lr=1e-4, gpu_ids='3', epoch=500)
        self.isTrain = True
        return parser


class Options_DIY_Bound_PICAI(BaseOptions):
    """Training options for DIY with boundary auxiliary supervision on PI-CAI."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--num_threads', default=50, type=int, help='# threads for loading data')
        parser.add_argument('--datapath', type=str, default='dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI', help='path of the data')
        parser.add_argument('--task_name', type=str, default='SegGland_DIY_PICAI_Bound_Wave_MSAGAttention_0.2', help='the current task name')
        parser.add_argument('--dice_weight', type=float, default=0.5, help='weight for Dice loss')
        parser.add_argument('--focal_weight', type=float, default=0.5, help='weight for Focal loss')
        parser.add_argument('--boundary_weight', type=float, default=0.2, help='weight for boundary auxiliary loss')
        parser.add_argument('--boundary_radius', type=int, default=1, help='3D morphology radius for boundary target')
        parser.add_argument('--batch_size', type=int, default=6, help='input train batch size')
        parser.add_argument('--resume', type=bool, default=False, help='resume training from checkpoint')
        parser.set_defaults(lr=1e-4, gpu_ids='2', epoch=500)
        self.isTrain = True
        return parser
    
