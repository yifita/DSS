import argparse
import os
import torch
import yaml


class BaseOptions():
    """This class defines options used for basic inverse rendering, e.g. large deformation.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('source', metavar="source", nargs='?',
                            default="example_data/scenes/sphere.json",
                            help='json|config file defining scenes initialization')
        parser.add_argument('-t', "--target", dest="ref", nargs='?',
                            default="example_data/scenes/bunny.json", help='reference scene.')
        parser.add_argument('-d', '--device', dest='device', default='cuda:0',
                            help='Device to run the computations on, options: cpu, cuda:{ID}')
        parser.add_argument('--name', default="experiment")
        parser.add_argument('-o', '--output', default="./learn_examples")
        parser.add_argument('-sS', '--startingStep', type=int, default=0)
        parser.add_argument('-C', '--cycles', type=int, default=12,
                            help="number of (step_point, step_normal) optimization cycles")
        parser.add_argument('--modifiers', type=str, nargs='+',
                            default=['localNormals', 'localPoints'])
        parser.add_argument('--steps', type=int, nargs='+', default=[15, 25])
        parser.add_argument('--learningRates', type=float,
                            nargs='+', default=[2000, 5])
        parser.add_argument('--clip', dest="clipGrad",
                            type=float, default=0.01, help='clip gradient')
        parser.add_argument('--verbose', action="store_true",
                            help='more prints')
        parser.add_argument('--debug', action="store_true",
                            help="log debugging information")
        parser.add_argument('--type', default="DSS",
                            help="DSS or Baseline", choices=["DSS", "Baseline"])
        parser.add_argument('--width', type=int,
                            default=256, help="image width")
        parser.add_argument('--height', type=int,
                            default=256, help="image height")
        parser.add_argument('--sv', default=128, help="view scale")
        parser.add_argument('-k', '--topK', dest="mergeTopK",
                            type=int, default=5, help='topK for merging depth')
        parser.add_argument('-mT', '--mergeThreshold', type=float,
                            default=0.05, help='threshold for merging depth')
        parser.add_argument('--img-loss-type',
                            choices=["SMAPE", "L1", "L2"], default="SMAPE")
        parser.add_argument('--no-z', dest='considerZ', action="store_false",
                            help='do not optimize Z in backward (default false)')
        parser.add_argument('-rR', '--repulsionRadius', type=float,
                            default=0.05, help='radius for repulsion loss')
        parser.add_argument('-rW', '--repulsionWeight', type=float,
                            default=0.03, help='weight for repulsion loss')
        parser.add_argument('-pR', '--projectionRadius', type=float,
                            default=0.3, help='radius for projection loss')
        parser.add_argument('-pW', '--projectionWeight', type=float,
                            default=0.05, help='weight for projection loss')
        parser.add_argument('-aW', '--averageWeight', type=float,
                            default=0, help='weight for average term')
        parser.add_argument('--average-term', action="store_true",
                            help="apply average term")
        parser.add_argument('-iW', '--imageWeight', type=float,
                            default=1, help='weight for projection loss')
        parser.add_argument('-fR', '--repulsionFreq', type=int,
                            default=1, help='frequency for repulsion term')
        parser.add_argument('-fP', '--projectionFreq', type=int,
                            default=2, help='frequency for denoising term')
        parser.add_argument('-c', '--genCamera', type=int,
                            default=12, help='number of random cameras')
        parser.add_argument(
            '--cameraFile', default="example_data/pointclouds/sphere_300.ply")
        parser.add_argument('-cO', '--camOffset', type=float,
                            default=15, help='depth offset for generated cameras')
        parser.add_argument('-cF', '--camFocalLength', type=float,
                            default=15, help='focal length for generated cameras')
        parser.add_argument('--cutOffThreshold',
                            type=float, default=1, help='cutoff threshold')
        parser.add_argument('--Vrk_h', type=float, default=0.02, help='standard deviation for V_r^h in EWA')
        parser.add_argument('--backwardLocalSize', default=128,
                            type=int, help='window size for computing pixel loss')
        parser.add_argument('--backwardLocalSizeDecay', default=0.9,
                            type=float, help='decay for backward window size after each cycle')
        parser.add_argument('--baseline', action="store_true",
                            help="use baseline depth renderer")
        parser.add_argument('--sharpnessSigma', default=60,
                            type=float, help="sharpness sigma for weighted PCA")
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='DSS optimization')
            parser = self.initialize(parser)

            # save and return the parser
            self.parser = parser
            # get the basic options
            opt, _ = self.parser.parse_known_args()
            return opt

        return self.parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        opt_dict = {}
        for k, v in sorted(vars(opt).items()):
            opt_dict[str(k)] = str(v)
            comment = ''
            default = self.parser.get_default(k)
            if str(k) == "device":
                opt_dict[str(k)] = str(v)
            else:
                opt_dict[k] = v
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.output, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        opt_file_name = os.path.join(expr_dir, 'opt.yaml')
        with open(opt_file_name, 'wt') as opt_file:
            yaml.dump(opt_dict, opt_file)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        device, isCuda = parse_device(opt.device)
        opt.device = device
        torch.cuda.set_device(opt.device)
        self.opt = opt
        return self.opt


def parse_device(device):
    if "cuda" in device:
        device = torch.device(device)
        isCpu = False
    elif device == 'cpu':
        device = torch.device('cpu')
        isCpu = True
    else:
        print("Unknown device name " + str(device) + ", falling back to cpu")
        device = torch.device('cuda:0')
        isCpu = False
    return (device, isCpu)
