from .base_options import BaseOptions
import yaml


class FilterOptions(BaseOptions):
    """This class defines options used during finetuning.
    """

    def initialize(self, parser):
        """
        defines additional paramters
        """
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--flip-normal', action="store_false",
                            dest="backfaceCulling", help='flip normal wrt view direction')
        parser.add_argument('--pix2pix', type=str, default="render_PCA_resnet",
                            help="model name for pix2pix")
        parser.add_argument('--recursiveFiltering', action="store_true",
                            help="apply the same filter on the output")
        parser.add_argument('--cloud', nargs='?', help='source cloud')
        parser.add_argument('--im_filter', '-f', default="Pix2PixDenoising", help='filter function')

        parser.set_defaults(steps=[19, 1], learningRates=[
                            2000, 1], projectionRadius=0.1, projectionWeight=0.05,
                            repulsionRadius=0.03, repulsionWeight=0.05,
                            repulsionFreq=1, projectionFreq=1,
                            averageWeight=0.01,
                            camOffset=10, camFocalLength=15, name="filter",
                            backward_bb=100)
        self.initialized = True
        return parser

    def parse(self):
        self.opt = super().parse()
        self.opt.modifiers = ["localNormals", "localPoints"]
        self.opt.isFiltering = True
        with open(self.opt.source, "r") as f:
            targetJson = yaml.load(f)
            if "cmdLineArgs" in targetJson:
                self.parser.set_defaults(**targetJson["cmdLineArgs"])
        self.opt, _ = self.parser.parse_known_args()
        if self.opt.im_filter == "Pix2PixDenoising":
            self.opt.shading = "diffuse"
            self.opt.average_term = True
            self.opt.recursiveFiltering = False
        self.print_options(self.opt)
        return self.opt
