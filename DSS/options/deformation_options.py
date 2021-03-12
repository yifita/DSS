from .base_options import BaseOptions
import json


class DeformationOptions(BaseOptions):
    """
    This class defines options used during finetuning.
    """

    def parse(self):
        self.opt = super().parse()
        with open(self.opt.ref, "r") as f:
            targetJson = json.load(f)
            if "cmdLineArgs" in targetJson:
                self.parser.set_defaults(**targetJson["cmdLineArgs"])
        # parser again with new defaults
        # self.opt, _ = self.parser.parse_known_args()
        self.opt = super().parse()
        self.print_options(self.opt)
        return self.opt
