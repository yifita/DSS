import argparse
import os
import torch
import pdb
from .base_options import BaseOptions
import json


class RenderOptions(BaseOptions):
    """This class defines options used during finetuning.
    """

    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument("--points", nargs="*", help="paths to points")
        parser.add_argument('--rot-axis', '-a', help="rotation axis", default="y")
        parser.set_defaults(output="renders/")
        return parser

    def parse(self):
        self.opt = super().parse()
        with open(self.opt.source, "r") as f:
            targetJson = json.load(f)
            if "cmdLineArgs" in targetJson:
                for key, value in targetJson['cmdLineArgs'].items():
                    if key == "source":
                        continue
                    setattr(self.opt, key, value)
        self.opt = super().parse()
        self.print_options(self.opt)
        return self.opt
