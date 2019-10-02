import argparse
import os
import torch
import json
from .deformation_options import DeformationOptions
from .base_options import BaseOptions


class FinetuneOptions(BaseOptions):
    """This class defines options used during finetuning.
    """

    def initialize(self, parser):
        """
        defines additional paramters
        """
        parser = super().initialize(parser)  # define shared options
        parser.set_defaults(steps=[20, 15], learningRates=[
                            2000, 1], cycles=15,
                            projectionRadius=0.1, projectionWeight=0.02,
                            repulsionRadius=0.05, repulsionWeight=0.05,
                            repulsionFreq=1, projectionFreq=1,
                            cutOffThreshold=1.5,
                            camOffset=9, camFocalLength=15,
                            backwardLocalSizeDecay=0.95)
        self.initialized = True
        return parser

    def parse(self):
        self.opt = super().parse()
        with open(self.opt.ref, "r") as f:
            targetJson = json.load(f)
            if "cmdLineArgs" in targetJson:
                self.parser.set_defaults(**targetJson["cmdLineArgs"])
            if "finetuneArgs" in targetJson:
                self.parser.set_defaults(**targetJson["finetuneArgs"])
        try:
            self.parser.set_defaults(camOffset=0.5*targetJson["cmdLineArgs"]["camOffset"])
        except KeyError:
            pass
        # parser again with new defaults
        self.opt = super().parse()
        self.print_options(self.opt)
        return self.opt
