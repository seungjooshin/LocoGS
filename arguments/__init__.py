#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.pcd_path = "none"
        self.data_device = "cuda"
        self.eval = False
        self.hash_size = 19
        self.width = 64
        self.depth = 2
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        # position lr
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        # grid lr
        self.grid_lr_init = 0.01
        self.grid_lr_final = 0.001
        self.grid_lr_delay_steps = 5_000
        self.grid_lr_delay_mult = 0.01
        self.grid_lr_max_steps = 30_000
        # opacity lr
        self.opacity_lr_init = 0.01
        self.opacity_lr_final = 0.001
        self.opacity_lr_delay_steps = 5_000
        self.opacity_lr_delay_mult = 0.01
        self.opacity_lr_max_steps = 30_000
        # feature_rest lr
        self.feature_rest_lr_init = 0.0025
        self.feature_rest_lr_final = 0.00025
        self.feature_rest_lr_delay_steps = 5_000
        self.feature_rest_lr_delay_mult = 0.01
        self.feature_rest_lr_max_steps = 30_000
        # scaling lr
        self.scaling_lr_init = 0.01
        self.scaling_lr_final = 0.001
        self.scaling_lr_delay_steps = 5_000
        self.scaling_lr_delay_mult = 0.01
        self.scaling_lr_max_steps = 30_000
        # rotation lr
        self.rotation_lr_init = 0.002
        self.rotation_lr_final = 0.0002
        self.rotation_lr_delay_steps = 5_000
        self.rotation_lr_delay_mult = 0.01
        self.rotation_lr_max_steps = 30_000
        # features_dc lr
        self.feature_dc_lr = 0.0025
        # features_base lr
        self.scaling_base_lr = 0.005
        # mask lr
        self.mask_lr = 0.01
        self.sh_mask_lr = 0.01
        # others
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_mask = 0.004
        self.lambda_sh_mask = 0.0001
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.prune_interval = 1000
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
