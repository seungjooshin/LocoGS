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

import torch
from scene.gaussian_model import GaussianModel
import os
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.system_utils import searchForMaxIteration

def convert(dataset : ModelParams, iteration : int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.hash_size, dataset.width, dataset.depth)

        if iteration:
            if iteration == -1:
                load_iteration = searchForMaxIteration(os.path.join(dataset.model_path, "compression"))
            else:
                load_iteration = iteration

        gaussians.load_attributes(os.path.join(dataset.model_path, 'compression/iteration_{}'.format(load_iteration)))
        
        point_cloud_path = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(load_iteration))
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Converting script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Converting " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    convert(model.extract(args), args.iteration)
