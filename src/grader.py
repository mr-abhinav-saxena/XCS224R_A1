#!/usr/bin/env python3
import inspect
import unittest
import argparse
from graderUtil import graded, CourseTestRunner, GradedTestCase
from subprocess import Popen
from subprocess import DEVNULL, STDOUT, check_call
import torch
import torch.nn as nn

import numpy as np
import pickle
import os
import random
from collections import defaultdict

import gymnasium as gym

from itertools import product

import tensorflow.compat.v1 as tf

from autograde_utils import if_text_in_py, text_in_cell, assert_allclose

# Import submission
from submission.xcs224r.infrastructure.pytorch_util import build_mlp, from_numpy
from submission.xcs224r.infrastructure.replay_buffer import ReplayBuffer
from submission.xcs224r.infrastructure.utils import sample_trajectory, sample_trajectories, sample_n_trajectories, convert_listofrollouts
from submission.xcs224r.infrastructure.bc_trainer import BCTrainer
from submission.xcs224r.agents.bc_agent import BCAgent
from submission.xcs224r.policies.MLP_policy import MLPPolicySL
from submission.xcs224r.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from submission.xcs224r.infrastructure.utils import MJ_ENV_KWARGS, MJ_ENV_NAMES

# Import reference solution
if os.path.exists("./solution"):
    from solution.xcs224r.infrastructure.pytorch_util import build_mlp as ref_build_mlp
    from solution.xcs224r.infrastructure.replay_buffer import ReplayBuffer as RefReplayBuffer
    from solution.xcs224r.infrastructure.utils import (
        sample_trajectory as ref_sample_trajectory,
        sample_trajectories as ref_sample_trajectories,
        sample_n_trajectories as ref_sample_n_trajectories
    )
    from solution.xcs224r.infrastructure.bc_trainer import BCTrainer as RefBCTrainer
    from solution.xcs224r.policies.MLP_policy import MLPPolicySL as RefMLPPolicySL
else:
    ref_build_mlp = build_mlp
    RefReplayBuffer = ReplayBuffer
    RefMLPPolicySL = MLPPolicySL
    ref_sample_trajectory = sample_trajectory
    ref_sample_trajectories = sample_trajectories
    ref_sample_n_trajectories = sample_n_trajectories
    RefBCTrainer = BCTrainer

ENV_NAMES = ["Ant", "Walker", "HalfCheetah", "Hopper"]

#########
# HELPER #
#########


def parse_file(file):
    eval_returns = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == "Eval_AverageReturn":
                eval_returns.append(v.simple_value)

    max_average_return = np.max(np.array(eval_returns))
    return max_average_return


def get_scores(rootdir, question):

    scores = defaultdict(float)

    for root, dirs, filelist in os.walk(rootdir):
        # Skip hidden or system folders like __MACOSX
        if "__MACOSX" in root or "/." in root:
            continue

        for file in filelist:
            if file.startswith(".") or "MACOSX" in file:
                continue

        for file in filelist:
            if "event" in file:
                dirname = str.split(root, "/")[-1]

                if dirname.startswith("q1_bc_") and question == "q1":
                    for env in ENV_NAMES:
                        if env in dirname:
                            try:
                                score = parse_file(root + "/" + file)
                            except:
                                print("Error parsing file BC,", file)
                                score = 0
                            if score > scores[env]:
                                scores[env] = score

                if (dirname.startswith("q2_dagger_") or dirname.startswith("q2_bc_")) and question == "q2":
                    for env in ENV_NAMES:
                        if env in dirname:
                            try:
                                score = parse_file(root + "/" + file)
                            except:
                                print("Error parsing file Dagger,", file)

                                score = 0

                            if score > scores[env]:
                                scores[env] = score

    return scores

### BEGIN_HIDE ###
### END_HIDE ###


#########
# TESTS #
#########


class Test_1a(GradedTestCase):

    def setUp(self):

        self.env = gym.make("Ant-v4")

        self.GRADING_RUBRIC_DAGGER = {
            "Ant": 4751.0,
            "HalfCheetah": 4097.0,
            "Walker": 5462.0,
            "Hopper": 3781.0,
        }

    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_2a(GradedTestCase):

    def setUp(self):

        self.GRADING_RUBRIC_DAGGER = {
            "Ant": 4751.0,
            "HalfCheetah": 4097.0,
            "Walker": 5462.0,
            "Hopper": 3781.0,
        }

    ### BEGIN_HIDE ###
    ### END_HIDE ###


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)

if __name__ == "__main__":
    # Parse for a specific test or mode
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test_case",
        nargs="?",
        default="all",
        help="Use 'all' (default), a specific test id like '1-3-basic', 'public', or 'hidden'",
    )
    test_id = parser.parse_args().test_case

    def _flatten(suite):
        """Recursively flatten unittest suites into individual tests."""
        for x in suite:
            if isinstance(x, unittest.TestSuite):
                yield from _flatten(x)
            else:
                yield x

    assignment = unittest.TestSuite()

    if test_id not in {"all", "public", "hidden"}:
        # Run a single specific test
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        # Discover all tests
        discovered = unittest.defaultTestLoader.discover(".", pattern="grader.py")

        if test_id == "all":
            assignment.addTests(discovered)
        else:
            # Filter tests by visibility marker in docstring ("basic" for public tests, "hidden" for hidden tests)
            keyword = "basic" if test_id == "public" else "hidden"
            filtered = [
                t for t in _flatten(discovered)
                if keyword in (getattr(t, "_testMethodDoc", "") or "")
            ]
            assignment.addTests(filtered)

    CourseTestRunner().run(assignment)
