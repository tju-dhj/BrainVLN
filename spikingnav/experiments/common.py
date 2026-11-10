"""Shared AllenAct experiment machinery for RoboTHOR PointNav / ObjectNav."""

from __future__ import annotations

import glob
import os
from abc import ABC
from math import ceil
from typing import Any, Dict, List, Optional, Sequence, Type

import gym
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from spikingnav.config import DEFAULT_CONFIG, OBJECTNAV_TARGETS, SpikingNavConfig
from spikingnav.sensors import make_rgb_sensor


def _require_allenact():
    from allenact.algorithms.onpolicy_sync.losses import PPO
    from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
    from allenact.base_abstractions.experiment_config import ExperimentConfig, MachineParams
    from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
    from allenact.base_abstractions.sensor import SensorSuite
    from allenact.base_abstractions.task import TaskSampler
    from allenact.utils.experiment_utils import (
        Builder,
        LinearDecay,
        PipelineStage,
        TrainingPipeline,
        evenly_distribute_count_into_bins,
    )
    from allenact.utils.system import get_logger
    from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
    from allenact_plugins.ithor_plugin.ithor_util import horizontal_to_vertical_fov
    from allenact_plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor
    from allenact_plugins.robothor_plugin.robothor_task_samplers import (
        ObjectNavDatasetTaskSampler,
        PointNavDatasetTaskSampler,
    )
    from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask, PointNavTask

    return {
        "PPO": PPO,
        "PPOConfig": PPOConfig,
        "ExperimentConfig": ExperimentConfig,
        "MachineParams": MachineParams,
        "SensorPreprocessorGraph": SensorPreprocessorGraph,
        "SensorSuite": SensorSuite,
        "TaskSampler": TaskSampler,
        "Builder": Builder,
        "LinearDecay": LinearDecay,
        "PipelineStage": PipelineStage,
        "TrainingPipeline": TrainingPipeline,
        "evenly_distribute_count_into_bins": evenly_distribute_count_into_bins,
        "get_logger": get_logger,
        "GoalObjectTypeThorSensor": GoalObjectTypeThorSensor,
        "horizontal_to_vertical_fov": horizontal_to_vertical_fov,
        "GPSCompassSensorRoboThor": GPSCompassSensorRoboThor,
        "ObjectNavDatasetTaskSampler": ObjectNavDatasetTaskSampler,
        "PointNavDatasetTaskSampler": PointNavDatasetTaskSampler,
        "ObjectNavTask": ObjectNavTask,
        "PointNavTask": PointNavTask,
    }


def build_experiment_class(
    task: str,
    model_kind: str,
    cfg: Optional[SpikingNavConfig] = None,
) -> Type:
    """Return an ExperimentConfig subclass for task x {spiking, ann}."""
    aa = _require_allenact()
    cfg = cfg or DEFAULT_CONFIG
    is_objectnav = task == "objectnav"
    is_spiking = model_kind == "spiking"

    class RoboThorNavExperiment(aa["ExperimentConfig"], ABC):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = cfg
            self.ADVANCE_SCENE_ROLLOUT_PERIOD = None
            self.STEP_SIZE = cfg.step_size
            self.ROTATION_DEGREES = cfg.rotation_degrees
            self.DISTANCE_TO_GOAL = cfg.distance_to_goal
            self.VISIBILITY_DISTANCE = cfg.visibility_distance
            self.STOCHASTIC = True
            self.HORIZONTAL_FIELD_OF_VIEW = cfg.horizontal_fov
            self.CAMERA_WIDTH = cfg.camera_width
            self.CAMERA_HEIGHT = cfg.camera_height
            self.SCREEN_SIZE = cfg.image_size
            self.MAX_STEPS = cfg.max_steps
            self.TARGET_TYPES = tuple(cfg.target_types)
            self.NUM_PROCESSES = int(os.environ.get("SPIKINGNAV_NUM_PROCESSES", "60"))
            all_gpus = list(range(torch.cuda.device_count()))
            gpu_override = os.environ.get("SPIKINGNAV_GPU_IDS")
            if gpu_override:
                all_gpus = [int(x) for x in gpu_override.split(",") if x.strip() != ""]
            if all_gpus and self.NUM_PROCESSES < len(all_gpus):
                all_gpus = all_gpus[: self.NUM_PROCESSES]
            self.TRAIN_GPU_IDS = all_gpus
            self.VALID_GPU_IDS = [all_gpus[-1]] if all_gpus else []
            self.TEST_GPU_IDS = self.VALID_GPU_IDS
            self.REWARD_CONFIG = {
                "step_penalty": -0.01,
                "goal_success_reward": 10.0,
                "failed_stop_reward": 0.0,
                "reached_max_steps_reward": 0.0,
                "shaping_weight": 1.0,
            }
            default_train = (
                "datasets/robothor-objectnav/train"
                if is_objectnav
                else "datasets/robothor-pointnav/train"
            )
            default_eval = (
                "datasets/robothor-objectnav/robustnav_eval"
                if is_objectnav
                else "datasets/robothor-pointnav/robustnav_eval"
            )
            self.TRAIN_DATASET_DIR = os.path.join(os.getcwd(), default_train)
            self.VAL_DATASET_DIR = os.path.join(os.getcwd(), default_eval)
            self.TEST_DATASET_DIR = self.VAL_DATASET_DIR
            self._corruptions: Optional[List[str]] = None
            self._severities: Optional[List[int]] = None
            env_corr = os.environ.get("SPIKINGNAV_CORRUPTION")
            if env_corr:
                self._corruptions = [env_corr]
                self._severities = [int(os.environ.get("SPIKINGNAV_SEVERITY", "5"))]
            self._setup_sensors()
            self.use_x11 = os.environ.get("SPIKINGNAV_USE_X11", "0") == "1"
            self.ENV_ARGS = dict(
                width=self.CAMERA_WIDTH,
                height=self.CAMERA_HEIGHT,
                continuousMode=True,
                applyActionNoise=self.STOCHASTIC,
                agentType="stochastic",
                rotateStepDegrees=self.ROTATION_DEGREES,
                visibilityDistance=self.VISIBILITY_DISTANCE,
                gridSize=self.STEP_SIZE,
                snapToGrid=False,
                agentMode="locobot",
                fieldOfView=aa["horizontal_to_vertical_fov"](
                    horizontal_fov_in_degrees=self.HORIZONTAL_FIELD_OF_VIEW,
                    width=self.CAMERA_WIDTH,
                    height=self.CAMERA_HEIGHT,
                ),
                include_private_scenes=False,
                renderDepthImage=False,
            )
            if not self.use_x11:
                import ai2thor.platform

                self.ENV_ARGS["platform"] = ai2thor.platform.CloudRendering

        def _setup_sensors(self) -> None:
            sensors = [
                make_rgb_sensor(
                    height=self.SCREEN_SIZE,
                    width=self.SCREEN_SIZE,
                    uuid="rgb_lowres",
                    corruptions=self._corruptions,
                    severities=self._severities,
                )
            ]
            if is_objectnav:
                sensors.append(
                    aa["GoalObjectTypeThorSensor"](object_types=self.TARGET_TYPES)
                )
            else:
                sensors.append(aa["GPSCompassSensorRoboThor"]())
            self.SENSORS = sensors

        def monkey_patch_datasets(self, train_dataset, val_dataset, test_dataset):
            if train_dataset:
                self.TRAIN_DATASET_DIR = os.path.join(os.getcwd(), train_dataset)
            if val_dataset:
                self.VAL_DATASET_DIR = os.path.join(os.getcwd(), val_dataset)
            if test_dataset:
                self.TEST_DATASET_DIR = os.path.join(os.getcwd(), test_dataset)

        def monkey_patch_sensor(
            self,
            corruptions=None,
            severities=None,
            **kwargs,
        ):
            self._corruptions = list(corruptions) if corruptions else None
            self._severities = list(severities) if severities else None
            if self._corruptions and "Lower FOV" in self._corruptions:
                self.HORIZONTAL_FIELD_OF_VIEW = self.cfg.lower_fov
                self.ENV_ARGS["fieldOfView"] = aa["horizontal_to_vertical_fov"](
                    horizontal_fov_in_degrees=self.HORIZONTAL_FIELD_OF_VIEW,
                    width=self.CAMERA_WIDTH,
                    height=self.CAMERA_HEIGHT,
                )
            self._setup_sensors()

        @classmethod
        def tag(cls):
            kind = "SpikingNav" if is_spiking else "ANNNav"
            name = "Objectnav" if is_objectnav else "Pointnav"
            return f"{name}-RoboTHOR-{kind}-RGB-DDPPO"

        def training_pipeline(self, **kwargs):
            ppo_steps = (
                int(os.environ.get("SPIKINGNAV_MAX_STEPS", "0"))
                or (cfg.objectnav_steps if is_objectnav else cfg.pointnav_steps)
            )
            ppo_cfg = dict(aa["PPOConfig"])
            ppo_cfg.update(
                clip_param=cfg.clip_param,
                value_loss_coef=cfg.value_loss_coef,
                entropy_coef=cfg.entropy_coef,
            )
            return aa["TrainingPipeline"](
                save_interval=5_000_000,
                metric_accumulate_interval=10_000 if torch.cuda.is_available() else 1,
                optimizer_builder=aa["Builder"](optim.Adam, dict(lr=cfg.learning_rate)),
                num_mini_batch=1,
                update_repeats=cfg.update_repeats,
                max_grad_norm=cfg.max_grad_norm,
                num_steps=cfg.rollout_steps,
                named_losses={"ppo_loss": aa["PPO"](**ppo_cfg)},
                gamma=cfg.gamma,
                use_gae=True,
                gae_lambda=cfg.gae_lambda,
                advance_scene_rollout_period=self.ADVANCE_SCENE_ROLLOUT_PERIOD,
                pipeline_stages=[
                    aa["PipelineStage"](loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
                ],
                lr_scheduler_builder=aa["Builder"](
                    LambdaLR, {"lr_lambda": aa["LinearDecay"](steps=ppo_steps)}
                ),
            )

        @classmethod
        def create_model(cls, **kwargs):
            from spikingnav.models.actor_critic import SpikingNavActorCritic
            from spikingnav.models.ann_nav import ANNNavActorCritic

            task_cls = aa["ObjectNavTask"] if is_objectnav else aa["PointNavTask"]
            action_space = gym.spaces.Discrete(len(task_cls.class_action_names()))
            goal_uuid = (
                "goal_object_type_ind" if is_objectnav else "target_coordinates_ind"
            )
            builder = SpikingNavActorCritic if is_spiking else ANNNavActorCritic
            return builder.build(
                action_space=action_space,
                observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
                goal_sensor_uuid=goal_uuid,
                rgb_uuid="rgb_lowres",
                num_categories=len(OBJECTNAV_TARGETS) if is_objectnav else None,
                cfg=cfg,
            )

        def machine_params(self, mode="train", **kwargs):
            if mode == "train":
                gpu_ids = [] if not torch.cuda.is_available() else self.TRAIN_GPU_IDS
                nprocesses = (
                    1
                    if not torch.cuda.is_available()
                    else aa["evenly_distribute_count_into_bins"](
                        self.NUM_PROCESSES, max(len(gpu_ids), 1)
                    )
                )
                sampler_devices = self.TRAIN_GPU_IDS
            elif mode == "valid":
                nprocesses = 1 if torch.cuda.is_available() else 0
                gpu_ids = [] if not torch.cuda.is_available() else self.VALID_GPU_IDS
                sampler_devices = gpu_ids
            elif mode == "test":
                nprocesses = 15 if torch.cuda.is_available() else 1
                gpu_ids = [] if not torch.cuda.is_available() else self.TEST_GPU_IDS
                sampler_devices = gpu_ids
            else:
                raise NotImplementedError(mode)

            graph = None
            if nprocesses if isinstance(nprocesses, int) else sum(nprocesses):
                graph = aa["SensorPreprocessorGraph"](
                    source_observation_spaces=aa["SensorSuite"](
                        self.SENSORS
                    ).observation_spaces,
                    preprocessors=[],
                )
            return aa["MachineParams"](
                nprocesses=nprocesses,
                devices=gpu_ids,
                sampler_devices=sampler_devices,
                sensor_preprocessor_graph=graph,
            )

        @classmethod
        def make_sampler_fn(cls, **kwargs):
            sampler = (
                aa["ObjectNavDatasetTaskSampler"]
                if is_objectnav
                else aa["PointNavDatasetTaskSampler"]
            )
            return sampler(**kwargs)

        @staticmethod
        def _partition_inds(n: int, num_parts: int):
            return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
                np.int32
            )

        def _device_env_args(self, devices, process_ind):
            if not devices:
                return {}
            local_index = int(devices[process_ind % len(devices)])
            if local_index < 0:
                return {}
            if self.use_x11:
                return {"x_display": f"0.{local_index}"}
            # Local index into CUDA_VISIBLE_DEVICES. CloudRendering then
            # rewrites it; set SPIKINGNAV_THOR_GPU_IDS when that rewrite does
            # not land on the nvidia-smi GPU this process is using.
            return {"gpu_device": local_index}

        def _get_sampler_args_for_scene_split(
            self,
            scenes_dir: str,
            process_ind: int,
            total_processes: int,
            devices: Optional[List[int]],
            seeds: Optional[List[int]],
            deterministic_cudnn: bool,
            allow_oversample: bool = True,
        ) -> Dict[str, Any]:
            path = os.path.join(scenes_dir, "*.json.gz")
            scenes = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(path)]
            if not scenes:
                raise RuntimeError(
                    f"No scene json.gz files in {scenes_dir}. "
                    "Run scripts/download_robothor.sh first."
                )
            if total_processes > len(scenes):
                if not allow_oversample:
                    raise RuntimeError(
                        f"total_processes ({total_processes}) > scenes ({len(scenes)})"
                    )
                scenes = scenes * int(ceil(total_processes / len(scenes)))
                scenes = scenes[: total_processes * (len(scenes) // total_processes)]
            inds = self._partition_inds(len(scenes), total_processes)
            task_cls = aa["ObjectNavTask"] if is_objectnav else aa["PointNavTask"]
            args: Dict[str, Any] = {
                "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
                "max_steps": self.MAX_STEPS,
                "sensors": self.SENSORS,
                "action_space": gym.spaces.Discrete(len(task_cls.class_action_names())),
                "seed": seeds[process_ind] if seeds is not None else None,
                "deterministic_cudnn": deterministic_cudnn,
                "rewards_config": self.REWARD_CONFIG,
                "env_args": {
                    **self.ENV_ARGS,
                    **self._device_env_args(devices, process_ind),
                },
            }
            if is_objectnav:
                args["object_types"] = self.TARGET_TYPES
            return args

        def train_task_sampler_args(self, process_ind, total_processes, devices=None, seeds=None, deterministic_cudnn=False):
            res = self._get_sampler_args_for_scene_split(
                os.path.join(self.TRAIN_DATASET_DIR, "episodes"),
                process_ind,
                total_processes,
                devices,
                seeds,
                deterministic_cudnn,
                allow_oversample=True,
            )
            res["scene_directory"] = self.TRAIN_DATASET_DIR
            res["loop_dataset"] = True
            res["allow_flipping"] = True
            return res

        def valid_task_sampler_args(self, process_ind, total_processes, devices=None, seeds=None, deterministic_cudnn=False):
            res = self._get_sampler_args_for_scene_split(
                os.path.join(self.VAL_DATASET_DIR, "episodes"),
                process_ind,
                total_processes,
                devices,
                seeds,
                deterministic_cudnn,
                allow_oversample=False,
            )
            res["scene_directory"] = self.VAL_DATASET_DIR
            res["loop_dataset"] = False
            return res

        def test_task_sampler_args(self, process_ind, total_processes, devices=None, seeds=None, deterministic_cudnn=False):
            res = self._get_sampler_args_for_scene_split(
                os.path.join(self.TEST_DATASET_DIR, "episodes"),
                process_ind,
                total_processes,
                devices,
                seeds,
                deterministic_cudnn,
                allow_oversample=False,
            )
            res["scene_directory"] = self.TEST_DATASET_DIR
            res["loop_dataset"] = False
            return res

    return RoboThorNavExperiment
