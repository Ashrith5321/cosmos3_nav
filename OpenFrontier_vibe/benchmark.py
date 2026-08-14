import gc
import os
import csv
import torch
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path

from nav.pointnav_agent import PointnavAgent
from utils.frontier_utils import read_config_yaml
from utils.config_utils import ovon_config, hm3d_config, mp3d_config, DATA_PATH

import habitat


# Was hardcoded to "0", which pins every worker to GPU 0 when running shards in
# parallel. Defaults to "0" so single-worker behaviour is unchanged.
# OF_ALL_GPUS=1: leave every GPU visible and select the device via habitat's
# gpu_device_id + torch.cuda.set_device instead. Masking breaks habitat-sim's
# EGL device matching on many-GPU nodes (EGL reports physical CUDA ids, which
# never match the remapped id 0).
if os.environ.get("OF_ALL_GPUS") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("OF_CUDA_DEVICE", "0")
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def write_metrics(metrics, path="objnav_hm3d.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_episodes", type=int, default=-1)

    p.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="hm3d, mp3d, or OVON",
    )


    p.add_argument(
        "--nickname",
        type=str,
        required=True,
        help="Nickname for the benchmark run",
    )

    p.add_argument(
        "--config",
        type=str,
        default="config/hm3d_navigation.yaml",
        help="OpenFrontier configuration file",
    )

    p.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum number of navigation steps",
    )
    p.add_argument(
        "--max_time", type=int, default=3600, help="Maximum navigation time in seconds"
    )
    p.add_argument(
        "--vis_graph",
        action="store_true",
        default=False,
        help="Visualize the topological graph",
    )
    p.add_argument(
        "--save-images",
        "--save_images",
        dest="save_images",
        action="store_true",
        default=False,
        help="Save intermediate navigation images during benchmark episodes",
    )
    p.add_argument(
        "--unet_weight",
        type=Path,
        default=Path("model_weights/rgbd_11cls.pth"),
        help="Path to UNet model weights",
    )

    p.add_argument(
        "--log_level",
        "-ll",
        type=int,
        default=20,
        help="logging level (0=notset, 10=debug, 20=info...)",
    )

    p.add_argument(
        "--split",
        type=int,
        default=(1, 1),
        nargs=2,
        help="Split number for evaluation in the format #subset_index #total_subsets",
    )

    p.add_argument(
        "--ep-shard",
        type=int,
        default=(0, 1),
        nargs=2,
        help="Episode-level shard within a scene: #shard_index #total_shards. "
        "Each shard evaluates episodes where i %% total == index and writes to "
        "metrics/<scene>.shard<i>of<n>.csv (merge afterwards). Default (0,1)=no sharding.",
    )

    p.add_argument(
        "--output-path",
        type=str,
        required=True,
    )

    p.add_argument(
        "--agent-radius",
        type=float,
        default=None,
        help="Recompute the navmesh with this agent radius in meters "
        "(e.g. 0.05 for a 10x10cm footprint). Height stays at the habitat "
        "default. None = stock navmesh.",
    )

    p.add_argument(
        "--wm-oracle",
        action="store_true",
        default=False,
        help="Log oracle frontier values (navmesh geodesic distance to goal) "
        "alongside predicted utilities for ranking evaluation. Requires a "
        "config with world_model.enabled: true.",
    )

    return p.parse_known_args()[0]


if __name__ == "__main__":
    args = get_args()

    # list all the scenes in objnav path

    benchmark = args.benchmark.lower()
    output_path = args.output_path.lower()

    if benchmark == "ovon":
        directory = Path(DATA_PATH + "ovon/val_unseen/content/")
    elif benchmark == "hm3d":
        directory = Path(DATA_PATH + f"objectnav_hm3d_v2/val/content/")
    elif benchmark == "mp3d":
        directory = Path(DATA_PATH + f"objectnav_mp3d_v1/val/content/")
    else:
        raise ValueError(f"Benchmark {benchmark} not recognized")

    scenes = sorted([f.stem.split(".")[0] for f in directory.glob("*.json.gz")])

    if len(scenes) == 0:
        raise ValueError(f"No scenes found in {directory}")

    # split the scenes based on args.split and select the corresponding subset
    total_splits = args.split[1]
    split_index = args.split[0] - 1
    scenes = [
        scene for i, scene in enumerate(scenes) if i % total_splits == split_index
    ]

    scenes_data = {}
    killed = False
    exception = None

    fn_config = read_config_yaml(args.config)

    probabilities_source = (
        fn_config.get("probabilities_source", "gemini-2.5").lower().split("-")[0]
    )
    detection_source = (
        fn_config.get("detection_source", "gemini-2.5").lower().split("-")[0]
    )
    segmentation_source = (
        fn_config.get("segmentation_source", "sam3").lower().split("-")[0]
    )

    benchmark_nickname = args.nickname

    output_dir = Path(
        f"{output_path}/{benchmark_nickname}_{segmentation_source}_{probabilities_source}_{detection_source}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        if killed:
            raise exception if exception is not None else KeyboardInterrupt()
        
        killed = False
        exception = None

        print("Evaluating Scene: %s" % scene)

        if args.benchmark.lower() == "hm3d":
            habitat_config = hm3d_config(stage=scene, episodes=args.eval_episodes)
        elif args.benchmark.lower() == "mp3d":
            habitat_config = mp3d_config(stage=scene, episodes=args.eval_episodes)
        elif args.benchmark.lower() == "ovon":
            habitat_config = ovon_config(stage=scene, episodes=args.eval_episodes)
        else:
            raise ValueError("Benchmark not recognized: %s" % args.benchmark)

        try:
            if os.environ.get("OF_ALL_GPUS") == "1":
                # explicit device selection with all GPUs visible
                _gpu = int(os.environ.get("OF_CUDA_DEVICE", "0"))
                import torch as _torch

                _torch.cuda.set_device(_gpu)
                from habitat.config import read_write as _rw

                with _rw(habitat_config):
                    habitat_config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                        _gpu
                    )

            habitat_env = habitat.Env(habitat_config)

            if args.agent_radius is not None:
                # Recompute the scene navmesh for a slimmer agent footprint
                # (keeps the default agent height). A smaller radius opens
                # narrow gaps the stock 0.18m cylinder cannot pass.
                import habitat_sim as _hsim

                _settings = _hsim.NavMeshSettings()
                _settings.set_defaults()
                _settings.agent_radius = float(args.agent_radius)
                _settings.agent_height = float(
                    habitat_config.habitat.simulator.agents.main_agent.height
                )
                ok = habitat_env.sim.recompute_navmesh(
                    habitat_env.sim.pathfinder, _settings
                )
                print(
                    f"navmesh recomputed for agent radius {args.agent_radius}m "
                    f"(height {_settings.agent_height}m): {ok}"
                )
        except Exception as e:
            print(e)
            continue

        if args.eval_episodes == -1:
            num_episodes = habitat_env.number_of_episodes
        else:
            # Clamp to the scene's actual pool, matching the sampling clamp in
            # hm3d_config. The episode iterator cycles, so asking for more than
            # the scene holds would re-run early episodes under later indices
            # and record them as distinct results.
            num_episodes = min(args.eval_episodes, habitat_env.number_of_episodes)

        # Inside output dir
        scene_dir = Path.joinpath(output_dir, scene)
        scene_dir.mkdir(parents=True, exist_ok=True)

        ep_shard_idx, ep_shard_tot = int(args.ep_shard[0]), int(args.ep_shard[1])
        main_metrics_path = Path.joinpath(output_dir, "metrics/%s.csv" % scene)
        if ep_shard_tot > 1:
            metrics_dir = Path.joinpath(
                output_dir,
                "metrics/%s.shard%dof%d.csv" % (scene, ep_shard_idx, ep_shard_tot),
            )
        else:
            metrics_dir = main_metrics_path
        metrics_dir.parent.mkdir(parents=True, exist_ok=True)

        def _load_metrics_csv(pth):
            out = []
            if pth.exists():
                with open(pth, mode="r") as csv_file:
                    for row in csv.DictReader(csv_file):
                        out.append(
                            {
                                "episode": int(row["episode"]),
                                "success": float(row["success"]),
                                "spl": float(row["spl"]),
                                "distance_to_goal": float(row["distance_to_goal"]),
                                "object_goal": row["object_goal"],
                                "termination_reason": row["termination_reason"],
                            }
                        )
            return out

        # evaluation_metrics = only what THIS shard owns (its own shard file);
        # already_done = every episode index to skip (this shard's file + when
        # sharding, the shared main csv so we never recompute a good episode).
        evaluation_metrics = _load_metrics_csv(metrics_dir)
        already_done = {m["episode"] for m in evaluation_metrics}
        if ep_shard_tot > 1:
            already_done |= {m["episode"] for m in _load_metrics_csv(main_metrics_path)}

        # episodes assigned to THIS shard
        shard_episode_ids = [
            i for i in range(num_episodes) if i % ep_shard_tot == ep_shard_idx
        ]
        shard_done = len([i for i in shard_episode_ids if i in already_done])

        if shard_done >= len(shard_episode_ids):
            print(
                "All %d shard episodes for scene %s (shard %d/%d) already evaluated. Skipping..."
                % (len(shard_episode_ids), scene, ep_shard_idx, ep_shard_tot)
            )
            habitat_env.close()
            continue


        for i in tqdm(range(num_episodes)):
            if killed:
                break

            habitat_env.reset()

            target = habitat_env.current_episode.object_category
            
            target_name = target.replace(" ", "_")
            folder_name = f"episode-{i}-{target_name}"

            path = scene_dir / folder_name
            make_dir = Path(path)

            if (i % ep_shard_tot != ep_shard_idx) or (i in already_done):
                if i in already_done:
                    print(
                        "Episode %d for scene %s has already been evaluated. Skipping..."
                        % (i, scene)
                    )
                continue

            episode = habitat_env.current_episode

            habitat_agent = PointnavAgent(
                habitat_env,
                args,
                save_dir=path,
                openfrontier_config=fn_config,
                habitat_config=habitat_config,
                scene=scene,
            )
 
            habitat_agent.setup_system()
            habitat_agent.initialize()

            if args.wm_oracle and getattr(habitat_agent, "world_model", None):
                try:
                    from worldmodel.oracle import OracleFrontierEvaluator

                    habitat_agent.world_model.oracle = (
                        OracleFrontierEvaluator.from_episode(
                            habitat_env.sim, episode
                        )
                    )
                except Exception as e:
                    print("Failed to attach WM oracle: %s" % str(e))

            reason = "unknown"
            try:
                killed = False
                exception = None
                while (
                    not habitat_env.episode_over
                    and habitat_agent.navigation_steps <= 500
                ):

                    navigate, reason = habitat_agent.navigation(
                        save_images=args.save_images
                    )
                            
                    habitat_agent.update_video()

                    if not navigate:
                        habitat_env.step(0)

            except Exception as e:
                exception = e
                # If keyboard interrupt, stop evaluation
                if isinstance(e, KeyboardInterrupt):
                    killed = True
                    print("Keyboard Interrupt. Stopping evaluation...")
                    reason = "keyboard_interrupt"
                elif "exhausted" in str(e).lower():
                    killed = True
                    print("All API keys exhausted. Stopping evaluation...")
                    reason = "api_keys_exhausted"
                elif "banned" in str(e).lower():
                    killed = True
                    print("Account banned. Stopping evaluation...")
                    reason = "account_banned"
                else:
                    print("Exception during evaluation of episode %d: %s" % (i, str(e)))
                    reason = "exception_occurred"

                    with open(path / "exception.txt", "w") as f:
                        # Write the full exception traceback to the file
                        import traceback

                        traceback.print_exc(file=f)

            finally:
                try:
                    if getattr(habitat_agent, "world_model", None):
                        try:
                            habitat_agent.world_model.close()
                        except Exception as e:
                            print("World model close failed: %s" % str(e))

                    metrics = habitat_env.get_metrics()

                    if int(metrics["success"]) == 0 and reason == "object_found":
                        reason = "false_positive"

                    episode_metrics = {
                        "episode": i,
                        "success": metrics["success"],
                        "spl": metrics["spl"],
                        "distance_to_goal": metrics["distance_to_goal"],
                        "object_goal": episode.object_category,
                        "termination_reason": reason,
                    }

                    episode_metrics_dir = Path.joinpath(path, "metrics.csv")
                    write_metrics(
                        [episode_metrics],
                        episode_metrics_dir,
                    )

                    habitat_agent.save_trajectory(path)

                    # subsitute if already existing
                    evaluation_metrics = [
                        m for m in evaluation_metrics if m["episode"] != i
                    ]
                    evaluation_metrics.append(episode_metrics)
                    write_metrics(evaluation_metrics, path=metrics_dir)

                    if metrics["success"]:
                        # move to success folder
                        success_path = scene_dir / f"success"

                        success_path.mkdir(parents=True, exist_ok=True)

                        new_folder = success_path / folder_name
                        # if folder exists, remove
                        if new_folder.exists():
                            shutil.rmtree(new_folder)

                        os.rename(path, new_folder)
                    else:
                        # move to failure folder
                        failure_path = scene_dir / f"failure"

                        failure_path.mkdir(parents=True, exist_ok=True)

                        folder_name = folder_name + f"-{reason}/"
                        new_folder = failure_path / folder_name

                        if new_folder.exists():
                            shutil.rmtree(new_folder)

                        os.rename(path, new_folder)
                    
                    del habitat_agent
                    del episode
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass

                except Exception as e:
                    print("Exception during metrics logging: %s" % str(e))

        habitat_env.close()
        del habitat_env
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass

        print("Closed environment for scene: %s" % scene)
