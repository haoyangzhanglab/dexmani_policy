import torch
import numpy as np
from termcolor import cprint
from collections import deque
from typing import Any, Dict, List

from dexmani_policy.common.pytorch_util import dict_apply


class BaseRunner:
    def __init__(
        self,
        n_obs_steps: int,
        env_video_fps: int,
        default_eval_episodes: int,
        sensor_modalities: List[str] | None = None,
        clear_cache_freq: int = 25,
    ):
        self.n_obs_steps = n_obs_steps
        self.sensor_modalities = sensor_modalities or ["point_cloud", "joint_state"]
        self.obs_deque = deque(maxlen=n_obs_steps)

        self.env_video_fps = env_video_fps
        self.default_eval_episodes = default_eval_episodes
        self.clear_cache_freq = clear_cache_freq
        self.dropped_keys_warned = False

        
    @staticmethod
    def stack_last_n(all_items, n_steps):
        all_list = list(all_items)
        if len(all_list) == 0:
            raise RuntimeError("Empty input to stack_last_n()")
        head = all_list[0]

        if isinstance(head, np.ndarray):
            result = np.zeros((n_steps,) + all_list[-1].shape, dtype=all_list[-1].dtype)
            start_idx = -min(n_steps, len(all_list))
            result[start_idx:] = np.asarray(all_list[start_idx:])
            if n_steps > len(all_list):
                # 重复最早帧填充（与训练时 episode 起始的 pad 行为语义一致：都是重复最早可用帧）
                result[:start_idx] = result[start_idx]
            return result

        if isinstance(head, torch.Tensor):
            shape = (n_steps,) + tuple(all_list[-1].shape)
            result = torch.zeros(shape, dtype=all_list[-1].dtype, device=all_list[-1].device)
            start_idx = -min(n_steps, len(all_list))
            result[start_idx:] = torch.stack(list(all_list[start_idx:]), dim=0)
            if n_steps > len(all_list):
                result[:start_idx] = result[start_idx]
            return result

        if isinstance(head, str):
            last = all_list[-1]
            return [last] * n_steps

        raise RuntimeError(f"Unsupported observation field type: {type(head)}")    


    def update_obs(self, observation: Dict[str, Any]):
        self.obs_deque.append(observation)
    

    def get_stacked_obs(self) -> Dict[str, Any]:
        if len(self.obs_deque) == 0:
            raise RuntimeError("No observation in deque")
        keys = list(self.obs_deque[-1].keys())
        out: Dict[str, Any] = {}
        dropped_keys = []
        for k in keys:
            if k in self.sensor_modalities:
                out[k] = self.stack_last_n((frame[k] for frame in self.obs_deque), self.n_obs_steps)
            else:
                dropped_keys.append(k)
        if len(out) == 0:
            raise RuntimeError("Stacked observation dict is empty")
        if dropped_keys and not self.dropped_keys_warned:
            cprint(f"⚠️ Dropped obs keys {dropped_keys} (not in sensor_modalities={self.sensor_modalities})", "yellow")
            self.dropped_keys_warned = True
        return out


    def get_obs_batch(self, device) -> Dict[str, Any]:
        def to_torch(x, *, dtype=None, device=None):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=dtype) if dtype is not None else x.to(device=device)
            if isinstance(x, np.ndarray):
                return torch.as_tensor(x, device=device, dtype=dtype)
            return x
    
        stacked_obs = self.get_stacked_obs()
        obs_batch = dict_apply(stacked_obs, lambda x: to_torch(x, device=device))
        obs_batch = dict_apply(obs_batch, lambda x: x.unsqueeze(0) if torch.is_tensor(x) else x)

        return obs_batch


    def reset(self):
        self.obs_deque.clear()
        self.dropped_keys_warned = False


    @torch.no_grad()
    def get_action_chunk(self, obs_batch, agent, denoise_timesteps:int=None) -> np.ndarray:
        action = agent.predict_action(obs_dict=obs_batch, denoise_timesteps=denoise_timesteps)
        return action["control_action"].detach().cpu().numpy().squeeze(0)
    

    def eval_one_episode(self, agent, env, episode_seed, denoise_timesteps:int=None, **kwargs):
        obs, info = env.reset(seed=episode_seed, options=kwargs.get("options", None))
        self.reset()
        self.update_obs(obs)

        truncated = False
        done = False
        prev_done = False
        task_done_step = None  # None 表示未成功完成，区分于成功时的 action_cnt
        device = next(agent.parameters()).device

        while not truncated:
            obs = self.get_obs_batch(device=device)
            action_chunk = self.get_action_chunk(obs, agent, denoise_timesteps=denoise_timesteps)
            for i in range(action_chunk.shape[0]):
                obs, reward, done, truncated, info = env.step(action_chunk[i])
                self.update_obs(obs)

                if done and not prev_done:
                    task_done_step = getattr(env, 'action_cnt', None)
                prev_done = done

                # dexmani_sim 不实现 done auto-reset，done 后继续 step 仅产生多余动作，不会 crash。
                # 与实机 teleoperation 行为一致：无自动终止信号，走满固定步数。
                if truncated:
                    break

        return done, task_done_step
    

    def run(self, agent, denoise_timesteps:int=None, eval_episodes:int=None):
        env = self.make_env()
        eval_seeds = self.get_seed_list()
        eval_episodes = eval_episodes if eval_episodes is not None else self.default_eval_episodes

        if eval_episodes > len(eval_seeds):
            cprint(f"⚠️ eval_episodes ({eval_episodes}) > available seeds ({len(eval_seeds)}), limiting to {len(eval_seeds)}", "yellow")
            eval_episodes = len(eval_seeds)

        num_episodes = eval_episodes
        success_list = []
        task_done_step_list = []
        episode_video_list = []
        episode_details = []  # per-episode: {seed, success, steps}
        env_failed_seeds = []
        completed = 0

        print("=" * 90)

        try:
            seed_idx = 0
            while len(success_list) < num_episodes and seed_idx < len(eval_seeds):
                eval_seed = eval_seeds[seed_idx]
                seed_idx += 1

                try:
                    done, task_done_step = self.eval_one_episode(agent, env, eval_seed, denoise_timesteps)
                    video = env.get_video()
                    completed += 1

                    # 定期清理 GPU 缓存：关闭并重建环境，防止 SAPIEN 渲染器显存累积
                    if self.clear_cache_freq > 0 and completed % self.clear_cache_freq == 0:
                        env.close()
                        env = self.make_env()

                    status = "success" if done else "fail"
                    done_step_str = task_done_step if task_done_step is not None else "N/A"
                    cprint(f"[progress {len(success_list)+1}/{num_episodes}] env seed: {eval_seed}, status: {status}, done step: {done_step_str}", "cyan")

                    success_list.append(done)
                    if done and task_done_step is not None:
                        task_done_step_list.append(task_done_step)
                    episode_details.append({
                        "seed": eval_seed,
                        "success": done,
                        "steps": task_done_step,
                    })
                    if video is not None:
                        episode_video_list.append({
                            f"episode_{eval_seed}": video
                        })
                    else:
                        cprint(f"⚠️ No video for seed {eval_seed}", "yellow")

                except (RuntimeError, ValueError, AttributeError) as e:
                    # 只捕获环境和策略相关的预期异常
                    # dexmani_sim base_env.py:316 raises RuntimeError("Reset Failed for seed {seed}! Unstable objects: ...")
                    # dexmani_sim base_env.py:429/473 raises ValueError("Failed to sample ... positions ...")
                    # AttributeError: 动态导入的 env 可能缺失某些可选属性（如 action_cnt）
                    error_msg = str(e)

                    # 环境初始化失败（跳过该 seed，不计入失败）
                    if "Reset Failed" in error_msg or "Unstable" in error_msg or "Failed to sample" in error_msg:
                        env_failed_seeds.append(eval_seed)
                        cprint(f"⚠️ Seed {eval_seed} env init failed, skipping", "yellow")
                    else:
                        # 策略执行失败（记录为失败 episode）
                        success_list.append(False)
                        episode_details.append({
                            "seed": eval_seed,
                            "success": False,
                            "steps": None,
                            "error": str(e),
                        })
                        cprint(f"❌ Seed {eval_seed} policy failed: {e}", "red")

                except KeyboardInterrupt:
                    # 用户中断（Ctrl+C），立即停止评估
                    cprint(f"\n⚠️ Evaluation interrupted by user at seed {eval_seed}", "yellow")
                    cprint(f"Collected {len(success_list)}/{num_episodes} episodes before interruption", "yellow")
                    break

                except Exception as e:
                    # 未预期的严重错误，打印详细信息后重新抛出
                    cprint(f"\n❌ Unexpected error at seed {eval_seed}: {type(e).__name__}: {e}", "red")
                    import traceback
                    traceback.print_exc()
                    cprint("This is an unexpected error. Please report this issue.", "red")
                    raise  # 重新抛出，让调用者处理

            # 统计指标
            if len(success_list) < num_episodes:
                cprint(f"⚠️ Warning: Only collected {len(success_list)}/{num_episodes} valid episodes (ran out of seeds)", "red")

            success_rate = float(np.mean(success_list)) if len(success_list) > 0 else None
            avg_steps = int(round(np.mean(task_done_step_list))) if len(task_done_step_list) > 0 else None

            sr_str = 'N/A' if success_rate is None else f'{success_rate*100.0:.1f}%'
            avg_steps_str = 'N/A' if avg_steps is None else str(avg_steps)
            if len(env_failed_seeds) > 0:
                cprint(f"[result] Valid: {len(success_list)}/{num_episodes}, Env failed: {len(env_failed_seeds)} seeds, Success rate: {sr_str}, Avg steps (success only): {avg_steps_str}", "yellow")
            else:
                cprint(f"[result] Valid: {len(success_list)}/{num_episodes}, Success rate: {sr_str}, Avg steps (success only): {avg_steps_str}", "yellow")
            print("=" * 90)

        finally:
            env.close()

        return {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "videos": episode_video_list,
            "episode_details": episode_details,
        }
    

    def make_env(self):
        raise NotImplementedError


    def get_seed_list(self) -> List[int]:
        raise NotImplementedError
