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
    ):
        self.n_obs_steps = n_obs_steps
        self.sensor_modalities = sensor_modalities or ["point_cloud", "joint_state"]
        self.obs_deque = deque(maxlen=n_obs_steps + 1)

        self.env_video_fps = env_video_fps
        self.default_eval_episodes = default_eval_episodes

        
    @staticmethod
    def _stack_last_n(all_items,  n_steps):
        all_list = list(all_items)
        assert len(all_list) > 0, "Empty input to _stack_last_n()."
        head = all_list[0]

        if isinstance(head, np.ndarray):
            result = np.zeros((n_steps,) + all_list[-1].shape, dtype=all_list[-1].dtype)
            start_idx = -min(n_steps, len(all_list))
            result[start_idx:] = np.asarray(all_list[start_idx:])
            if n_steps > len(all_list):
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
        assert len(self.obs_deque) > 0, "No observation in deque"
        keys = list(self.obs_deque[-1].keys())
        out: Dict[str, Any] = {}
        for k in keys:
            if k in self.sensor_modalities:
                out[k] = self._stack_last_n((frame[k] for frame in self.obs_deque), self.n_obs_steps)
        # 断言out不为空
        assert len(out) > 0, "Stacked observation dict is empty."
        return out


    def get_nobs(self, device) -> Dict[str, Any]:
        def to_torch(x, *, dtype=None, device=None):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=dtype) if dtype is not None else x.to(device=device)
            if isinstance(x, np.ndarray):
                return torch.as_tensor(x, device=device, dtype=dtype)
            return x
    
        stacked_obs = self.get_stacked_obs()
        nobs = dict_apply(stacked_obs, lambda x: to_torch(x, device=device))
        nobs = dict_apply(nobs, lambda x: x.unsqueeze(0) if torch.is_tensor(x) else x)

        return nobs


    def reset(self):
        self.obs_deque.clear()


    @torch.no_grad()
    def get_action_chunk(self, nobs, agent, denoise_timesteps:int=None) -> np.ndarray:
        action = agent.predict_action(obs_dict=nobs, denoise_timesteps=denoise_timesteps)
        action_chunk = action["control_action"].detach().cpu().numpy().squeeze(0)
        return action_chunk
    

    def eval_one_episode(self, agent, env, episode_seed, denoise_timesteps:int=None, **kwargs):
        obs, info = env.reset(seed=episode_seed, options=kwargs.get("options", None))
        self.reset()
        self.update_obs(obs)

        truncated = False
        task_done_step = 1
        while not truncated:
            nobs = self.get_nobs(device=agent.device)
            action_chunk = self.get_action_chunk(nobs, agent, denoise_timesteps=denoise_timesteps)
            for i in range(action_chunk.shape[0]):
                obs, reward, done, truncated, info = env.step(action_chunk[i])
                self.update_obs(obs)

                if not done:
                    task_done_step += 1
                if truncated:
                    break

        return done, task_done_step
    

    def run(self, agent, denoise_timesteps:int=None, eval_episodes:int=None):
        env = self.make_env()
        eval_seeds = self.get_seed_list()
        eval_episodes = eval_episodes if eval_episodes is not None else self.default_eval_episodes

        num_episodes = min(eval_episodes, len(eval_seeds))
        success_list = []
        task_done_step_list = []
        episode_video_list = []

        for ep_idx in range(num_episodes):
            eval_seed = eval_seeds[ep_idx % len(eval_seeds)]
            try:
                done, task_done_step = self.eval_one_episode(agent, env, eval_seed, denoise_timesteps)
                video = env.get_video()

                postfix = "success" if done else "fail"
                cprint(f"Agent rollout for env seed {eval_seed}: {postfix}ed! Complete task in {task_done_step} steps")

                success_list.append(done)
                if done:
                    task_done_step_list.append(task_done_step) 
                episode_video_list.append({
                    f"episode_{eval_seed}_{postfix}": video
                })
            except Exception as e:
                cprint(f"Error during evaluation with seed {eval_seed}: {e}", "grey")
        
        # 统计指标
        success_rate = float(np.mean(success_list)) if len(success_list) > 0 else 0.0
        avg_steps = int(round(np.mean(task_done_step_list))) if len(task_done_step_list) > 0 else 0
        cprint(f"Eval Agent for {num_episodes} eposides, success rate {success_rate*100.0:1f}%, average complete steps {avg_steps}", "yellow")

        return {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "videos": episode_video_list
        }
    

    def make_env(self):
        raise NotImplementedError


    def get_seed_list(self) -> List[int]:
        raise NotImplementedError

