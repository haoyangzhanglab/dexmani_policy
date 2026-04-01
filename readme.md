# 训练命令
```python
python train.py --config-name dp3 seed=233 task_name=multi_grasp
```

# 测试命令
```python
python eval_sim.py --agent-name dp3 --task-name multi_grasp --exp-name 2026-04-01_11-18_233
```

# Wandb结果同步
```bash
wandb login
wandb sync ./wandb/offline-run-20260401_111839-m6zq0mtq
wandb sync ./wandb --include-offline --sync-all     #一次性上传整个目录
```