import copy
import io
import json
from collections import Counter
import torch
from dexmani_policy.agents.obs_encoder.rgb.dino import DINO
from dexmani_policy.training.ema_model import EMAModel

torch.manual_seed(42); torch.set_num_threads(1)
model=DINO(model_name='facebook/dinov2-small',tune_mode='lora').cuda().train()
ema=copy.deepcopy(model).eval(); updater=EMAModel(ema,power=.75)
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
image=torch.rand(2,3,224,224,device='cuda')
report={'scope':'DINO encoder only; synthetic fixed pixels; current LoRA recipe; no policy quality claim',
        'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
        'parameters':dict(Counter((('trainable' if p.requires_grad else 'frozen')+' '+str(p.dtype)) for p in model.parameters())),
        'steps':[]}
for step in range(3):
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    ebefore={n:p.detach().clone() for n,p in ema.named_parameters() if n in before}
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=model(image)['patch_tokens'].float().square().mean()
    loss.backward()
    grad=sum(p.grad.count_nonzero().item() for p in model.parameters() if p.grad is not None)
    opt.step(); updater.step(model)
    report['steps'].append({'loss':float(loss),'nonzero_grad_elements':grad,
         'changed_trainable_fraction':sum((p!=before[n]).count_nonzero().item() for n,p in model.named_parameters() if n in before)/sum(p.numel() for p in before.values()),
         'changed_ema_fraction':sum((p!=ebefore[n]).count_nonzero().item() for n,p in ema.named_parameters() if n in ebefore)/sum(p.numel() for p in ebefore.values())})
# Existing loop/foreach numerical diagnostic using the actual encoder's tensors.
other=copy.deepcopy(ema); fast=EMAModel(other,power=.75,foreach=True); fast.optimization_step=updater.optimization_step
updater.step(model); fast.step(model)
max_error=max((a.float()-b.float()).abs().max().item() for a,b in zip(ema.parameters(),other.parameters()))
report['ema_loop_foreach_max_abs_error']=max_error
buffer=io.BytesIO(); torch.save(model.state_dict(),buffer); buffer.seek(0)
model.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
report['strict_restore']=True
report['peak_cuda_mib']=torch.cuda.max_memory_allocated()/1024**2
print(json.dumps(report))
