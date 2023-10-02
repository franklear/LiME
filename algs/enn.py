import logging

import higher
import torch
import torch.nn as nn
import transformers

from editable_model import EditableModel
from hooks import hook_model
from lime_utils import L0Embedding
from utils import _logits

LOG = logging.getLogger(__name__)


def fomaml_callback(all_grads):
    return [g.detach() if g is not None else None for g in all_grads]


class ENN(EditableModel):
    def __init__(self, model, config, model_constructor, edit_lrs=None, edit_loss_fn=None, raw_model=None):
        super().__init__(model, config, model_constructor)

        if edit_lrs is None:
            edit_lrs = nn.Parameter(torch.tensor([config.edit_lr] * len(self.config.model.inner_params)))
        self.edit_lrs = edit_lrs

        if edit_loss_fn is not None:
            self.edit_loss_fn = edit_loss_fn

        self.grad_callback = fomaml_callback if config.enn.first_order else lambda x: x

        if raw_model is None:
            with torch.no_grad():
                self.raw_model = model_constructor()
            self.raw_model.eval()
            for p in self.raw_model.parameters():
                p.requires_grad_(False)
        else:
            self.raw_model = raw_model

        if config.lime.enabled:
            self.num_mask = max(config.lime.language_idx.values()) + 1
            self.mask_dict = {}
            for n, p in model.named_parameters():
                if n in config.model.inner_params:
                    if config.lime.mask_l0:
                        mask = L0Embedding(self.num_mask, sum(p.shape), mask_without_scale=config.lime.mask_without_scale)
                    else:
                        mask = torch.nn.Embedding(self.num_mask, sum(p.shape))
                        mask.weight.data.fill_(1)
                    self.mask_dict[n.replace(".", "#")] = mask
            self.mask_dict = torch.nn.ModuleDict(self.mask_dict)

        for n, p in self.model.named_parameters():
            if n not in config.model.inner_params:
                p.requires_grad_(False)

    def outer_parameters(self):
        params = [self.edit_lrs]
        if self.config.lime.enabled:
            params += list(self.mask_dict.parameters())
        for n, p in self.model.named_parameters():
            if n in self.config.model.inner_params:
                params.append(p)
        return params

    def mask_parameters(self):
        p = []
        if self.config.lime.enabled:
            p.extend(list(self.mask_dict.parameters()))
        return p

    def get_state_dict(self):
        return self.state_dict()

    def l0_reg(self):
        if not self.config.lime.enabled or not self.config.lime.mask_l0:
            return torch.tensor(0.0)

        return torch.stack([gt.l0_reg() for gt in self.mask_dict.values()]).mean()

    def pre_step_constrain_parameters(self):
        if self.config.lime.enabled and self.config.lime.mask_l0:
            for gt in self.mask_dict.values():
                gt.pre_step_constrain_parameters()

    def edit(self, batch, condition=None, language_idx=None, detach_history=False, update_params=True, virtual_batch_size=None):
        assert update_params is True
        assert virtual_batch_size is None

        opt = torch.optim.SGD([{"params": p, "lr": None}
                               for (n, p) in self.model.named_parameters() if n in self.config.model.inner_params])
        with torch.enable_grad(), higher.innerloop_ctx(
            self.model,
            opt,
            override={'lr': list(self.edit_lrs)},
            copy_initial_weights=False,
            track_higher_grads=self.training,
            in_place=self.training
        ) as (fmodel, diffopt):
            def lime_wrapped_callback(all_grads):
                all_grads = self.grad_callback(all_grads)

                if language_idx is not None:
                    all_grads_with_mask = []
                    for (n, p), g in zip(fmodel.named_parameters(), all_grads):
                        n = n.replace(".", "#")
                        if n in self.mask_dict:
                            u, v = p.__x__.to(torch.float32), p.__delta__.to(torch.float32)

                            with torch.autograd.profiler.record_function("flatten input"):
                                u_ = u.view(-1, u.shape[-1])
                                v_ = v.view(-1, v.shape[-1])

                                nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(-1)  # Skip batch elements with zero grad
                                u_ = u_[nz_mask]
                                v_ = v_[nz_mask]

                            with torch.autograd.profiler.record_function("get mask"):
                                bs = u.shape[0]
                                assert bs == v.shape[0]
                                seq = 1
                                for x in u.shape[1:-1]:
                                    seq *= x
                                mli = language_idx.unsqueeze(-1).repeat(1, seq).view(-1)
                                mli = mli[nz_mask]
                                m1, m2 = self.mask_dict[n](mli).split([u.shape[-1], v.shape[-1]], -1)
                                if self.config.lime.mask_basic_offset is not None:
                                    m1, m2 = m1 + self.config.lime.mask_basic_offset, m2 + self.config.lime.mask_basic_offset

                            with torch.autograd.profiler.record_function("apply mask"):
                                u_, v_ = u_ * m1, v_ * m2

                            with torch.autograd.profiler.record_function("grad agg"):
                                if isinstance(self.model, transformers.GPT2LMHeadModel):
                                    targ = "ij"
                                else:
                                    targ = "ji"
                                g = torch.einsum(f"bi,bj->{targ}", u_, v_)

                        all_grads_with_mask.append(g)

                    all_grads = all_grads_with_mask

                return all_grads

            if self.config.lime.enabled:
                hook_model(fmodel, self.config.model.inner_params, detach=False, once=True)
                LOG.debug(f"Hooked {len(fmodel.handles) // 2} modules")

            fmodel.eval()
            for edit_step in range(self.config.enn.n_edit_steps):
                output = _logits(fmodel(**batch))
                loss = self.edit_loss_fn(output, batch["labels"])["nll"]
                diffopt.step(loss, grad_callback=lime_wrapped_callback)

            if self.config.lime.enabled:
                for handle in fmodel.handles:
                    handle.remove()
                LOG.debug(f"Removed {len(fmodel.handles)} handles")
                del fmodel.handles

        for n, p in fmodel.named_parameters():
            p.requires_grad_(n in self.config.model.inner_params)

        if not detach_history:
            model_edited = fmodel
        else:
            model_edited = self.model_constructor()
            model_edited.load_state_dict(fmodel.state_dict())
        model_edited.train(self.training)

        return ENN(model_edited, self.config, self.model_constructor, edit_lrs=self.edit_lrs, edit_loss_fn=self.edit_loss_fn, raw_model=self.raw_model), {}, {}
