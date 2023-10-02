import functools
import logging
from collections import defaultdict

import higher
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from higher.patch import monkeypatch as make_functional

import nn as local_nn
from editable_model import EditableModel
from hooks import hook_model
from lime_utils import L0Embedding
from losses import masked_log_probs
from utils import _logits, _inner_params, shift_targets, parent_module

LOG = logging.getLogger(__name__)


def update_counter(x, m, s, k):
    new_m = m + 1 / k * (x - m)
    new_s = s + (k - 1) / k * (x - m) ** 2

    return new_m, new_s


class GradientTransform(nn.Module):
    def __init__(self, x_dim: int, delta_dim: int, cfg, n_modes = None):
        super().__init__()

        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.cfg = cfg
        if cfg.mend.combine and (cfg.mend.one_sided or cfg.mend.x_only or cfg.mend.delta_only):
            raise ValueError("mend.combine cannot be used with one-sided MEND variants")

        if cfg.mend.norm:
            if cfg.mend.norm_type in ["bn"]:
                self.bn_u = torch.nn.BatchNorm1d(x_dim, momentum=None, affine=False, eps=1e-10)
                self.bn_v = torch.nn.BatchNorm1d(delta_dim, momentum=None, affine=False, eps=1e-10)
            elif cfg.mend.norm_type in ["ln"]:
                self.ln_u = torch.nn.LayerNorm(x_dim, elementwise_affine=True)
                self.ln_v = torch.nn.LayerNorm(delta_dim, elementwise_affine=True)
            else:
                self.norm_init = False
                self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
                self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
                self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
                self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
                self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
                self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))
                self.register_buffer("k", torch.full((1,), float("nan")))

        MlpClass = getattr(local_nn, cfg.mend.mlp_class)
        LOG.info(f"Building Gradient Transform with MLP class {MlpClass}")

        def delta_net():
            return MlpClass(delta_dim, delta_dim, delta_dim * 2, cfg.mend.n_hidden, init=cfg.mend.init, act=cfg.mend.act, rank=cfg.mend.rank, n_modes=n_modes)

        def x_net():
            return MlpClass(x_dim, x_dim, x_dim * 2, cfg.mend.n_hidden, init=cfg.mend.init, act=cfg.mend.act, rank=cfg.mend.rank, n_modes=n_modes)

        def combined_net():
            return MlpClass(delta_dim + x_dim, delta_dim + x_dim, (delta_dim + x_dim) * 2,
                            cfg.mend.n_hidden, init=cfg.mend.init, act=cfg.mend.act, rank=cfg.mend.rank, n_modes=n_modes)

        def ID():
            return lambda x, mode=None: x

        if cfg.mend.combine:
            self.mlp = combined_net()
        elif cfg.mend.one_sided:
            if x_dim > delta_dim:
                self.mlp1, self.mlp2 = ID(), delta_net()
            else:
                self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.mend.x_only:
            self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.mend.delta_only:
            self.mlp1, self.mlp2 = ID(), delta_net()
        else:
            self.mlp1, self.mlp2 = x_net(), delta_net()

        if cfg.lime.enabled:
            self.num_mask = max(cfg.lime.language_idx.values()) + 1
            num_mask = self.num_mask
            if n_modes is not None:
                num_mask *= n_modes

            if cfg.lime.mask_l0:
                self.mask = L0Embedding(num_mask, delta_dim + x_dim, mask_without_scale=cfg.lime.mask_without_scale)
            else:
                self.mask = torch.nn.Embedding(num_mask, delta_dim + x_dim)
                self.mask.weight.data.fill_(1)

    def l0_reg(self):
        if not self.cfg.lime.enabled or not self.cfg.lime.mask_l0:
            return torch.tensor(0.0)

        return self.mask.l0_reg()

    def pre_step_constrain_parameters(self):
        if self.cfg.lime.enabled and self.cfg.lime.mask_l0:
            self.mask.pre_step_constrain_parameters()

    def forward(self, u, v, param_idx=None, language_idx=None, virtual_batch_size=None):
        u, v = u.to(torch.float32), v.to(torch.float32)

        with torch.autograd.profiler.record_function("flatten input"):
            u_ = u.view(-1, u.shape[-1])
            v_ = v.view(-1, v.shape[-1])

            nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(-1)  # Skip batch elements with zero grad
            u_ = u_[nz_mask]
            v_ = v_[nz_mask]

        if self.cfg.lime.enabled:
            with torch.autograd.profiler.record_function("get mask"):
                assert language_idx is not None, "need `language_idx` for language specific masks"
                if param_idx is not None:
                    language_idx = param_idx * self.num_mask + language_idx

                bs = u.shape[0]
                assert bs == v.shape[0]
                seq = 1
                for x in u.shape[1:-1]:
                    seq *= x
                mli = language_idx.unsqueeze(-1).repeat(1, seq).view(-1)
                mli = mli[nz_mask]
                m1, m2 = self.mask(mli).split([u.shape[-1], v.shape[-1]], -1)
                if self.cfg.lime.enabled and self.cfg.lime.mask_basic_offset is not None:
                    m1, m2 = m1 + self.cfg.lime.mask_basic_offset, m2 + self.cfg.lime.mask_basic_offset

        if self.cfg.lime.enabled and self.cfg.lime.mask_type in ["pre"]:
            with torch.autograd.profiler.record_function("apply mask pre"):
                u_, v_ = u_ * m1, v_ * m2

        with torch.autograd.profiler.record_function("norm"):
            if self.training:
                if self.cfg.mend.norm_type in [None]:
                    with torch.autograd.profiler.record_function("token norm train"):
                        with torch.no_grad():
                            for idx in range(u_.shape[0]):
                                if not self.norm_init:
                                    self.u_mean = u_[idx].clone().detach()
                                    self.v_mean = v_[idx].clone().detach()
                                    self.u_s.zero_()
                                    self.v_s.zero_()
                                    self.k[:] = 1
                                    self.norm_init = True
                                else:
                                    self.k = self.k + 1
                                    self.u_mean, self.u_s = update_counter(u_[idx], self.u_mean, self.u_s, self.k)
                                    self.v_mean, self.v_s = update_counter(v_[idx], self.v_mean, self.v_s, self.k)

                            if self.k < 2:
                                raise RuntimeError(f"Can't perform normalization with only {self.k} samples so far")
                            self.u_std = (self.u_s / (self.k - 1)) ** 0.5
                            self.v_std = (self.v_s / (self.k - 1)) ** 0.5

            if self.cfg.mend.norm:
                if self.cfg.mend.norm_type in ["bn"]:
                    with torch.autograd.profiler.record_function("torch bn"):
                        u_input = self.bn_u(u_)
                        v_input = self.bn_v(v_)
                elif self.cfg.mend.norm_type in ["ln"]:
                    with torch.autograd.profiler.record_function("torch ln"):
                        u_input = self.ln_u(u_)
                        v_input = self.ln_v(v_)
                else:
                    with torch.autograd.profiler.record_function("token norm apply"):
                        u_input = (u_ - self.u_mean) / (self.u_std + 1e-7)
                        v_input = (v_ - self.v_mean) / (self.v_std + 1e-7)
            else:
                u_input = u_
                v_input = v_

        if self.cfg.lime.enabled and self.cfg.lime.mask_type in ["pre_after_norm"]:
            with torch.autograd.profiler.record_function("apply mask pre_after_norm"):
                u_input, v_input = u_input * m1, v_input * m2

        with torch.autograd.profiler.record_function("map grad"):
            if self.cfg.mend.combine:
                output = self.mlp(torch.cat((u_input, v_input), -1), mode=param_idx)
                out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
            else:
                out1, out2 = self.mlp1(u_input, mode=param_idx), self.mlp2(v_input, mode=param_idx)

        if self.cfg.lime.enabled and self.cfg.lime.mask_type in ["post"]:
            with torch.autograd.profiler.record_function("apply mask post"):
                out1, out2 = out1 * m1, out2 * m2

        if virtual_batch_size is not None:
            assert nz_mask.dim() == 1
            nz_mask_as_batched = nz_mask.split(nz_mask.shape[0] // u.shape[0] * virtual_batch_size)
            split_sizes = torch.stack([x.sum() for x in nz_mask_as_batched])
            pad_len = split_sizes.max() - split_sizes

            def split_and_pad(t):
                return [
                    torch.nn.functional.pad(t_s, (0, 0, 0, l_s))
                    for t_s, l_s in zip(t.split_with_sizes(split_sizes.tolist()), pad_len)
                ]

            out1, out2 = split_and_pad(out1), split_and_pad(out2)

        return out1, out2


class MEND(EditableModel):
    def get_shape(self, p):
        # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
        return p.shape if isinstance(self.model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])

    def __init__(self, model, config, model_constructor, mend=None, edit_lrs=None, mean_grads=None):
        super().__init__(model, config, model_constructor)

        if edit_lrs is None:
            edit_lrs = nn.Parameter(torch.tensor([config.edit_lr] * len(self.config.model.inner_params)))
        self.edit_lrs = edit_lrs

        if config.mend.shared:
            shape_dict = defaultdict(list)
            for n, p in _inner_params(model.named_parameters(), self.config.model.inner_params):
                shape_dict[self.get_shape(p)].append(n)
            self.shape_dict = shape_dict

        if mend is None:
            if not config.mend.shared:
                self.mend = nn.ModuleDict({
                    n.replace(".", "#"): GradientTransform(*self.get_shape(p), config)
                    for (n, p) in _inner_params(model.named_parameters(), self.config.model.inner_params)
                })
            else:
                self.mend = nn.ModuleDict({
                    str(tuple(s)): GradientTransform(*s, config, len(shape_dict[s]))
                    for s in shape_dict.keys()
                })
        else:
            self.mend = mend

        for n, p in self.model.named_parameters():
            if p.is_leaf:
                p.requires_grad_(False)

        self.mean_grads = mean_grads

        _edit_loss_fn = functools.partial(masked_log_probs, shift=shift_targets(self.config))
        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = _edit_loss_fn

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(prefix=prefix, keep_vars=keep_vars)  # Get default state dict
        model_keys = self.model.state_dict(prefix=prefix, keep_vars=keep_vars).keys()  # Remove model params
        for k in model_keys:
            del state_dict[f"model.{k}"]
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        res = super().load_state_dict(state_dict, False)
        # We should only have missing keys for the model, and no unexpected keys
        assert len([k for k in res.missing_keys if not k.startswith("model.")]) == 0, "Should only have missing keys for model."
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def outer_parameters(self):
        return list(self.mend.parameters()) + [self.edit_lrs]

    def mask_parameters(self):
        p = []
        if self.config.lime.enabled:
            for gt in self.mend.values():
                p.extend(list(gt.mask.parameters()))
        return p

    def l0_reg(self):
        if not self.config.lime.enabled or not self.config.lime.mask_l0:
            return torch.tensor(0.0)

        return torch.stack([gt.l0_reg() for gt in self.mend.values()]).mean()

    def pre_step_constrain_parameters(self):
        if self.config.lime.enabled and self.config.lime.mask_l0:
            for gt in self.mend.values():
                gt.pre_step_constrain_parameters()

    def _forward_with_for(self, virtual_batch_size, inputs, kwargs):
        # a naive implementation
        pset = set(self.config.model.inner_params)
        raw_params = [p.clone() if n in pset else p for n, p in self.model.named_parameters()]

        ret = []
        for i, mean_grads in enumerate(self.mean_grads):
            with torch.autograd.profiler.record_function("for edited params update"):
                updates = {n: lr * g for lr, (n, g) in
                           zip(self.edit_lrs, filter(lambda t: "#" not in t[0], mean_grads.items()))}
                assert set(updates.keys()) == pset
                new_params = raw_params.copy()
                for j, (n, p) in enumerate(self.model.named_parameters()):
                    if n in pset:
                        new_params[j] = raw_params[j] + updates[n]
                self.model.update_params(new_params)

            if virtual_batch_size is None:
                with torch.autograd.profiler.record_function("model forward with for"):
                    cur = _logits(self.model(*inputs, **kwargs))
            else:
                with torch.autograd.profiler.record_function("model forward with for"):
                    cur = _logits(self.model(*inputs, **kwargs))
                    cur = cur.split(virtual_batch_size)
                    torch.testing.assert_close(len(cur), len(self.mean_grads))
                    cur = cur[i]
            ret.append(cur)

        with torch.autograd.profiler.record_function("for edited params recover"):
            self.model.update_params(raw_params)

        return ret

    def _forward_with_hook(self, virtual_batch_size, inputs, kwargs):
        # an optimized implementation with hook
        # but has to utilize a for loop to avoid dynamically changing the batch size when `virtual_batch_size` is `None`
        def get_linear_fwd_hook(lr, pname, mean_grads):
            def linear_fwd_hook(module, input, output):
                assert isinstance(module, torch.nn.Linear), \
                    f"{module} is of type `{type(module)}` instead of `torch.nn.Linear`"

                assert len(input) == 1
                input = input[0]

                if get_linear_fwd_hook.first_entry:
                    if virtual_batch_size is None:
                        input = input.expand(len(mean_grads), *input.shape)
                        output = output.expand(len(mean_grads), *output.shape)
                    else:
                        input = torch.stack(input.split(virtual_batch_size))
                        output = torch.stack(output.split(virtual_batch_size))

                    get_linear_fwd_hook.first_entry = False
                else:
                    input = torch.stack(input.chunk(len(mean_grads)))
                    output = torch.stack(output.chunk(len(mean_grads)))
                torch.testing.assert_close(output.shape[0], len(mean_grads))
                torch.testing.assert_allclose(input.shape[:-1], output.shape[:-1])

                with torch.autograd.profiler.record_function("hook edited transformation"):
                    # x: [virtual_edits_groups, edit_tokens, in_features]
                    x = torch.stack([mean_grads_s[f"{pname}#x"] for mean_grads_s in mean_grads])
                    # delta: [virtual_edits_groups, edit_tokens, out_features]
                    delta = torch.stack([mean_grads_s[f"{pname}#delta"] for mean_grads_s in mean_grads])

                    # input: [virtual_edits_groups, virtual_batch_size, forward_tokens, in_features]
                    # edited: [virtual_edits_groups, virtual_batch_size, forward_tokens, out_features]
                    # lr * input @ x.T @ delta -> edited
                    # g...i,gbi,gbj->g...j
                    edited = lr * torch.einsum("g...i,gbi,gbj->g...j", input, x, delta)
                    torch.testing.assert_allclose(edited.shape, output.shape)

                output_with_edit = output + edited
                output_with_edit = output_with_edit.view(-1, *output_with_edit.shape[2:])
                return output_with_edit

            return linear_fwd_hook

        def hook_model(mean_grads):
            get_linear_fwd_hook.first_entry = True

            handles = []
            for i, m in enumerate([parent_module(self.model, pname) for pname in self.config.model.inner_params]):
                handles.append(m.register_forward_hook(get_linear_fwd_hook(
                    self.edit_lrs[i],
                    self.config.model.inner_params[i],
                    mean_grads
                )))

            LOG.debug(f"(forward) Hooked {len(handles)} modules")

            return handles

        def remove_handles(handles):
            for handle in handles:
                handle.remove()

            LOG.debug(f"(forward) Removed {len(handles)} handles")

        if virtual_batch_size is not None:
            handles = hook_model(self.mean_grads)

            with torch.autograd.profiler.record_function("model forward with hook"):
                ret = _logits(self.model(*inputs, **kwargs))
                ret = list(ret.chunk(len(self.mean_grads)))

            remove_handles(handles)
        else:
            ret = []
            for mean_grads_s in self.mean_grads:
                handles = hook_model([mean_grads_s])

                with torch.autograd.profiler.record_function("model forward with hook"):
                    ret.append(_logits(self.model(*inputs, **kwargs)))

                remove_handles(handles)

        return ret

    def forward(self, *inputs, virtual_batch_size=None, **kwargs):
        if self.mean_grads is None:
            return _logits(self.model(*inputs, **kwargs))

        ret = self._forward_with_hook(virtual_batch_size, inputs, kwargs)
        return ret

    def edit(self, batch, condition=None, language_idx=None, detach_history=False, update_params=True, virtual_batch_size=None):
        for n, p in self.model.named_parameters():
            p.requires_grad_(n in self.config.model.inner_params)

        if not hasattr(self.model, "handles"):
            hook_model(self.model, self.config.model.inner_params)
            LOG.debug(f"Hooked {len(self.model.handles) // 2} modules")

        with torch.autograd.profiler.record_function("raw forward"):
            outputs = _logits(self.model(**batch))
            loss = self.edit_loss_fn(outputs, batch["labels"])["nll"]
            if virtual_batch_size is not None:
                loss = loss * (outputs.shape[0] / virtual_batch_size)

        names = set([n for n, p in self.model.named_parameters()])
        pset = set(self.config.model.inner_params)
        for p in pset:
            assert p in names, f"inner param {p} not in model"

        with torch.autograd.profiler.record_function("raw backward"):
            loss.backward()

        for handle in self.model.handles:
            handle.remove()
        LOG.debug(f"Removed {len(self.model.handles)} handles")
        del self.model.handles

        with torch.autograd.profiler.record_function("grad trans"):
            if self.config.mend.shared:
                param_idx = lambda n, p: self.shape_dict[self.get_shape(p)].index(n) if self.config.mend.shared else None  # noqa: E731
                transformed_factors = {
                    n: self.mend[str(tuple(self.get_shape(p)))](p.__x__, p.__delta__, param_idx(n, p), language_idx, virtual_batch_size)
                    for n, p in _inner_params(self.model.named_parameters(), self.config.model.inner_params)
                }
            else:
                transformed_factors = {
                    n: self.mend[n.replace(".", "#")](p.__x__, p.__delta__, language_idx, virtual_batch_size)
                    for n, p in _inner_params(self.model.named_parameters(), self.config.model.inner_params)
                }

        # Should be bi,bj->ji for nn.Linear, but GPT2 uses Conv1d instead...
        with torch.autograd.profiler.record_function("grad agg"):
            if isinstance(self.model, transformers.GPT2LMHeadModel):
                targ = "ij"
            else:
                targ = "ji"

            if virtual_batch_size is None:
                mean_grads = {
                    n: torch.einsum(f"bi,bj->{targ}", x, delta)
                    for n, (x, delta) in transformed_factors.items()
                }
            else:
                batch_size = outputs.shape[0]
                num_virtual_batch = (batch_size + virtual_batch_size - 1) // virtual_batch_size
                mean_grads = [{} for _ in range(num_virtual_batch)]
                for n, (x, delta) in transformed_factors.items():
                    for mean_grads_s, x_s, delta_s in zip(mean_grads, x, delta):
                        # for hook optimization
                        mean_grads_s[f"{n}#x"] = x_s
                        mean_grads_s[f"{n}#delta"] = delta_s
                self.mean_grads = mean_grads if update_params else None

        info_dict = {}
        if self.mean_grads is None and virtual_batch_size is None:
            idx = 0
            for n, p in _inner_params(self.model.named_parameters(), self.config.model.inner_params):
                info_dict[f"grad/true_mag{idx}"] = p.grad.norm(2).item()
                info_dict[f"grad/pseudo_mag{idx}"] = mean_grads[n].norm(2).item()
                info_dict[f"grad/true_std{idx}"] = p.grad.std().item()
                info_dict[f"grad/pseudo_std{idx}"] = mean_grads[n].std().item()
                info_dict[f"grad/diff{idx}"] = (p.grad - mean_grads[n]).norm(2).item()
                info_dict[f"grad/cos{idx}"] = F.cosine_similarity(p.grad.reshape(-1), mean_grads[n].reshape(-1), dim=0).item()
                idx += 1

        self.model.zero_grad()

        for n, p in self.model.named_parameters():
            p.requires_grad_(False)

        edited_model = self.model
        if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
            edited_model = make_functional(edited_model, in_place=True)
        else:
            assert False
        assert edited_model is not self.model

        with torch.autograd.profiler.record_function("update raw"):
            if update_params and virtual_batch_size is None:
                assert len(self.edit_lrs) == len(list(mean_grads.items()))
                updates = {n: lr * g for lr, (n, g) in zip(self.edit_lrs, mean_grads.items())}

                new_params = []
                for n, p in edited_model.named_parameters():
                    if n in pset:
                        new_params.append(p + updates[n])
                    else:
                        new_params.append(p)

                edited_model.update_params(new_params)

        if detach_history:
            new_model = self.model_constructor()
            new_model.load_state_dict(edited_model.state_dict())
            edited_model = new_model

        mend_with_edited = MEND(edited_model, self.config, self.model_constructor, self.mend, edit_lrs=self.edit_lrs, mean_grads=self.mean_grads)
        self.mean_grads = None
        return mend_with_edited, mean_grads, info_dict
