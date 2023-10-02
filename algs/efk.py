# Adapted from https://github.com/nicola-decao/KnowledgeEditor/blob/main/src/models/one_shot_learner.py
"""
@inproceedings{decao2020editing,
 title={Editing Factual Knowledge in Language Models},
 author={Nicola De Cao and Wilker Aziz and Ivan Titov},
 booktitle={arXiv pre-print 2104.08164},
 url={https://arxiv.org/abs/2104.08164},
 year={2021},
}
"""
import copy
import itertools
import logging

import higher
import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from higher.patch import monkeypatch as make_functional
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration

from editable_model import EditableModel
from lime_utils import L0Embedding
from models import BertClassifier
from utils import _logits, _inner_params

LOG = logging.getLogger(__name__)


class EFK(EditableModel):
    def __init__(self, model, config, model_constructor, editor=None):
        super().__init__(model, config, model_constructor)

        if editor is None:
            if isinstance(model, BertClassifier):
                embedding = model.model.embeddings.word_embeddings.weight.data
            elif isinstance(model, BartForConditionalGeneration):
                embedding = model.model.shared.weight.data
            elif isinstance(model, T5ForConditionalGeneration):
                embedding = model.shared.weight.data
            elif hasattr(getattr(model, 'transformer', None), 'wte'):
                embedding = model.transformer.wte.weight.data
            else:
                embedding = model.get_input_embeddings().weight.data

            editor = OneShotLearner(model, vocab_dim=model.config.vocab_size,
                                    include_set=config.model.inner_params,
                                    embedding_dim=embedding.shape[-1],
                                    embedding_init=embedding.clone().to(torch.float32),
                                    max_scale=1)
        self.editor = editor

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

        for p in self.model.parameters():
            if p.is_leaf:
                p.requires_grad_(False)

    def outer_parameters(self):
        params = [self.editor.parameters()]
        if self.config.lime.enabled:
            params.append(self.mask_dict.parameters())
        return itertools.chain(*params)

    def mask_parameters(self):
        p = []
        if self.config.lime.enabled:
            p.extend(list(self.mask_dict.parameters()))
        return p

    def l0_reg(self):
        if not self.config.lime.enabled or not self.config.lime.mask_l0:
            return torch.tensor(0.0)

        return torch.stack([gt.l0_reg() for gt in self.mask_dict.values()]).mean()

    def pre_step_constrain_parameters(self):
        if self.config.lime.enabled and self.config.lime.mask_l0:
            for gt in self.mask_dict.values():
                gt.pre_step_constrain_parameters()

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

    def edit(self, batch, condition, language_idx=None, detach_history=False, update_params=True, virtual_batch_size=None):
        assert update_params is True
        assert virtual_batch_size is None

        for n, p in self.model.named_parameters():
            p.requires_grad_(n in self.config.model.inner_params)

        outputs = _logits(self.model(**batch))
        loss = self.edit_loss_fn(outputs, batch["labels"])["nll"]

        names = set([n for n, p in self.model.named_parameters()])
        pset = set(self.config.model.inner_params)
        for p in pset:
            assert p in names, f"inner param {p} not in model"

        grads = torch.autograd.grad(
            loss,
            [p for (n, p) in _inner_params(self.model.named_parameters(), self.config.model.inner_params)]
        )

        for n, p in self.model.named_parameters():
            p.requires_grad_(False)

        if self.config.lime.enabled:
            # only for batchsize of 1
            grads_with_mask = []
            for (n, p), g in zip(_inner_params(self.model.named_parameters(), self.config.model.inner_params), grads):
                m1, m2 = self.mask_dict[n.replace(".", "#")](language_idx).squeeze(0).split(p.shape, -1)
                if self.config.lime.mask_basic_offset is not None:
                    m1, m2 = m1 + self.config.lime.mask_basic_offset, m2 + self.config.lime.mask_basic_offset
                g = m1.unsqueeze(-1) * g * m2
                grads_with_mask.append(g)
            grads = grads_with_mask

        params_dict = self.editor(
            condition["input_ids"] if condition is not None else batch["input_ids"],
            condition["attention_mask"] if condition is not None else batch["attention_mask"],
            {n: g.to(torch.float32) for (n, g) in zip(self.config.model.inner_params, grads)},
        )

        edited_model = self.model
        if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
            edited_model = make_functional(edited_model, in_place=True)

        def new_param(n, p):
            if n not in params_dict:
                return p

            if p.shape[0] == params_dict[n].shape[0]:
                return p + params_dict[n]
            else:
                return p + params_dict[n].T

        edited_model.update_params(
            [new_param(n, p) for (n, p) in edited_model.named_parameters()]
        )

        if detach_history:
            new_model = self.model_constructor()
            new_model.load_state_dict(edited_model.state_dict())
            edited_model = new_model

        return EFK(edited_model, self.config, self.model_constructor, editor=self.editor), {}, {}


class ConditionedParameter(torch.nn.Module):
    def __init__(self, parameter, condition_dim=1024, hidden_dim=128, max_scale=1):
        super().__init__()
        self.parameter_shape = parameter.shape

        if len(self.parameter_shape) == 2:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(
                        hidden_dim, 2 * (parameter.shape[0] + parameter.shape[1]) + 1
                    )
                ),
            )
        elif len(self.parameter_shape) == 1:
            self.conditioners = torch.nn.Sequential(
                torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.utils.weight_norm(
                    torch.nn.Linear(hidden_dim, 2 * parameter.shape[0] + 1)
                ),
            )
        else:
            raise RuntimeError()

        self.max_scale = max_scale

    def forward(self, inputs, grad):
        if inputs.shape[0] > 1:
            raise RuntimeError("Can only condition on batches of size 1")

        if len(self.parameter_shape) == 2:
            (
                conditioner_cola,
                conditioner_rowa,
                conditioner_colb,
                conditioner_rowb,
                conditioner_norm,
            ) = self.conditioners(inputs).split(
                [
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    self.parameter_shape[1],
                    self.parameter_shape[0],
                    1,
                ],
                dim=-1,
            )

            a = conditioner_rowa.softmax(-1).T @ conditioner_cola
            b = conditioner_rowb.softmax(-1).T @ conditioner_colb

        elif len(self.parameter_shape) == 1:
            a, b, conditioner_norm = self.conditioners(inputs).split(
                [self.parameter_shape[0], self.parameter_shape[0], 1], dim=-1
            )
        else:
            raise RuntimeError()

        if a.squeeze().shape[0] != grad.shape[0]:
            return self.max_scale * conditioner_norm.sigmoid().squeeze() * (grad * a.squeeze().T + b.squeeze().T)
        else:
            return self.max_scale * conditioner_norm.sigmoid().squeeze() * (grad * a.squeeze() + b.squeeze())


class LSTMConditioner(torch.nn.Module):
    def __init__(
        self,
        vocab_dim=30522,
        embedding_dim=768,
        hidden_dim=256,
        output_dim=1024,
        embedding_init=None,
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_dim,
            embedding_dim=embedding_dim,
            padding_idx=0,
            _weight=embedding_init,
        )
        self.lstm = PytorchSeq2VecWrapper(
            torch.nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        )
        self.linear = FeedForward(
            input_dim=hidden_dim * 2,
            num_layers=1,
            hidden_dims=[output_dim],
            activations=[torch.nn.Tanh()],
        )

    def forward(self, inputs, masks):
        return self.linear(self.lstm(self.embedding(inputs), masks))


class OneShotLearner(torch.nn.Module):
    def __init__(
        self,
        model,
        vocab_dim,
        embedding_dim=768,
        hidden_dim=512,
        condition_dim=768,
        include_set={},
        max_scale=1e-3,
        embedding_init=None,
    ):
        super().__init__()

        self.param2conditioner_map = {
            n: "{}_conditioner".format(n).replace(".", "_")
            for n, p in model.named_parameters()
            if n in include_set
        }

        self.conditioners = torch.nn.ModuleDict(
            {
                self.param2conditioner_map[n]: ConditionedParameter(
                    p,
                    condition_dim,
                    hidden_dim,
                    max_scale=max_scale,
                )
                for n, p in model.named_parameters()
                if n in include_set
            }
        )

        self.condition = LSTMConditioner(
            vocab_dim,
            embedding_dim,
            hidden_dim,
            condition_dim,
            embedding_init=embedding_init,
        )

    def forward(self, inputs, masks, grads=None):
        condition = self.condition(inputs, masks)
        return {
            p: self.conditioners[self.param2conditioner_map[p]](
                condition,
                grad=grads[p] if grads else None,
            )
            for p, c in self.param2conditioner_map.items()
        }
