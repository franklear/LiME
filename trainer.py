import gzip
import json
import logging
import os
import shutil
import tempfile
import time

import torch
import wandb
from omegaconf import OmegaConf
from torch.utils.data import Dataset

import utils
from losses import kl_loc_loss
from utils import safe_backward, RunningStatAverager, EarlyStopper, formatted_timestamp, time_delta_seconds, \
    chech_graphic_mem

LOG = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, model, config, train_set: Dataset, val_set: Dataset):
        self.model = model
        self.config = config

        if config.train_base:
            self.original_model = self.model.model_constructor()
            self.original_model.load_state_dict(self.model.model.state_dict())
            self.original_model.to(self.config.device)
        else:
            self.original_model = self.model.model

        self.model.to(self.config.device)

        self.train_set = train_set
        self.val_set = val_set

        if self.config.eval_only:
            # Eval once and quit
            self.config.max_iters = 0

        if not self.config.eval_only:
            self.OptimizerClass = getattr(torch.optim, config.opt)
            LOG.info(f"Building optimizer {self.OptimizerClass} with lr {config.lr}")
            if len(list(self.model.outer_parameters())) == 0:
                self.opt = None
                LOG.warning("no outer parameter to optimize")
            else:
                if self.config.lime.enabled and self.config.lime.mask_lr is not None:
                    mask_params = self.model.mask_parameters()
                    mask_params_set = set(mask_params)
                    other_params = [p for p in self.model.outer_parameters() if p not in mask_params_set]
                    param_groups = [
                        {"params": mask_params, "lr": config.lime.mask_lr},
                        {"params": other_params},
                    ]
                    self.opt: torch.optim.Optimizer = self.OptimizerClass(param_groups, lr=config.lr)
                else:
                    self.opt: torch.optim.Optimizer = self.OptimizerClass(self.model.outer_parameters(), lr=config.lr)

        if config.archive is not None:
            archive, config.archive = utils.load_archive(str(config.archive))
            self.model.load_state_dict(archive["model"])
            del archive["model"]
            if not self.config.eval_only and self.opt is not None:
                self.opt.load_state_dict(archive["opt"])
            del archive["opt"]

            self.archive = archive  # Save for later to load e.g. lr_opt params if they exist
        else:
            self.archive = None

        # outfiles
        with open(os.getcwd() + "/config.json", "w") as f:
            json.dump(OmegaConf.to_container(config), f)

        model_dir = os.path.join(os.getcwd(), 'models')
        if not (self.config.debug and not self.config.save):
            os.makedirs(model_dir)
        run_date = os.getcwd().split('/')[-1]
        self.run_date = run_date
        safe_model_name = self.config.model.name.split("/")[-1]  # Make sure no slashes
        self.save_path = f"{model_dir}/{safe_model_name}.{run_date}"

        if not (self.config.debug or self.config.eval_only):
            wandb_dir = tempfile.mkdtemp()
            if self.config.task in ["lama", "xnli"]:
                wandb_name = f"{self.config.dataset} - {'lime:' if self.config.lime.enabled else ''}{self.config.alg}" \
                             f" - {safe_model_name} - {run_date}" \
                             f" - edit {self.config.edit_languages} - all {self.config.languages}"
            else:
                wandb_name = f"{self.config.dataset} - {self.config.alg} - {safe_model_name} - {run_date}"
            if self.config.ref is not None:
                wandb_name += f" - {self.config.ref}"
            LOG.info(f"Writing wandb run \"{wandb_name}\" to {wandb_dir}")
            wandb.init(
                project=config.wandb.project,  # project name
                entity=config.wandb.entity,  # user name
                config=utils.flatten_dict(self.config),
                name=wandb_name,
                dir=wandb_dir,
                tags=[self.config.ref] if self.config.ref is not None else None,
                settings=wandb.Settings(start_method="thread"),
            )

        if self.config.debug:
            if self.config.results_dir is not None:
                self.debug_dir = self.config.results_dir
            else:
                self.debug_dir = os.getcwd()
            self.debug_dir = os.path.join(self.debug_dir, "debug")
            os.makedirs(self.debug_dir, exist_ok=True)

            self.debug_dict = None

        if self.config.profile:
            prof_acts = [torch.profiler.ProfilerActivity.CPU]
            # if self.config.device in ["cuda"]:
            #     prof_acts.append(torch.profiler.ProfilerActivity.CUDA)
            self.prof = torch.profiler.profile(
                activities=prof_acts,
                schedule=None,
                on_trace_ready=None,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
            )

        self.start_time = formatted_timestamp()

    def save_state(self, stats):
        if (self.config.debug and not self.config.save) or self.config.eval_only:
            return

        obj = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict() if self.opt is not None else None,
            "lr_opt": self.lr_opt.state_dict() if self.lr_opt is not None else None,
            "val_stats": stats,
            "start_time": self.start_time,
            "elapsed_time": time_delta_seconds(self.start_time),
            "step": self.global_iter
        }
        LOG.info(f"Saving model to {self.save_path}")

        if os.path.exists(self.save_path):
            bk_path = f"{self.save_path}.bk"
            LOG.info(f"Moving old archive to {bk_path}")
            os.rename(self.save_path, bk_path)

        torch.save(obj, self.save_path)
        LOG.info("Write complete.")

    def echo(self, train_step, info_dict, pretty=False):
        if not self.config.silent:
            sep = "\n" if pretty else "; "

            def key_format(k):
                return k.ljust(20) if pretty else k
            LOG.info(f"Step {train_step}:")
            LOG.info(sep.join([f"{key_format(k)}: {v: 0.5f}" for k, v in info_dict.items()]))

    def wandb_log(self, step, info_dict):
        if not (self.config.debug or self.config.eval_only):
            wandb.log(info_dict, step=step)

    def run(self):
        if self.config.debug:
            if self.debug_dict is None:
                self.debug_dict = {}
            if True not in self.debug_dict:
                self.debug_dict[True] = {}
            self.debug_dict[True][False] = {}

        averager = RunningStatAverager("train")
        stopper = EarlyStopper(self.config.early_stop_patience, self.config.early_stop_key)
        self.global_iter = 0
        for global_iter in range(0, self.config.max_iters):
            self.global_iter = global_iter

            if not self.config.eval_only:
                train_info = self.train_step()
                averager.add(train_info)

                if global_iter % self.config.log_interval == 0:
                    avg_info = averager.average()
                    averager.reset()
                    self.echo(global_iter, avg_info)
                    self.wandb_log(global_iter, avg_info)

            if global_iter % self.config.val_interval == 0:
                val_info, _ = self.validate(steps=self.config.val_steps, predict=False)
                self.echo(global_iter, val_info)
                self.wandb_log(global_iter, val_info)

                if stopper.update(self.global_iter, val_info):
                    self.save_state(val_info)  # New best

                if stopper.should_stop():
                    LOG.info(f"No decrease in {self.config.early_stop_key} for {self.config.early_stop_patience} steps")
                    break

        if not self.config.eval_only:
            LOG.info(f"Training complete after {self.global_iter+1} steps.")

        if not self.config.eval.final_eval:
            return

        if not self.config.eval_only:
            if (not self.config.debug) or self.config.save:
                archive = torch.load(self.save_path, map_location="cpu")
                LOG.info(f"Loading best model from step {archive['step']}, elapsed time {archive['elapsed_time']}")
                self.model.to("cpu")
                self.model.load_state_dict(archive["model"])
                self.model.to(self.config.device)

        val_steps = self.config.val_steps if self.config.debug else None
        val_info, pred = self.validate(log=True, steps=val_steps, predict=True)
        self.echo(self.global_iter, val_info, pretty=True)
        self.wandb_log(self.global_iter + self.config.val_interval, val_info)

        if self.config.results_dir is not None:
            results_dir = self.config.results_dir
            results_path = f"{self.config.results_dir}/results_{self.run_date}.json"
            latest_path = f"{self.config.results_dir}/results_latest.json"
        else:
            results_dir = os.getcwd()
            results_path = f"{os.getcwd()}/results.json"
            latest_path = f"{os.getcwd()}/results_latest.json"
        os.makedirs(results_dir, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump({"results": val_info, "config": OmegaConf.to_container(self.config)}, f)
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        shutil.copy(results_path, latest_path)
        LOG.info("Copied to:")
        LOG.info(latest_path)

        pred_path = os.path.join(os.path.split(results_path)[0], "pred.json")
        with open(pred_path, "w", encoding="utf8") as fp:
            json.dump(pred, fp, indent=4, ensure_ascii=False)
            LOG.info(f"Wrote predictions to {pred_path}")
        pred_gz_path = f"{pred_path}.gz"
        with open(pred_path, "rb") as f_in:
            with gzip.open(pred_gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
                LOG.info(f"Compressed predictions to {pred_gz_path}")
        os.remove(pred_path)


class EditTrainer(BaseTrainer):
    def __init__(self, model, config, train_set: Dataset, val_set: Dataset):
        super().__init__(model, config, train_set, val_set)

        if not config.eval_only or config.alg == "ft" and config.ft.locality.enabled:
            self.edit_gen = self.train_set.edit_generator(batch_size=config.batch_size)
        else:
            self.edit_gen = None
        if hasattr(model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def edit_step(self, batch, training: bool, predict: bool):
        self.model.train(training)
        self.original_model.train(training)
        # if self.config.debug:
        #     self.model.train(False)
        #     self.original_model.train(False)

        if self.config.lime.enabled:
            self.model.pre_step_constrain_parameters()

        with torch.no_grad():
            with torch.autograd.profiler.record_function("raw base_logits fwd"):
                if self.model.raw_model is None:
                    base_logits = self.model(**batch["loc"])
                else:
                    base_logits = self.model.raw_model_forward(**batch["loc"])

            if predict:
                with torch.autograd.profiler.record_function("raw inner_logits fwd for predict"):
                    inner_logits = self.model(**batch["edit_inner"])
                with torch.autograd.profiler.record_function("raw edit_logits fwd for predict"):
                    edit_logits = self.model(**batch["edit_outer"])

        # Do the edit
        with torch.autograd.profiler.record_function("edit"):
            start = time.time()
            edited_model, inner_grad, model_info = self.model.edit(batch["edit_inner"], batch["cond"], batch.get("language_idx_inner", None), virtual_batch_size=self.config.virtual_n_edits)
            edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            # Editing loss
            with torch.autograd.profiler.record_function("edited post_edit_logits fwd for editing loss"):
                if self.config.virtual_n_edits is None:
                    post_edit_logits = edited_model(**batch["edit_outer"])
                else:
                    post_edit_logits = edited_model(**batch["edit_outer"], virtual_batch_size=self.config.virtual_n_edits)
                l_edit = self.model.edit_loss_fn(post_edit_logits, batch["edit_outer"]["labels"])["nll"]

            # Locality loss
            with torch.autograd.profiler.record_function("edited post_base_logits fwd for locality loss"):
                post_base_logits = edited_model(**batch["loc"])
                kl_mask = batch["loc"].get("decoder_attention_mask", batch["loc"]["attention_mask"])
                l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask, all_to_all=True)

            # Mask L0
            l_l0 = torch.tensor(0.)
            if self.config.alg in ["mex"] or self.config.lime.enabled:
                assert callable(getattr(self.model, 'l0_reg', None))
            if self.config.lime.enabled and self.config.lime.mask_l0:
                l_l0 = self.model.l0_reg()

        if predict:
            with torch.no_grad():
                with torch.autograd.profiler.record_function("edited post_inner_logits fwd for predict"):
                    if self.config.virtual_n_edits is None:
                        post_inner_logits = edited_model(**batch["edit_inner"])
                    else:
                        post_inner_logits = edited_model(**batch["edit_inner"], virtual_batch_size=self.config.virtual_n_edits)

        l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc
        l_total_edit_with_reg = l_total_edit
        if self.config.lime.enabled:
            l_total_edit_with_reg = l_total_edit_with_reg + self.config.lime.cl0 * l_l0

        if training and self.opt is not None:
            with torch.autograd.profiler.record_function("total edit loss backward"):
                safe_backward(l_total_edit_with_reg, self.model.outer_parameters(), self.config.accumulate_bs)

        # magic to avoid memory leaking of higher triggered by the backward hook
        # https://github.com/facebookresearch/higher/issues/75
        if self.config.alg in ["enn"] and self.config.lime.enabled:
            for pname in self.config.model.inner_params:
                m = utils.parent_module(edited_model.model, pname)
                delattr(m, pname.rsplit(".", maxsplit=1)[-1])

        # Collect some useful metrics
        with torch.no_grad():
            post_edit_dict = self.model.edit_loss_fn(post_edit_logits, batch["edit_outer"]["labels"])
            if "labels" in batch["loc"]:
                post_loc_dict = self.model.loc_loss_fn(post_base_logits, batch["loc"]["labels"], all_to_all=True)
                pre_loc_dict = self.model.loc_loss_fn(base_logits, batch["loc"]["labels"])

            loc_cons_label = torch.argmax(base_logits, dim=-1)
            if "labels" in batch["loc"]:
                assert loc_cons_label.shape == batch["loc"]["labels"].shape
                loc_cons_label[batch["loc"]["labels"] < 0] = batch["loc"]["labels"][batch["loc"]["labels"] < 0]
            loc_cons_dict = self.model.loc_loss_fn(post_base_logits, loc_cons_label, all_to_all=True)

        info_dict = {}
        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict['loss/l0'] = l_l0.item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
        info_dict['edit/prob'] = post_edit_dict["prob"].item()

        info_dict["acc/id"] = loc_cons_dict["acc"].item()
        info_dict["nll/id"] = loc_cons_dict["nll"].item()
        info_dict["n_tokens/id"] = loc_cons_dict["n_tokens"].item()
        if "labels" in batch["loc"]:
            info_dict["acc/pre"] = pre_loc_dict["acc"].item()
            info_dict["acc/post"] = post_loc_dict["acc"].item()
            info_dict["nll/pre"] = pre_loc_dict["nll"].item()
            info_dict["nll/post"] = post_loc_dict["nll"].item()
            info_dict["n_tokens/pre"] = pre_loc_dict["n_tokens"].item()
            info_dict["n_tokens/post"] = post_loc_dict["n_tokens"].item()

        info_dict["acc/edit_id_avg"] = (info_dict['edit/acc'] + info_dict["acc/id"]) / 2
        info_dict["acc/neg_edit_id_avg"] = -info_dict["acc/edit_id_avg"]
        if abs(info_dict['edit/acc'] + info_dict["acc/id"]) < 1e-9:
            info_dict["acc/edit_id_havg"] = 0.0
        else:
            info_dict["acc/edit_id_havg"] = 2 * info_dict['edit/acc'] * info_dict["acc/id"] / (info_dict['edit/acc'] + info_dict["acc/id"])
        info_dict["acc/neg_edit_id_havg"] = -info_dict["acc/edit_id_havg"]

        info_dict["time/edit"] = edit_time

        # Base loss
        if self.config.train_base:
            with torch.no_grad():
                original_logits = base_logits
                original_loc_dict = self.model.loc_loss_fn(original_logits, batch["loc"]["labels"])

            base_logits_for_loss = self.model(**batch["loc"])
            l_base = self.config.cbase * kl_loc_loss(original_logits, base_logits_for_loss, mask=kl_mask.detach())

            if training and self.opt is not None:
                safe_backward(l_base, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

            info_dict['loss/base'] = l_base.item()
            info_dict['nll/original'] = original_loc_dict["nll"].item()
            info_dict['acc/original'] = original_loc_dict["acc"].item()
            info_dict["n_tokens/original"] = original_loc_dict["n_tokens"]
        else:
            l_base = torch.tensor(0.)

        l_total = l_total_edit_with_reg + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["loss/total_edit_with_reg"] = l_total_edit_with_reg.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        predict_dict = None
        if predict:
            predict_dict = {
                "loc": {
                    "logits": {
                        "pre": base_logits.detach() if self.config.virtual_n_edits is None
                               else base_logits.repeat([sum(p.shape[0] for p in post_base_logits) // base_logits.shape[0]] + [1] * (base_logits.dim() - 1)).detach(),
                        "post": post_base_logits.detach() if self.config.virtual_n_edits is None
                                else torch.cat(post_base_logits, dim=0).detach()
                    }
                },
                "edit_inner": {
                    "logits": {
                        "pre": inner_logits.detach(),
                        "post": post_inner_logits.detach() if self.config.virtual_n_edits is None
                                else torch.cat(post_inner_logits, dim=0).detach()
                    }
                },
                "edit_outer": {
                    "logits": {
                        "pre": edit_logits.detach(),
                        "post": post_edit_logits.detach() if self.config.virtual_n_edits is None
                                else torch.cat(post_edit_logits, dim=0).detach()
                    }
                }
            }
            for tp in predict_dict:
                tp_dict = predict_dict[tp]
                bt_dict = batch[tp]

                tp_dict["labels"] = {k: torch.argmax(v, dim=-1).cpu() for k, v in tp_dict["logits"].items()}
                if "labels" in bt_dict:
                    tp_dict["labels"]["gold"] = bt_dict["labels"].detach().cpu()

                tp_dict["input_ids"] = bt_dict["input_ids"].detach().cpu()

                del tp_dict["logits"]

        if self.config.profile:
            self.prof.step()

        if self.config.debug:
            chech_graphic_mem("end of edit_step")

        return l_total, l_edit, l_loc, l_base, info_dict, predict_dict

    def train_step(self):
        if self.config.profile:
            self.prof.step()

        if self.config.debug:
            chech_graphic_mem("(train_step) before edit_step")

        with torch.autograd.profiler.record_function("train_step"):
            with torch.autograd.profiler.record_function("edit step"):
                l_total, l_edit, l_loc, l_base, info_dict, _ = self.edit_step(next(self.edit_gen), training=True, predict=False)

            if self.config.debug:
                chech_graphic_mem("(train_step) after edit_step")

            if self.opt is not None and self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
                with torch.autograd.profiler.record_function("clip grad"):
                    grad = torch.nn.utils.clip_grad_norm_(self.model.outer_parameters(), self.config.grad_clip, error_if_nonfinite=True)
                info_dict['grad'] = grad.item()

                with torch.autograd.profiler.record_function("opt step"):
                    self.opt.step()
                    self.opt.zero_grad()

                if self.lr_opt is not None:
                    with torch.autograd.profiler.record_function("lr opt step"):
                        self.lr_opt.step()
                        self.lr_opt.zero_grad()

                    for lr_idx, lr in enumerate(self.model.edit_lrs):
                        info_dict[f'lr/lr{lr_idx}'] = lr.item()

                if self.config.debug:
                    utils.report_tensors_in_mem()

        if self.config.profile:
            self.prof.step()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = time.time() - start_time
        prog = f"{step+1}/{steps}".ljust(20)
        acc = f"{stats['edit/acc_val']:<12.5f}"
        log_str = f"Step {prog}" \
                  f"\n        edit_acc: {acc} it_time: {elapsed / (step + 1):.4f} elapsed: {elapsed:.4f}"

        if self.config.task in ["fc", "qa", "lama"] and 'acc/pre_val' in stats:
            draw_pre = f"{stats['acc/pre_val']:<12.5f}"
            draw_post = f"{stats['acc/post_val']:<12.5f}"
            draw_diff = f"{stats['acc/pre_val']-stats['acc/post_val']:<12.5f}"
            dn = "acc"  # drawdown name
            log_str += f"\n        {dn}_pre: {draw_pre} {dn}_post: {draw_post} {dn}_delta: {draw_diff}"
        if self.config.task in ["gen", "lama"] and 'perplexity/pre_val' in stats:
            draw_pre = f"{stats['perplexity/pre_val']:<12.5f}"
            draw_post = f"{stats['perplexity/post_val']:<12.5f}"
            draw_diff = f"{stats['perplexity/post_val']-stats['perplexity/pre_val']:<12.5f}"
            dn = "ppl"  # drawdown name
            log_str += f"\n        {dn}_pre: {draw_pre} {dn}_post: {draw_post} {dn}_delta: {draw_diff}"

        LOG.info(log_str)

    def validate(self, steps=None, log: bool = False, predict: bool = False):
        all_steps = (len(self.val_set) + self.config.data.n_edits - 1) // self.config.data.n_edits
        if steps is None or steps > all_steps:
            steps = all_steps

        if self.config.debug:
            if self.debug_dict is None:
                self.debug_dict = {}
            if False not in self.debug_dict:
                self.debug_dict[False] = {}
            self.debug_dict[False][predict] = {}

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")
        val_edit_gen = self.val_set.edit_generator(batch_size=self.config.val_batch_size, n=steps * self.config.data.n_edits)

        if self.config.profile:
            self.prof.step()

        start_time = time.time()
        pred = [] if predict else None
        for val_step in range(steps):
            with torch.autograd.profiler.record_function("val step"):

                if self.config.profile:
                    self.prof.step()

                if self.config.debug:
                    chech_graphic_mem("(val_step) before edit_step")

                with torch.autograd.profiler.record_function("edit step"):
                    _, _, _, _, info_dict, predict_dict = self.edit_step(next(val_edit_gen), training=False, predict=predict)

                if self.config.debug:
                    chech_graphic_mem("(val_step) after edit_step")

                averager.add(info_dict)
                if predict:
                    pred.append(predict_dict)

                if log and self.config.eval.verbose and (val_step + 1) % self.config.eval.log_interval == 0:
                    self._inline_validation_log(val_step, averager.average(), start_time, steps)

        if log and self.config.eval.verbose:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        if predict:
            if self.config.profile:
                self.prof.step()

            with torch.autograd.profiler.record_function("pred final stat"):
                for i, pred_dict in enumerate(pred):
                    for tp, tp_dict in pred_dict.items():
                        input_ids_mask = tp_dict["input_ids"] != self.val_set.tokenizer.pad_token_id
                        if "gold" in tp_dict["labels"]:
                            labels_mask = tp_dict["labels"]["gold"] >= 0
                        elif tp_dict["labels"]["pre"].shape == tp_dict["input_ids"].shape:
                            labels_mask = input_ids_mask
                        elif tp_dict["labels"]["pre"].shape == tp_dict["input_ids"].shape[:-1]:
                            labels_mask = [Ellipsis] * input_ids_mask.shape[0]
                        else:
                            labels_mask = [Ellipsis] * input_ids_mask.shape[0]
                            LOG.warning(f"Didn't recognize prediction postprocessing type: input_ids shape {tp_dict['input_ids'].shape} labels shape {tp_dict['labels'].shape}")

                        label_string_map = getattr(self.val_set, "label_string_map", None)
                        gen_labels_string = tp_dict["labels"]["pre"].shape == tp_dict["input_ids"].shape or label_string_map is not None

                        tp_dict["input_ids"] = [r[m].tolist() for r, m in zip(tp_dict["input_ids"], input_ids_mask)]
                        tp_dict["sentences"] = [self.val_set.tokenizer.batch_decode(tp_dict["input_ids"])]
                        del tp_dict["input_ids"]

                        tp_dict["labels"] = {k: [r[m].tolist() for r, m in zip(v, labels_mask)] for k, v in tp_dict["labels"].items()}
                        if gen_labels_string:
                            if label_string_map is not None:
                                tp_dict["labels_string"] = {k: label_string_map[v] for k, v in tp_dict["labels"].items()}
                            else:
                                tp_dict["labels_string"] = {k: self.val_set.tokenizer.batch_decode(v) for k, v in tp_dict["labels"].items()}
                            del tp_dict["labels"]

                    pred_dict["step"] = i

        if self.config.profile:
            self.prof.step()

        return stats, pred
