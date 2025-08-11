"""
Module to find sparse subset of edges.

Code given by *Finding Transformer Circuits with Edge Pruning*  (Adithya Bhaskar et al. 2024).
<https://github.com/princeton-nlp/Edge-Pruning>

MIT License

Copyright (c) 2024 Adithya Bhaskar, Alexander Wettig, Dan Friedman, and Danqi Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import math
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.data.data_collator import DataCollator

from edge_pruning.modeling.modeling_fpt2 import FPT2LMHeadModel
from edge_pruning.modeling.modeling_fqwen_kv_expansion import FQwen2ForCausalLM
from edge_pruning.modeling.params import SparsityConfig


class EPTrainer(Seq2SeqTrainer):
    """
    Extension of a Seq2SeqTrainer to perform edge-pruning for a single-token prediction task.

    Args:
        model (Union[PreTrainedModel, nn.Module]): The main model to be trained.
        base_model (Union[PreTrainedModel, nn.Module]): The base model with frozen parameters.
        args (Seq2SeqTrainingArguments): Training arguments for the Seq2SeqTrainer.
        data_collator (Optional[DataCollator]): The function to combine samples into a batch.
        train_dataset (Optional[Union[Dataset, IterableDataset, datasets.Dataset]]): The dataset for training.
        eval_dataset (Optional[Union[Dataset, dict[str, Dataset]]]): The dataset for evaluation.
        compute_metrics (Optional[Callable[[EvalPrediction], dict]]): Function to compute metrics during evaluation.
        optimizers (Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]]):
            Tuple containing the optimizer and scheduler.
        sparsity_config (SparsityConfig): Configuration for edge and layer sparsity settings.
        **kwargs: Additional keyword arguments for the superclass.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        base_model: Union[PreTrainedModel, nn.Module],
        args: Seq2SeqTrainingArguments,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[
            Union[Dataset, "IterableDataset", "datasets.Dataset"]
        ] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        optimizers: Tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        sparsity_config: SparsityConfig = SparsityConfig(),
        **kwargs,
    ) -> None:
        self.base_model = base_model.requires_grad_(False)
        self._device_count = torch.cuda.device_count()

        # Store sparsity targets & schedules
        self.target_edge_sparsity = sparsity_config.target_edge_sparsity
        self.start_edge_sparsity = sparsity_config.start_edge_sparsity
        self.target_layer_sparsity = sparsity_config.target_layer_sparsity
        self.start_layer_sparsity = sparsity_config.start_layer_sparsity
        self.skip_layer_loss_if_higher_sparsity = (
            sparsity_config.skip_layer_loss_if_higher_sparsity
        )

        # Warm-up durations
        self.num_edge_sparsity_warmup_steps = (
            sparsity_config.num_edge_sparsity_warmup_steps
            if sparsity_config.num_edge_sparsity_warmup_steps is not None
            else sparsity_config.num_sparsity_warmup_steps
        )
        self.num_layer_sparsity_warmup_steps = (
            sparsity_config.num_layer_sparsity_warmup_steps
            if sparsity_config.num_layer_sparsity_warmup_steps is not None
            else self.num_edge_sparsity_warmup_steps
        )
        self.warmup_type = sparsity_config.warmup_type
        super().__init__(
            model=model,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            args=args,
            **kwargs,
        )

    @property
    def device_count(self) -> int:
        """
        Return the number of visible CUDA devices.

        Returns:
            int: ``max(torch.cuda.device_count(), 1)`` to cover CPU-only setups.
        """
        return max(self._device_count, 1)

    @staticmethod
    def _scheduled_sparsity(
        step: int,
        start_value: float,
        target_value: float,
        warmup_steps: int,
        schedule_type: Literal["linear", "logarithmic"] = "linear",
    ) -> float:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be ≥ 0")

        # Optional clamp to keep log1p domain safe
        if schedule_type == "logarithmic":
            eps = 1e-7
            start_value = min(max(start_value, 0.0), 1.0 - eps)
            target_value = min(max(target_value, 0.0), 1.0 - eps)

        if warmup_steps == 0:
            return target_value

        # Smooth the end-point if you dislike the tiny jump
        step = min(step, warmup_steps)
        progress = step / warmup_steps

        if schedule_type == "linear":
            return start_value + (target_value - start_value) * progress
        elif schedule_type == "logarithmic":
            log_start = math.log1p(-start_value)
            log_target = math.log1p(-target_value)
            return 1.0 - math.exp(log_start + (log_target - log_start) * progress)

        raise ValueError(f"Unknown schedule_type: {schedule_type!r}")

    def _current_edge_target(self, step: int) -> float:
        """
        Edge sparsity target for the given *step*.

        Args:
            step (int): Global optimisation step.

        Returns:
            float: Edge sparsity.
        """
        return self._scheduled_sparsity(
            step=step,
            start_value=self.start_edge_sparsity,
            target_value=self.target_edge_sparsity,
            warmup_steps=self.num_edge_sparsity_warmup_steps,
            schedule_type=self.warmup_type,
        )

    def _current_layer_target(self, step: int) -> float:
        """
        Layer sparsity target for the given *step*.

        Args:
            step (int): Global optimisation step.

        Returns:
            float: Layer sparsity.
        """
        return self._scheduled_sparsity(
            step=step,
            start_value=self.start_layer_sparsity,
            target_value=self.target_layer_sparsity,
            warmup_steps=self.num_layer_sparsity_warmup_steps,
            schedule_type=self.warmup_type,
        )

    def _kl_divergence(
        self,
        sparse_model_logits: torch.Tensor,
        base_model_logits: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """
        Compute KL divergence between *base model* and *sparse model* logits.

        Args:
            sparse_model_logits (torch.Tensor): Logits from the sparse model
                (unnormalised).
            base_model_logits (torch.Tensor): Logits from the frozen teacher model
                (unnormalised).
            start (int): Slice start index (inclusive).
            end (int): Slice end index (exclusive).

        Returns:
            torch.Tensor: Scalar KL divergence.
        """
        s_logp = nn.functional.log_softmax(sparse_model_logits[start:end], dim=-1)
        t_logp = nn.functional.log_softmax(base_model_logits[start:end], dim=-1)
        return nn.functional.kl_div(
            s_logp, t_logp, reduction="batchmean", log_target=True
        )

    def compute_loss(
        self,
        model: Union[FPT2LMHeadModel, FQwen2ForCausalLM],
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass, KL distillation, and sparsity regularisation.

        Args:
            model (nn.Module): Sparse model to optimise.
            inputs (Dict[str, torch.Tensor]): Batch dictionary coming from the
                ``Seq2SeqTrainer`` dataloader.  Keys ``start_idxes``,
                ``end_idxes``, ``corr_input_ids`` and ``input_ids`` are
                consumed.
            return_outputs (bool): Whether to include the full *outputs* dict in
                the return value.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]: Either the
            scalar loss or ``(loss, outputs)`` if *return_outputs* is *True*.
        """
        labels = inputs.pop("labels")
        start_idxes = inputs.pop("start_idxes")
        end_idxes = inputs.pop("end_idxes")
        corr_input_ids = inputs.pop("corr_input_ids")
        corr_attention_mask = inputs.pop("corr_attention_mask", None)
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask", None)
        _ = inputs.pop("labels_at_metric", None)

        batch_size = input_ids.size(0)

        # ---------------------------- base model pass -------------------------- #
        with torch.no_grad():
            base_model_out = self.base_model(
                input_ids=input_ids, attention_mask=attention_mask, **inputs
            )
            base_model_logits = base_model_out.logits  # type: ignore[attr-defined]

            # run the corrupted inputs through it, and retain the activations
            corr_states = self.base_model(
                input_ids=corr_input_ids,
                attention_mask=corr_attention_mask,
                **inputs,
                output_writer_states=True,
            ).writer_states

            # reshape for multi-GPU writer states
            if corr_states.size(1) != batch_size:
                corr_states = corr_states.view(
                    corr_states.size(0),
                    batch_size // self.device_count,
                    *corr_states.shape[2:],
                )

        # ---------------------------- sparse model pass -------------------------- #
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **inputs,
            target_edge_sparsity=self._current_edge_target(self.state.global_step),  # type: ignore[attr-defined]
            target_node_sparsity=self._current_layer_target(self.state.global_step),  # type: ignore[attr-defined]
            corr_x=corr_states,
        )

        # ------------------------ Regularisation --------------------------- #
        edge_loss: torch.Tensor = outputs["edge_loss"]
        node_loss: torch.Tensor = outputs["node_loss"]
        current_node_sparsity: float = float(outputs["model_node_sparsity"])
        target_node_sparsity: float = float(outputs["target_node_sparsity"])

        if (
            self.skip_layer_loss_if_higher_sparsity
            and current_node_sparsity > target_node_sparsity
        ):
            node_loss = torch.zeros_like(edge_loss)

        reg_loss = edge_loss + node_loss
        sparse_model_logits = outputs["logits"]

        # -------------------------- KL divergence --------------------------- #
        kl_accum = torch.tensor(0.0, device=sparse_model_logits.device)
        for i in range(batch_size):
            kl_accum += self._kl_divergence(
                sparse_model_logits=sparse_model_logits[i],
                base_model_logits=base_model_logits[i],
                start=int(start_idxes[i]),
                end=int(end_idxes[i]),
            )
        kl_loss = kl_accum / batch_size  # type: ignore[assignment]

        # ------------------------------ Total ------------------------------- #
        total_loss = kl_loss + reg_loss
        outputs["loss"] = total_loss
        outputs["kl_loss"] = kl_loss

        if self.state.global_step % 20 == 0 and model.training:
            with torch.no_grad():
                preds = sparse_model_logits.argmax(dim=-1)[:, :-1]
                labels = labels[:, 1:]
                mask = labels != -100
                acc = ((preds * mask) == (labels * mask)).all(dim=1).float().mean()

                self.log(
                    {
                        "train/loss": round(total_loss.detach().item(), 3),
                        "train/kl_loss": round(kl_loss.detach().item(), 3),
                        "train/edge_loss": round(edge_loss.detach().item(), 3),
                        "train/layer_loss": round(node_loss.detach().item(), 3),
                        "train/accuracy": round(acc.detach().item(), 3),
                        "train/edge_sparsity": round(
                            model.get_edge_sparsity().detach().item(), 3
                        ),
                        "train/layer_sparsity": round(
                            model.get_node_sparsity().detach().item(), 3
                        ),
                    },
                )

        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only=False,
        ignore_keys=None,
    ):
        labels = inputs["labels"]

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs.copy(), return_outputs=True)

            logits = outputs["logits"]  # [B, L, V]
            preds = logits.argmax(dim=-1)[:, :-1].cpu()  # [B, L] on CPU
            del logits  # free GPU memory
            torch.cuda.empty_cache()

            payload = (
                preds,
                outputs["target_edge_sparsity"].cpu(),
                outputs["target_node_sparsity"].cpu(),
                outputs["model_edge_sparsity"].cpu(),
                outputs["model_node_sparsity"].cpu(),
                outputs["edge_loss"].cpu(),
                outputs["node_loss"].cpu(),
                outputs["kl_loss"].cpu(),
            )

        if prediction_loss_only:
            return loss.detach().cpu(), None, None

        return loss.detach().cpu(), payload, labels.cpu()
