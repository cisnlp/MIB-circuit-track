"""
Classes and functions to handle data for edge pruning.

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

from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from MIB_circuit_track.dataset import HFEAPDataset


class EPDataCollator:
    """
    Data collator for edge-pruning experiments.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_id = tokenizer.pad_token_id

    def _encode(self, text: str):
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=False,
        )

    def __call__(self, examples: List[Dict[str, str]]):
        input_ids_all, corr_input_ids_all, labels_all = [], [], []
        attention_masks_all, corr_attention_masks_all = [], []
        start_idxes, end_idxes = [], []
        labels_logit_diff_all = []

        key = "text" if "text" in examples[0] else "ioi_sentences"

        for bidx, ex in enumerate(examples):
            text: str = ex[key]
            corr_text: str = ex["corr_" + key]

            split_char_idx = text.rfind(" ")
            if split_char_idx == -1:
                raise ValueError(
                    "Example text has no space; cannot determine prefix/target split."
                )

            prefix = text[:split_char_idx]

            # Encode full text and prefix text separately
            full_enc = self._encode(text)
            pref_enc = self._encode(prefix)
            corr_full_enc = self._encode(corr_text)

            input_ids = full_enc.input_ids.squeeze(0)
            attn_mask = full_enc.attention_mask.squeeze(0)
            corr_attn_mask = corr_full_enc.attention_mask.squeeze(0)

            corr_input_ids = corr_full_enc.input_ids.squeeze(0)

            # Number of tokens in *prefix* (no loss region)
            len_non_loss = int(pref_enc.attention_mask.squeeze(0).sum().item())

            # Build labels: mask prefix and padding
            labels = input_ids.clone()
            labels[:len_non_loss] = -100
            labels[~attn_mask.bool()] = -100

            # Determine [start, end) indices for KL/accuracy using attention mask.
            seq_len_no_pad = int(attn_mask.sum().item())
            start_idx = len_non_loss
            end_idx = seq_len_no_pad

            # For next-token logits alignment, use (idx - 1)
            start_idx_next_token = max(start_idx - 1, 0)
            end_idx_next_token = max(end_idx - 1, 0)

            # Accumulate
            input_ids_all.append(input_ids)
            corr_input_ids_all.append(corr_input_ids)
            labels_all.append(labels)
            start_idxes.append(start_idx_next_token)
            end_idxes.append(end_idx_next_token)
            attention_masks_all.append(attn_mask)
            corr_attention_masks_all.append(corr_attn_mask)
            labels_logit_diff_all.append(
                torch.tensor([ex["label_raw"], ex["neg_raw"]], dtype=torch.long)
            )

        batch = {
            "input_ids": torch.stack(input_ids_all),
            "corr_input_ids": torch.stack(corr_input_ids_all),
            "labels": torch.stack(labels_all),
            "start_idxes": torch.LongTensor(start_idxes),
            "end_idxes": torch.LongTensor(end_idxes),
            "labels_at_metric": torch.stack(labels_logit_diff_all),
        }
        return batch


class EdgePruningAdapter(Dataset):
    """
    Wraps an HFEAPDataset so that each item is a dictionary understood by EPDataCollator.
    """

    def __init__(self, mib_dataset: HFEAPDataset, tokenizer: PreTrainedTokenizer):
        self.base = mib_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.base)):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self.base)}"
            )

        try:
            clean, corrupt, label_pair = self.base[idx]
        except ValueError as e:
            raise ValueError(
                "Each item of `self.base` must unpack into (clean, corrupt, (clean_label, corr_label))"
            ) from e

        clean_label, corr_label = label_pair

        clean_label_str = self.tokenizer.decode(clean_label, skip_special_tokens=False)
        corr_label_str = self.tokenizer.decode(corr_label, skip_special_tokens=False)

        return {
            "text": f"{clean.rstrip()} {clean_label_str.lstrip()}",
            "corr_text": f"{corrupt.rstrip()} {corr_label_str.lstrip()}",
            "label_raw": clean_label,
            "neg_raw": corr_label,
        }
