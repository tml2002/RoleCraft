import os
from typing import Optional
from transformers import Trainer

import torch
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import logging

logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.pt"
TRAINING_ARGS_NAME = "training_args.bin"

class LoRATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        return model(**inputs).loss

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model_to_save = unwrap_model(self.model)
        saved_params = {
            k: v.to("cuda") for k, v in model_to_save.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
