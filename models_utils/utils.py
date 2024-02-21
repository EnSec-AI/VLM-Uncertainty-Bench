import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path


def get_model(model_path: str):
    model_name = model_path.split("/")[-1]
    if model_name.startswith('llava'):
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
        model.cuda()
    elif model_name.startswith(('cogvlm', 'cogagent')):
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).cuda().eval()
        image_processor = None
        context_len = None
    elif model_name.startswith(('Qwen', 'Monkey')):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda", trust_remote_code=True
        ).eval()
        image_processor = None
        context_len = None
    elif model_name.startswith('Yi-VL'):
        from .yi_vl.mm_utils import (
            get_model_name_from_path as yi_get_model_name_from_path,
            load_pretrained_model as yi_load_pretrained_model,
        )
        from models_utils.yi_vl.model.constants import key_info
        disable_torch_init()
        model_path = os.path.expanduser(model_path)
        key_info["model_path"] = model_path
        yi_get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = yi_load_pretrained_model(model_path)
    elif model_name.startswith('internlm-xcomposer2'):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True
        ).eval().cuda().half()
        image_processor = None
        context_len = None
    elif model_name.startswith('MoE-LLaVA'):
        from moellava.model.builder import load_pretrained_model as moe_load_pretrained_model
        from moellava.mm_utils import get_model_name_from_path as moe_get_model_name_from_path
        disable_torch_init()
        load_4bit, load_8bit = False, False
        device = torch.device("cuda")
        model_name = moe_get_model_name_from_path(model_path)
        tokenizer, model, processor, context_len = moe_load_pretrained_model(
            model_path, None, model_name, load_8bit, load_4bit, device=device
        )
        image_processor = processor['image']
    elif model_name.startswith('mplug'):
        from mplug_owl2.model.builder import load_pretrained_model as mplug_load_pretrained_model
        from mplug_owl2.mm_utils import get_model_name_from_path as mplug_get_model_name_from_path
        model_name = mplug_get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = mplug_load_pretrained_model(
            model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda"
        )
    elif model_name.startswith('MobileVLM'):
        from models_utils.mobilevlm.model.mobilevlm import load_pretrained_model as mvlm_load_pretrained_model

        disable_torch_init()
        load_4bit, load_8bit = False, False
        tokenizer, model, image_processor, context_len = mvlm_load_pretrained_model(
            model_path, load_8bit, load_4bit
        )
    else:
        raise ValueError('Unrecognized model')
    return tokenizer, model, image_processor, context_len


def get_logits(model, inputs, option_ids):
    with torch.no_grad():
        output = model(
            **inputs
        )
    logits = output.logits.detach()
    logits = logits[:, -1, :]
    logits_full = logits.squeeze(0)
    logits_options = logits_full[option_ids].float().cpu().numpy()
    value, index = torch.max(logits, 1)
    logits_options = np.append(logits_options, index.item())
    return logits_options