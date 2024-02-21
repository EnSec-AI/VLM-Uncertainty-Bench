import torch
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images


from data_utils.common_utils import get_options, is_none, OPTIONS, ALL_OPTIONS
from prompt_utils import tokenizer_image_token_mmmu, internlm_xcomposer2_system_prompt

from .qwen_chat_utils import make_context


def get_inputs(row, tokenizer, image_processor, model_name, model, conv_mode, dataset_name):

    last_part_model_name: str = model_name.split("/")[-1]
    options = get_options(row, OPTIONS)
    options = options + ['I don’t know', 'None of the above']
    question_text = row['question']

    hint = row['hint']
    if not is_none(hint):
        question_text = hint + '\n' + question_text
    for option_char, option in zip(ALL_OPTIONS, options):
        question_text = question_text + '\n' + option_char + '. ' + option
    qs = question_text
    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

    image = Image.open(row['image_path']).convert("RGB")
    image_size = image.size

    if last_part_model_name.startswith('llava'):
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        #got gtom repo
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        image_tensor = process_images([image], image_processor, model.config)[0]
        return {
            'input_ids': input_ids.unsqueeze(0).cuda(),
            'images': image_tensor.unsqueeze(0).half().cuda(),
            'image_sizes': [image_size]
        }
    elif last_part_model_name.startswith(('cogvlm', 'cogagent')):
        #got because it has vicuna lm
        #check changes when vicuna prompt change to chat_template
        """if 'vicuna' in conv_mode:
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
        else:
            prompt = qs"""
        prompt = qs
        input_by_model = model.build_conversation_input_ids(
            tokenizer, query=prompt, history=[], images=[image], #template_version='base'
        )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).cuda(),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).cuda(),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).cuda(),
            'images': [[input_by_model['images'][0].cuda().to(torch.bfloat16)]],
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].cuda().to(torch.bfloat16)]]
        return inputs
    elif last_part_model_name.startswith('Yi-VL'):
        from models_utils.yi_vl.conversation import conv_templates as yi_conv_templates
        from models_utils.yi_vl.model.constants import DEFAULT_IMAGE_TOKEN as YI_DEFAULT_IMAGE_TOKEN
        from models_utils.yi_vl.model.constants import IMAGE_TOKEN_INDEX as YI_IMAGE_TOKEN_INDEX
        from models_utils.yi_vl.mm_utils import (
            expand2square,
            tokenizer_image_token_yi,
        )
        #got from yi-vl repo
        qs = YI_DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = yi_conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token_yi(prompt, tokenizer, YI_IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        if getattr(model.config, "image_aspect_ratio", None) == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in image_processor.image_mean)
            )
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attention_mask = attention_mask.bool()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask.cuda(),
            'images': image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
        }
    elif last_part_model_name.startswith('Qwen') or last_part_model_name.startswith('Monkey'):
        #got from repo qwen-vl
        #we can slightly change it for chat (add system prompt and roles) and check whether something change
        choice_list = []
        for i, option in enumerate(options):
            choice_list.append('{}. {}'.format(ALL_OPTIONS[i], option))
        choice_txt = '\n'.join(choice_list)
        extra_inst = "Answer with the option's letter from the given choices directly." + "\n"

        prompt = '<img>{}</img>Context: {}\nQuestion: {}\nOptions: \n{}\n{}\nAnswer:'
        hint = row['hint'] if row['hint'] else 'N/A'
        question_text = row['question']
        image = row['image_path']

        prompt = prompt.format(image, hint, question_text, choice_txt, extra_inst)

        if not 'Chat' in last_part_model_name:
            input_ids = tokenizer(prompt, return_tensors='pt')
            attention_mask = input_ids.attention_mask
            input_ids = input_ids.input_ids
        else:
            input_ids = tokenizer(prompt, return_tensors='pt')
            attention_mask = input_ids.attention_mask
            input_ids = input_ids.input_ids
        return {
            'input_ids': input_ids.cuda(),
            'attention_mask': attention_mask.cuda(),
        }

    elif last_part_model_name.startswith('internlm-xcomposer2'):
        #got from hf demo
        qs = '<ImageHere>' + qs
        image = model.vis_processor(image)
        image = model.encode_img(image.unsqueeze(0).half())
        history = []
        inputs, im_mask = model.interleav_wrap_chat(
            tokenizer, qs, image, history, internlm_xcomposer2_system_prompt
        )
        inputs = {
            k: v.to(model.device)
            for k, v in inputs.items() if torch.is_tensor(v)
        }
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        #check whether we really need this
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
        ]
        inputs.update([('eos_token_id', eos_token_id)])
        return inputs
    elif last_part_model_name.startswith('MoE-LLaVA'):
        #got from repo
        from moellava.constants import IMAGE_TOKEN_INDEX as MOE_IMAGE_TOKEN_INDEX
        from moellava.constants import DEFAULT_IMAGE_TOKEN as MOE_DEFAULT_IMAGE_TOKEN
        from moellava.conversation import conv_templates as moe_conv_templates
        from moellava.mm_utils import tokenizer_image_token as moe_tokenizer_image_token
        qs = MOE_DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = moe_conv_templates[conv_mode].copy()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')[
            'pixel_values'].to(model.device, dtype=torch.float16)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = moe_tokenizer_image_token(
            prompt, tokenizer, MOE_IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        #stop_str = conv.sep if conv.sep_style != MoeSeparatorStyle.TWO else conv.sep2
        #keywords = [stop_str]
        #stopping_criteria = MoeKeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        return {
            'input_ids': input_ids,
            'images': image_tensor,
            #'return_dict': True,
            #'stopping_criteria': [stopping_criteria]
        }
    elif last_part_model_name.startswith('mplug'):
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX as MPLUG_IMAGE_TOKEN_INDEX
        from mplug_owl2.constants import DEFAULT_IMAGE_TOKEN as MPLUG_DEFAULT_IMAGE_TOKEN
        from mplug_owl2.conversation import conv_templates as mplug_conv_templates
        from mplug_owl2.mm_utils import process_images as mplug_process_images
        from mplug_owl2.mm_utils import tokenizer_image_token as mplug_tokenizer_image_token

        conv = mplug_conv_templates["mplug_owl2"].copy()
        max_edge = max(image_size)  # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = mplug_process_images([image], image_processor)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        qs = MPLUG_DEFAULT_IMAGE_TOKEN + qs
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = mplug_tokenizer_image_token(
            prompt, tokenizer, MPLUG_IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(model.device)
        return {
            'input_ids': input_ids,
            'images': image_tensor
        }
    elif last_part_model_name.startswith('MobileVLM'):
        from models_utils.mobilevlm.conversation import conv_templates as mvlm_conv_templates
        from models_utils.mobilevlm.utils import (
            process_images as mvlm_process_images,
            tokenizer_image_token as mvlm_tokenizer_image_token
        )
        from models_utils.mobilevlm.constants import (
            IMAGE_TOKEN_INDEX as MVLM_IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN as MVLM_DEFAULT_IMAGE_TOKEN
        )
        images_tensor = mvlm_process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)

        conv = mvlm_conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], MVLM_DEFAULT_IMAGE_TOKEN + "\n" + qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = (mvlm_tokenizer_image_token(
            prompt, tokenizer, MVLM_IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda())
        return {
            'input_ids': input_ids,
            'images': images_tensor,
        }
    else:
        raise NotImplementedError