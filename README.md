# Uncertainty-Aware Evaluation for Vision-Language Models



<p align="center">
  <img src="images/logo.png" width="50%" />
  <p align="center">Two LLMs can achieve the same accuracy score but demonstrate different levels of uncertainty.</p>
</p>



## Introduction

## Datasets


## Evaluation



## Getting started


6 groups of models could be launch from one environment: LLaVa, CogVLM, Yi-VL, Qwen-VL,
internlm-xcomposer, MoE-LLaVA. This environment could be created by the following code:
```shell
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/haotian-liu/LLaVA.git 
pip install git+https://github.com/PKU-YuanGroup/MoE-LLaVA.git --no-deps
pip install deepspeed==0.9.5
pip install -r requirements.txt
pip install xformers==0.0.23 --no-deps
```
mPLUG-Owl model can be launched from the following environment:
```shell
python3 -m venv venv_mplug
source venv_mplug/bin/activate
git clone https://github.com/X-PLUG/mPLUG-Owl.git
cd mPLUG-Owl/mPLUG-Owl2
git checkout 74f6be9f0b8d42f4c0ff9142a405481e0f859e5c
pip install -e .
pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps
cd ../../
pip install -r requirements.txt
```
Monkey models can be launched from the following environment:
```shell
python3 -m venv venv_monkey
source venv_monkey/bin/activate
git clone https://github.com/Yuliang-Liu/Monkey.git
cd ./Monkey
pip install -r requirements.txt
pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps
cd ../
pip install -r requirements.txt
```

To check all models you can run ```scripts/test_model_logits.sh```


To work with Yi-VL:
```shell
apt-get install git-lfs
cd ../
git clone https://huggingface.co/01-ai/Yi-VL-6B
```


### Model logits

To get model logits in four benchmarks run command from `scripts/run.sh`.

### To quantify uncertainty by logits

```shell
python -m uncertainty_quantification_via_cp --result_data_path 'output' --file_to_write 'full_result.json'
```

### To get result tables by uncertainty

```shell
python -m make_tables --result_path 'full_result.json' --dir_to_write 'tables'
```

## Citation

```bibtex

```

## Acknowledgement

[LLM-Uncertainty-Bench](https://github.com/smartyfh/LLM-Uncertainty-Bench): conformal prediction applied to LLM. Thanks for the authors for providing the framework.


## Contact
We welcome suggestions to help us improve benchmark. For any query, please contact us at v.kostumov@ensec.ai. If you find something interesting, please also feel free to share with us through email or open an issue. Thanks!

