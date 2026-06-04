# "Principal Components" Enable A New Language of Images
### (ICCV 2025) A New Paradigm for Compact and Interpretable Image Representations
<a href="https://arxiv.org/abs/2503.08685">[Read the Paper]</a> &nbsp; | &nbsp;
<a href="https://visual-gen.github.io/semanticist/">[Project Page]</a> &nbsp; | &nbsp;
<a href="https://huggingface.co/tennant/semanticist/tree/main">[Model Checkpoints]</a>

[Xin Wen](https://wen-xin.info/)<sup>1*</sup>, 
[Bingchen Zhao](https://bzhao.me/)<sup>2*</sup>, 
[Ismail Elezi](https://therevanchist.github.io/)<sup>3</sup>, 
[Jiankang Deng](https://jiankangdeng.github.io/)<sup>4</sup>, 
[Xiaojuan Qi](https://xjqi.github.io/)<sup>1</sup>
<br/>
<small><sup>*</sup> Equal Contribution &nbsp;</small>
<br/>
    <sup>1</sup> University of Hong Kong &nbsp; | &nbsp;
    <sup>2</sup> University of Edinburgh
<br/>
    <sup>3</sup> Huawei London Research Centre &nbsp; | &nbsp;
    <sup>4</sup> Imperial College London

![Semanticist Teaser](pages/figs/teaser.jpg)

## Introduction & Motivation
Deep generative models have revolutionized image synthesis, but how we tokenize visual data remains an open question. 
While classical methods like **Principal Component Analysis (PCA)** introduced compact, structured representations, modern **visual tokenizers**—from **VQ-VAE** to **SD-VAE**—often prioritize **reconstruction fidelity** at the cost of interpretability and efficiency.

### The Problem

- **Lack of Structure:** Tokens are arbitrarily learned, without an ordering that prioritizes important visual features first.
- **Semantic-Spectrum Coupling:** Tokens entangle *high-level semantics* with *low-level spectral details*, leading to inefficiencies in downstream applications.

Can we design a **compact, structured tokenizer** that retains the benefits of PCA while leveraging modern generative techniques?

### Key Contributions (What's New?)
- **📌 PCA-Guided Tokenization:** Introduces a *causal ordering* where earlier tokens capture the most important visual details, reducing redundancy.
- **⚡ Semantic-Spectrum Decoupling:** Resolves the issue of semantic-spectrum coupling to ensure tokens focus on high-level semantic information.
- **🌀 Diffusion-Based Decoding:** Uses a *diffusion decoder* for the spectral auto-regressive property to naturally separate semantic and spectral content.
- **🚀 Compact & Interpretability-Friendly:** Enables *flexible token selection*, where fewer tokens can still yield high-quality reconstructions.

For more details, please refer to our [project page](https://visual-gen.github.io/semanticist/).

## Getting Started

### Preparation

First please make sure pytorch is installed (we used 2.5.1 but we expect any version >= 2.0 to work).

Then install the rest of the dependencies.

```
pip install -r requirements.txt
```

Please then download [ImageNet](https://www.image-net.org/) and soft-link it to `./dataset/imagenet`. For evaluating FID, it is recommended to pre-process the validation set of ImageNet with [this script](https://github.com/LTH14/rcg/blob/main/prepare_imgnet_val.py). The target folder is `./dataset/imagenet/val256` in our case.

### Training

Our codebase supports DDP training with accelerate, torchrun, and submitit (for slurm users). To train a Semanticist tokenizer with DiT-L tokenizer on 8 GPUs, you can run
```bash
accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_l.yaml
```
or
```bash
torchrun --nproc-per-node=8 train_net.py --cfg configs/tokenizer_l.yaml
```
or
```bash
python submitit_train.py --ngpus=8 --nodes=1 --partition=xxx --config configs/tokenizer_l.yaml
```
We used a global batch size of 2048 and thus the effective batch size per GPU is 256 in this case. Your may modify the batch size and gradient accumulation steps in the config file accrrding to your training resources.

To train a ϵLlamaGen autoregressive model with a tokenizer trained as above, you can run the following command. Remember to change the path to the tokenizer in the config file. The EMA model is `custom_checkpoint_1.pkl` under the output folder.
```bash
accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/autoregressive_l.yaml
```
Note that caching is enabled by default and it takes around 400GB memory (dumped to `/dev/shm`) for ten_crop augmentation on ImageNet. If you want to disable it, you can set `enable_cache_latents` to False in the config file and/or specify a different data augmentation method (e.g., centercrop_cached, centercrop, randcrop).

### Evaluation

By default, when evaluating online we do not use the EMA model. Thus to obtain the final performance, you are suggested to perform a separate evaluation after training. Like above, our scripts are compatible with accelerate, torchrun, and submitit.
```bash
accelerate launch --config_file=configs/onenode_config.yaml test_net.py --model ./output/tokenizer/models_l --step 250000 --cfg_value 3.0 --test_num_slots 32
```
or
```bash
torchrun --nproc-per-node=8 test_net.py --model ./output/tokenizer/models_l --step 250000 --cfg_value 3.0 --test_num_slots 32
```
or
```bash
python submitit_eval.py --ngpus=8 --nodes=1 --partition=xxx --model ./output/tokenizer/models_l --step 250000 --cfg_value 3.0 --test_num_slots 32
```
And for the AR model:
```bash
torchrun --nproc-per-node=8 test_net.py --model ./output/autoregressive/models_l --step 250000 --cfg_value 6.0 --ae_cfg 1.0 --test_num_slots 32
```
If `enable_ema` is set to True, the EMA model will be loaded automatically. You can adjust the number of GPUs flexibly. You can also specify multiple arguments in the command line to perform a grid search.

### Demos

Please refer to our demo pages on Huggingface for the tokenizer and the AR model.
- [Tokenizer Demo](https://huggingface.co/spaces/tennant/semanticist_tokenizer)
- [AR Demo](https://huggingface.co/spaces/tennant/Semanticist_AR)

## Note

It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. Additionally, we'll make an effort to refine the README and code, and carry out sanity-check experiments in the near future.

## Acknowledgements

Our codebase builds upon several existing publicly available codes. Specifically, we have modified or taken inspiration from the following repos: [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT), [DiffAE](https://github.com/phizaz/diffae), [LlamaGen](https://github.com/FoundationVision/LlamaGen), [RCG](https://github.com/LTH14/rcg), [MAR](https://github.com/LTH14/mar), [REPA](https://github.com/sihyun-yu/REPA), etc. We thank the authors for their contributions to the community.

## Citation

If you find this work useful in your research, please consider citing us!

```bibtex
@inproceedings{semanticist,
    title={``{P}rincipal Components'' Enable A New Language of Images},
    author={Wen, Xin and Zhao, Bingchen and Elezi, Ismail and Deng, Jiankang and Qi, Xiaojuan},
    booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}
}
```
