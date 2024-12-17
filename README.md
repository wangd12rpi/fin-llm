# FinLoRA: Finetuning Quantized Large Language Models for Financial Applications Using Low-Rank Adaptation

FinLoRA: Open counterpart of BloombergGPT

## Introduction

### Motivation

Financial large Language Models (FinLLMs) have shown remarkable performance, but pre-training LLMs can be
computationally expensive. The low-rank adaptation and quantization techniques are promising. 

### Current XBRL Tasks

| Name                  | Category | Train samples | Link |
|:----------------------|:---------|:--------------|:-----|
| XBRL Tags Extraction  | XBRL QA  | > 500         | -    |
| XBRL Value Extraction | XBRL qA  | > 2K          | -    |

## Next Steps

### Additional XBRL Tasks

Current XBRL tasks are limit in variety and the datasets are new and might be perceived as unreliable, therefore more
established XBRL tasks are required.

| Name                             | Category         | Train samples | Link                                                                                                                          |
|:---------------------------------|:-----------------|:--------------|:------------------------------------------------------------------------------------------------------------------------------|
| FiNER                            | XBRL Tagging     | 900K          | [HF](https://huggingface.co/datasets/nlpaueb/finer-139?row=16)                                                                |
| FNXL                             | XBRL Tagging     | 1K            | [GitHub](https://github.com/soummyaah/FNXL)                                                                                   |
| XBRL Term                        | XBRL Terminology | -             | [GitHub](https://github.com/KirkHan0920/XBRL-Agent/blob/main/Datasets/XBRL%20Terminology.xlsx)                                |
| Financial Math                   | Math             | -             | [GitHub](https://github.com/KirkHan0920/XBRL-Agent/blob/main/Datasets/formulas_with_explanations_with_questions_with_gt.xlsx) |
| XBRL Formula Formatted with Tags | XBRL QA          | > 2K          | -                                                                                                                             |
| XBRL Formula Calculations        | XBRL QA          | > 2K          | -                                                                                                                             |

### Cross-task Generalization (LoRA MoE)

Currently we finetune one LoRA adaptor for every task. Although single-task finetuning have higher performance, it might
not be practical in application.

Mixture of LoRA Experts (LoRA MoE): each LoRA module acts as an expert, a router network assigns the LoRA weights. One
implementation is [X-LoRA](https://arxiv.org/pdf/2402.07148) [4]. X-LoRA is built on top of huggingface PEFT, therefore
implementation should be relatively straightforward.

### Improve Performance and Scalability for Inference

S-LoRA [5] is designed for serving many LoRA adapters efficiently. It stores all adapters in the memory and
fetches the adapters needed to GPU memory. It might be possible to use some of the ideas of SLoRA with LoRA MoE for a
more efficient implementation of LoRA MoE.

Difficulty: Current S-LoRA implementation does not work with HuggingFace, and does not support newer model like Llama 3.

### Federated Learning for Financial Data Privacy

Federated Learning setting: Multiple institutions (banks) might collaborate in the finetuning task using locally stored data.
Differentially Private Low-Rank Adaptation (DP-LoRA) [6] offers an approach by adding noise in weight updates to avoid inferring sensitive information from model weights/outputs. Adding zero-knowledge learning on top of DP-LoRA may enhance privacy-preserving.

## References

[1] Xiao-Yang Liu, Jie Zhang, Guoxuan Wang, Weiqing Tong, Anwar Walid. FinGPT-HPC: Efficient Pretraining and Finetuning
Large Language Models for Financial Applications with High-Performance Computing. IEEE ICDCS 2024.

[2] Mao, Y., Ge, Y., Fan, Y., Xu, W., Mi, Y., Hu, Z. and Gao, Y., 2024. A Survey on LoRA of Large Language Models. arXiv
preprint arXiv:2407.11046.

[3] Vlad Fomenko, Han Yu, Jongho Lee, Stanley Hsieh, Weizhu Chen. A Note on LoRA, 2024. https://arxiv.org/abs/2404.05086

[4] E.L. Buehler, M.J. Buehler. X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language
Models with Applications in Protein Mechanics and Design. APL Machine Learning, 2024.

[5] Sheng, Ying and Cao, Shiyi and Li. Dacheng and Hooper, et al. S-LoRA: Serving Thousands of Concurrent LoRA
Adapters, https://arxiv.org/pdf/2311.03285

[6] Xiao-Yang Liu, Rongyi Zhu, Daochen Zha, Jiechao Gao, Shan Zhong, Matt White, Meikang Qiu, Differentially Private Low-Rank Adaptation of Large Language Model Using Federated
Learning, ACM Transactions on Management Information Systems, 2024.

[7] Dannong Wang, Daniel Kim, Bo Jin, Xingjian Zhao, Tianfan Fu, Steve Yang, and Xiao-Yang Liu. FinLoRA: Finetuning quantized financial large language models using low-rank adaptation. AAAI Workshop on Connecting Low-Rank Representations in AI, 2025
