# FinLoRA: Finetuning Qauntized Financial Large Language Models using Low-Rank Adaptation


### Motivation
Large Language Models (LLMs) have shown remarkable performance, but pre-training and fine-tuning LLMs can be computationally expensive. The low-rank and quantization techniques show success. Our project aims to provide a thorough evaluation of these techniques with a focus on financial applications.

### Key Methodology
The paper FinGPT-HPC [1] explores low-rank structure and quantization techniques in pretraining and finetuning LLMs and evaluates its performance on both general tasks and financial tasks, and it shows significant speedup and lower GPU memory consumption. This ICDCS conference paper will be extended into a journal paper by employing more comprehensive testing.

Task 1: Evaluate performance on larger and newer LLMs. The current paper tested GPT-2-127M, GPT-2-1.5B, Llama-2-7B, and Llama-2-13B. We plan on testing on newer and larger LLMs including Llama-3.1-8B and Llama-3.1-70B. This will provide a more comprehensive evaluation of the proposed techniques across a range of LLM sizes and architectures.

Task 2: Explore more low–rank and quantization techniques. Other LoRA techniques [5], such as information retention QLora (IR-QLoRA) [2], LoRA+ [3], and Random Subspace Adaptation (ROSA) [4], showed improved performance in the fine-tuning stage. We plan to test these techniques on financial tasks to evaluate its domain-specific performance and make comparisons. 

Task 3: More comprehensive evaluation with more diverse financial test datasets. The present test datasets incorporate both sentiment analysis and named entity recognition tasks [1]. Our objective is to expand the range of tasks, such as XBRL files, news headline classification, for a more thorough assessment.

### Expected Outcomes 
This project would produce comprehensive understanding in using multiple novel low-rank [5] and quantization structures in both pretraining and finetuning stages, and its performance for both general tasks and financial tasks.
The first milestone is an extended evaluation using newer LLMs, low-rank techniques, and datasets.
The second milestone involves extending this conference paper [1] into a longer journal paper by inserting new methodologies and results.

### References
[1] Xiao-Yang Liu, Jie Zhang, Guoxuan Wang, Weiqing Tong, Anwar Walid. FinGPT-HPC: Efficient Pretraining and Finetuning Large Language Models for Financial Applications with High-Performance Computing. IEEE ICDCS 2024. 
[2] Mao, Y., Ge, Y., Fan, Y., Xu, W., Mi, Y., Hu, Z. and Gao, Y., 2024. A Survey on LoRA of Large Language Models. arXiv preprint arXiv:2407.11046.
[3] Vlad Fomenko, Han Yu, Jongho Lee, Stanley Hsieh, Weizhu Chen. A Note on LoRA, 2024. https://arxiv.org/abs/2404.05086 
  

