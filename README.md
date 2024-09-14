# Efficient Pretraining and Finetuning Large Language Models


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

### Estimated Project Timeline
09/01 ~ 09/15/2024: Formulate a plan for testing new LLMs, low-rank techniques, and datasets. Specify which LLMs and datasets will be included in the testing process. 
09/16 ~ 10/31/2024: Complete codes required for testing, and start the evaluation process. 
10/31 ~ 11/30/2024: Complete a journal paper with new results.



### References
[1] Xiao-Yang Liu, Jie Zhang, Guoxuan Wang, Weiqing Tong, Anwar Walid. FinGPT-HPC: Efficient Pretraining and Finetuning Large Language Models for Financial Applications with High-Performance Computing. IEEE ICDCS 2024.  
[2] Qin, H., Ma, X., Zheng, X., Li, X., Zhang, Y., Liu, S., ... & Magno, M. (2024). Accurate LoRA-finetuning quantization of LLMs via information retention.   
[3] Hayou, S., Ghosh, N., & Yu, B. (2024). Lora+: Efficient low rank adaptation of large models.   
[4] Hameed, M. G. A., Milios, A., Reddy, S., & Rabusseau, G. (2024). ROSA: Random Subspace Adaptation for Efficient Fine-Tuning.  
[5] Mao, Y., Ge, Y., Fan, Y., Xu, W., Mi, Y., Hu, Z. and Gao, Y., 2024. A Survey on LoRA of Large Language Models. arXiv preprint arXiv:2407.11046.  

