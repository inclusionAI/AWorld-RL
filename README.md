<div align="center">

<h1 align="center">
Agentic Learning Powered by <a href="https://github.com/inclusionAI/AWorld"><img src="assets/aworld_logo.png" alt="AWorld Logo" height="32" style="vertical-align: text-bottom; margin-right: 4px;">AWorld</a>
</h1>

</div>

<p align="center">
<img src="./assets/arxiv.png" width="14px" style="display:inline;"> <a href="https://arxiv.org/abs/2508.13634" target="_blank">arXiv(V2P)</a> ｜
<img src="./assets/arxiv.png" width="14px" style="display:inline;"> <a href="https://arxiv.org/abs/2507.02962" target="_blank">arXiv(RAG-R1)</a> ｜
<img src="./assets/arxiv.png" width="14px" style="display:inline;"> <a href="https://arxiv.org/abs/2505.20192" target="_blank">arXiv(FunReason)</a> ｜
<img src="./assets/arxiv.png" width="14px" style="display:inline;"> <a href="https://arxiv.org/abs/2510.10197" target="_blank">arXiv(EnvTuning)</a>
</p>

<p align="center">
🤗 <a href="https://huggingface.co/papers/2508.13634" target="_blank">Paper(V2P)</a> ｜
🤗 <a href="https://huggingface.co/papers/2507.02962" target="_blank">Paper(RAG-R1)</a> ｜
🤗 <a href="https://huggingface.co/papers/2505.20192" target="_blank">Paper(FunReason)</a> ｜
🤗 <a href="https://huggingface.co/papers/2510.10197" target="_blank">Paper(EnvTuning)</a>
</p>

<p align="center">
<img src="./assets/xiaohongshu.png" width="14px" style="display:inline;"> <a href="http://xhslink.com/o/A5W5duyHWlf" target="_blank">EnvTuning</a>
</p>

## 📣 News
[2025/08/19] 🔥🔥🔥[**V2P**](./V2P) We propose **V2P**, a novel training method for multi-modal models that enables coordinate-free, human-like visual GUI Grounding.

[2025/07/01] 🔥🔥🔥[**RAG-R1**](./RAG-R1) We propose **RAG-R1**, a deepsearch training framework that incentivizing the search and reasoning capabilities of LLMs through multi-query parallelism.

[2025/05/16] 🔥🔥🔥[**FunReason**](https://github.com/BingguangHao/FunReason/) We propose **FunReason**, a novel framework that enhances LLMs' function calling capabilities through an automated data refinement strategy and a Self-Refinement Multiscale Loss approach.

## 📖 Introduction

**AWorld-RL** is a comprehensive collection of cutting-edge agentic reinforcement learning algorithms developed by the AWorld Team. Built upon the [AWorld Framework](https://github.com/inclusionAI/AWorld), this repository provides complete **codebases**, **datasets**, and **checkpoints** for training and evaluating autonomous agents that learn through multi-turn interactions with dynamic environments.

Our work focuses on enabling agents to effectively leverage environmental feedback for complex problem-solving across diverse domains, including multi-modal understanding, deep search, and function calling.

![AgenticLearning Framework](assets/framework.png "AgenticLearning Framework")

## 🚀 Projects

**[Don't Just Fine-tune the Agent, Tune the Environment](./EnvTuning)**  
**Authors:** Siyuan Lu, Zechuan Wang, Hongxuan Zhang, Qintong Wu, Leilei Gan, Chenyi Zhuang, Jinjie Gu, Tao Lin  
[![arXiv](https://img.shields.io/badge/arXiv-2510.10197-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.10197) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2510.10197)

**[V2P: From Background Suppression to Center Peaking for Robust GUI Grounding](./V2P)**  
**Authors:** Jikai Chen, Long Chen, Dong Wang, Leilei Gan, Chenyi Zhuang, Jinjie Gu  
[![arXiv](https://img.shields.io/badge/arXiv-2508.13634-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2508.13634) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2508.13634) [![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow?logo=huggingface)](https://huggingface.co/inclusionAI/V2P-7B)

**[RAG-R1: Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism](./RAG-R1)**  
**Authors:** Zhiwen Tan, Jiaming Huang, Qintong Wu, Hongxuan Zhang, Chenyi Zhuang, Jinjie Gu  
[![arXiv](https://img.shields.io/badge/arXiv-2507.02962-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.02962) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2507.02962)

**[FunReason: Enhancing Large Language Models' Function Calling via Self-Refinement Multiscale Loss and Automated Data Refinement](https://github.com/BingguangHao/FunReason/)**  
**Authors:** Bingguang Hao, Maolin Wang, Zengzhuang Xu, Cunyin Peng, Yicheng Chen, Xiangyu Zhao, Jinjie Gu, Chenyi Zhuang  
[![arXiv](https://img.shields.io/badge/arXiv-2505.20192-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2505.20192) [![Paper](https://img.shields.io/badge/Hugging%20Face-Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2505.20192)
                         
## 📚 Overview

### Table of Contents

- [Multi-Modal](#multi-modal)
  - [V2P](#v2p)
- [Deepsearch](#deepsearch)
  - [RAG-R1](#rag-r1)
- [FunctionCall](#functioncall)
  - [FunReason](#funreason)

### Multi-Modal
#### [V2P](./V2P) 

- Tools: PyAutoGUI Tools
- LLM: Qwen2.5-7b-instruct

![V2P-framework](V2P/assets/main.png)

<h5 align="center">Overall framework of V2P.</h5>

![V2P-result](V2P/assets/results.png)

<h5 align="center">Performance on both SreenSpot-v2 (left) and ScreenSpot-Pro (right).</h5>
  

### Deepsearch

#### [RAG-R1](./RAG-R1)

- Tools: Search Engines (offline or [online](https://github.com/qingw-dev/aworld-mcp-servers))
- LLM: Qwen2.5-7b-instruct

![RAG-R1-framework](RAG-R1/assets/RAG-R1.png)

<h5 align="center">Overall framework of RAG-R1.</h5>

![RAG-R1-result](RAG-R1/assets/RAG-R1-result.png)

<h5 align="left">Performance comparisons on QA benchmarks under the EM metric. The best and second
best results are bold and underlined, respectively.</h5>

### FunctionCall

#### [FunReason](https://github.com/BingguangHao/FunReason/)

- Tools: Real Human Function calling (BFCLv2 live&non-live)
- LLM: Qwen2.5-7b-Coder-instruct

FunReason is a framework designed to enhance LLMs' function calling capabilities, achieving GPT-4o-comparable performance on BFCL, surpassing RL-based methods, mitigating catastrophic forgetting on HumanEval and MBPP, and using a data refinement strategy where natural CoT data outperforms artificial ones.

![FunReason-Performance](FunctionCall/assets/Fun_pipline.png)

<h5 align="center">Data refinement pipline of FunReason.</h5>

**Overview of FunReason's data refinement pipeline.** The pipeline consists of five stages: Function Call Classification, Query and Tool Identification, CoT Identification, Function and Parameter Identification, and Format Identification. Each stage ensures specific aspects of data quality, with failing examples either being discarded or regenerated.

![FunReason-Performance](FunctionCall/assets/Fun_per.png)

<h5 align="center">Performance of FunReason.</h5>

## Citation

Please cite our repo if our works are helpful for your research.
```
@article{chen2025v2p,
  title={V2P: From Background Suppression to Center Peaking for Robust GUI Grounding Task},
  author={Chen, Jikai and Chen, Long and Wang, Dong and Gan, Leilei and Zhuang, Chenyi and Gu, Jinjie},
  journal={arXiv preprint arXiv:2508.13634},
  year={2025}
}

@article{tan2025rag,
  title={RAG-R1: Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism},
  author={Tan, Zhiwen and Huang, Jiaming and Wu, Qintong and Zhang, Hongxuan and Zhuang, Chenyi and Gu, Jinjie},
  journal={arXiv preprint arXiv:2507.02962},
  year={2025}
}

@article{hao2025funreason,
  title={FunReason: Enhancing Large Language Models' Function Calling via Self-Refinement Multiscale Loss and Automated Data Refinement},
  author={Hao, Bingguang and Wang, Maolin and Xu, Zengzhuang and Peng, Cunyin and Chen, Yicheng and Zhao, Xiangyu and Gu, Jinjie and Zhuang, Chenyi},
  journal={arXiv preprint arXiv:2505.20192},
  year={2025}
}
```

## 📞 Contact

For any question or feedback, please reach out to us at [ender.tzw@antgroup.com](mailto:ender.tzw@antgroup.com) or [chenyi.zcy@antgroup.com](mailto:chenyi.zcy@antgroup.com)

## License

This project is licensed under the [MIT License](LICENSE).
