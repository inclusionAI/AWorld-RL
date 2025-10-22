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
[2025/10/22] 🔥🔥🔥[**EnvTuning**](./EnvTuning) We propose **Environment Tuning**, a novel training paradigm that enables agents to learn complex multi-turn tool use behaviors through environmental interaction rather than trajectory imitation, achieving significant improvements with only 400 training samples.

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
- [Tool Use](#tool-use)
  - [Environment Tuning](#environment-tuning)
  - [FunReason](#funreason)

### Multi-Modal
#### [V2P](./V2P) 

- Tools: PyAutoGUI Tools
- LLM: Qwen2.5-7b-instruct

<div align="center">
  <img src="V2P/assets/main.png" alt="V2P-framework">
  <p>Overall framework of V2P.</p>
</div>

<div align="center">
  <img src="V2P/assets/results.png" alt="V2P-result">
  <p>Performance on both SreenSpot-v2 (left) and ScreenSpot-Pro (right).</p>
</div>
  

### Deepsearch

#### [RAG-R1](./RAG-R1)

- Tools: Search Engines (offline or [online](https://github.com/qingw-dev/aworld-mcp-servers))
- LLM: Qwen2.5-7b-instruct

<div align="center">
  <img src="RAG-R1/assets/RAG-R1.png" alt="RAG-R1-framework">
  <p>Overall framework of RAG-R1.</p>
</div>

<div align="center">
  <img src="RAG-R1/assets/RAG-R1-result.png" alt="RAG-R1-result">
  <p>Performance comparisons on QA benchmarks under the EM metric. The best and second best results are bold and underlined, respectively.</p>
</div>

### Tool Use
#### [Environment Tuning](./EnvTuning)

- Tools: Multi-turn Tool Use (BFCL Benchmark)
- LLM: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, watt-tool-8B

Training agents for complex multi-turn tool use tasks faces critical challenges: extreme scarcity of high-quality training data, overfitting with supervised fine-tuning (SFT) on synthetic data, and cold-start problems with training instability in standard reinforcement learning approaches. **Environment Tuning** addresses these challenges through a novel training paradigm that enables agents to learn complex behaviors through environmental interaction rather than trajectory imitation, even with minimal data.

<div align="center">
  <img src="EnvTuning/assets/introduction.png" alt="EnvTuning-introduction">
  <p>Limitations of existing paradigms (SFT overfitting and standard RL cold-start) and the advantages of Environment Tuning approach.</p>
</div>

<div align="center">
  <img src="EnvTuning/assets/pipeline.png" alt="EnvTuning-pipeline">
  <p>Four-stage curriculum learning pipeline with actionable environment augmentation and fine-grained progress rewards.</p>
</div>

<div align="center">
  <img src="EnvTuning/assets/main_results.png" alt="EnvTuning-results">
  <p>With only 400 training samples, Environment Tuning achieves significant improvements on BFCL V3.</p>
</div>


#### [FunReason](https://github.com/BingguangHao/FunReason/)

- Tools: Real Human Function calling (BFCLv2 live&non-live)
- LLM: Qwen2.5-7b-Coder-instruct

FunReason is a framework designed to enhance LLMs' function calling capabilities, achieving GPT-4o-comparable performance on BFCL, surpassing RL-based methods, mitigating catastrophic forgetting on HumanEval and MBPP, and using a data refinement strategy where natural CoT data outperforms artificial ones.

<div align="center">
  <img src="FunReason/assets/Fun_pipline.png" alt="FunReason-Performance">
  <p>Data refinement pipeline of FunReason.</p>
</div>

**Overview of FunReason's data refinement pipeline.** The pipeline consists of five stages: Function Call Classification, Query and Tool Identification, CoT Identification, Function and Parameter Identification, and Format Identification. Each stage ensures specific aspects of data quality, with failing examples either being discarded or regenerated.

<div align="center">
  <img src="FunReason/assets/Fun_per.png" alt="FunReason-Performance">
  <p>Performance of FunReason.</p>
</div>

## Citation

Please cite our repo if our works are helpful for your research.
```
@article{lu2025don,
  title={Don't Just Fine-tune the Agent, Tune the Environment},
  author={Lu, Siyuan and Wang, Zechuan and Zhang, Hongxuan and Wu, Qintong and Gan, Leilei and Zhuang, Chenyi and Gu, Jinjie and Lin, Tao},
  journal={arXiv preprint arXiv:2510.10197},
  year={2025}
}

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
