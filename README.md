# AgenticLearning

## Introduction

Agent exhibits powerful capabilities by interacting with the external environment and making decisions based on the feedback it receives from the environment.
For complex problems, it is often necessary for an agent to have multi-turn interactions with the environment to reach a solution. The complexity and dynamism of environments, coupled with the necessity for multi-turn interactions, pose numerous challenges in training agents.

We introduce **AgenticLearning**, an open-source agent training paradigm designed to empower researchers to train and evaluate autonomous agents effectively. AgenticLearning offers a framework for multi-turn interactions with the environment, enabling models to learn how to interact with the environment and make decisions based on its feedback, thereby enhancing the models' ability to leverage the environment to solve complex problems.

![AgenticLearning Framework](assets/framework.png "AgenticLearning Framework")

## News

[2025/07/01] 🔥🔥🔥[**RAG-R1**](https://github.com/inclusionAI/AgenticLearning/blob/main/RAG-R1/README.md) We propose **RAG-R1**, a deepsearch training framework that incentivizing the search and reasoning capabilities of LLMs through multi-query parallelism.

[2025/05/16] 🔥🔥🔥[**FunReason**](https://github.com/BingguangHao/FunReason/) We propose **FunReason**, a novel framework that enhances LLMs' function calling capabilities through an automated data refinement strategy and a Self-Refinement Multiscale Loss approach.

## Advancements

### Deepsearch

#### [RAG-R1](RAG-R1/README.md)

- Environment: Search Engines (offline or [online](https://github.com/qingw-dev/aworld-mcp-servers))
- LLM: Qwen2.5-7b-instruct
- Algorithm: SFT + RL(PPO)

![RAG-R1-framework](RAG-R1/assets/RAG-R1.png)

<h5 align="center">Overall framework of RAG-R1.</h5>

![RAG-R1-result](RAG-R1/assets/RAG-R1-result.png)

<h5 align="left">Performance comparisons on QA benchmarks under the EM metric. The best and second
best results are bold and underlined, respectively.</h5>

### FunctionCall

#### [FunReason](https://github.com/BingguangHao/FunReason/)

- environment: Real Human Function calling (BFCLv2 live&non-live)
- LLM: Qwen2.5-7b-Coder-instruct

FunReason is a framework designed to enhance LLMs' function calling capabilities, achieving GPT-4o-comparable performance on BFCL, surpassing RL-based methods, mitigating catastrophic forgetting on HumanEval and MBPP, and using a data refinement strategy where natural CoT data outperforms artificial ones.

![FunReason-Performance](FunctionCall/assets/Fun_pipline.png)

<h5 align="center">Data refinement pipline of FunReason.</h5>

**Overview of FunReason's data refinement pipeline.** The pipeline consists of five stages: Function Call Classification, Query and Tool Identification, CoT Identification, Function and Parameter Identification, and Format Identification. Each stage ensures specific aspects of data quality, with failing examples either being discarded or regenerated.

![FunReason-Performance](FunctionCall/assets/Fun_per.png)

<h5 align="center">Performance of FunReason.</h5>

### Citation

Please cite our repo if our works are helpful for your research.
#### FunReason

```
@article{FunReason,
  title={FunReason: Enhancing Large Language Models' Function Calling via Self-Refinement Multiscale Loss and Automated Data Refinement},
  author={Bingguang Hao, Maolin Wang, Zengzhuang Xu, Cunyin Peng, Yicheng Chen, Xiangyu Zhao, Jinjie Gu, Chenyi Zhuang},
  journal={arXiv preprint arXiv:2505.20192},
  year={2025}
}

```

## Contact

For any question or feedback, please reach out to us at [ender.tzw@antgroup.com](mailto:ender.tzw@antgroup.com) or [chenyi.zcy@antgroup.com](mailto:chenyi.zcy@antgroup.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
