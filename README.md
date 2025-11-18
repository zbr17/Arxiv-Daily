# ArXiv 每日推荐

> 更新于北京时间：2025-11-18 22:05:33
> 已自动阅读了 665 篇最新的论文。
> 使用模型：doubao-seed-1-6-thinking-250715 | 消耗 Tokens：355227

## 快速导航

- [高效大模型训练与推理](#高效大模型训练与推理)
- [大模型安全与对齐](#大模型安全与对齐)
- [原生多模态大模型](#原生多模态大模型)
- [多模态智能体](#多模态智能体)
- [深度学习可解释性](#深度学习可解释性)
- [大模型新技术](#大模型新技术)
- [深度学习理论](#深度学习理论)

<h2 id='高效大模型训练与推理'>高效大模型训练与推理</h2>

### [Score: 9.0/10] CompressNAS : A Fast and Efficient Technique for Model Compression using Decomposition
- **Authors:** Sudhakar Sah, Nikhil Chabbra, Matthieu Durnerin
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11716](https://arxiv.org/abs/2511.11716)
- **Reason:** 提出基于MicroNAS的模型压缩框架CompressNAS，通过全局秩选择解决低秩分解的精度与效率权衡问题，在ResNet-18和YOLOv5上实现高效压缩，属于高效大模型训练与推理中的模型压缩方向。
Score: 9
Field: 高效大模型训练与推理

### [Score: 9.0/10] PipeDiT: Accelerating Diffusion Transformers in Video Generation with Task Pipelining and Model Decoupling
- **Authors:** Sijie Wang, Qiang Wang, Shaohuai Shi
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12056](https://arxiv.org/abs/2511.12056)
- **Reason:** 提出流水线框架PipeDiT加速视频生成的Diffusion Transformers，通过任务流水线与模型解耦提升推理速度（1.06x-4.02x）与内存效率，属于高效大模型训练与推理中的推理加速方向。
Score: 9
Field: 高效大模型训练与推理

### [Score: 9.0/10] RedVTP: Training-Free Acceleration of Diffusion Vision-Language Models Inference via Masked Token-Guided Visual Token Pruning
- **Authors:** Jingqi Xu, Jingxi Lu, Chenghao Li, Sreetama Sarkar, Souvik Kundu, Peter A. Beerel
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12428](https://arxiv.org/abs/2511.12428)
- **Reason:** 提出训练无关的视觉token剪枝方法，利用掩码token引导压缩扩散视觉语言模型的推理成本，显著提升吞吐量与延迟性能，属于高效大模型推理的关键突破。
Score: 9
Field: 高效大模型训练与推理

### [Score: 9.0/10] BitSnap: Checkpoint Sparsification and Quantization in LLM Training
- **Authors:** Qingping Li, Yanxin Peng, Baodong Wu, Shigang Li, Guohao Dai, Shengen Yan, Yu Wang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12376](https://arxiv.org/abs/2511.12376)
- **Reason:** 提出LLM训练中checkpoint的稀疏化与量化方法，动态适应不同训练阶段与模型架构，平衡压缩比、速度与精度，有效提升训练存储与加载效率，与高效大模型训练与推理方向直接相关
Score: 9
Field: 高效大模型训练与推理

### [Score: 9.0/10] SLMQuant:Benchmarking Small Language Model Quantization for Practical Deployment
- **Authors:** Jiacheng Wang, Yejun Zeng, Jinyang Guo, Yuqing Ma, Aishan Liu, Xianglong Liu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13023](https://arxiv.org/abs/2511.13023)
- **Reason:** 系统评估小语言模型的量化技术，揭示SLM与LLM在量化敏感性上的差异，提出SLM优化的量化原则，属于高效大模型训练与推理中的模型压缩方向，对小模型的边缘部署有重要价值。
Score: 9
Field: 高效大模型训练与推理

### [Score: 8.0/10] Teaching Prompts to Coordinate: Hierarchical Layer-Grouped Prompt Tuning for Continual Learning
- **Authors:** Shengqin Jiang, Tianqi Kong, Yuankai Qi, Haokui Zhang, Lina Yao, Quan Z. Sheng, Qingshan Liu, Ming-Hsuan Yang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12090](https://arxiv.org/abs/2511.12090)
- **Reason:** 提出分层分组提示调优方法，通过层组共享提示和根提示生成子提示的方式，解决持续学习中独立提示更新导致的灾难性遗忘问题，实验验证在四个基准数据集上优于现有方法，提升了提示协调能力和模型稳定性，属于高效大模型训练的重要优化。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] Learning from Dense Events: Towards Fast Spiking Neural Networks Training via Event Dataset Distillation
- **Authors:** Shuhan Ye, Yi Yu, Qixin Zhang, Chenqi Kong, Qiangqiang Wu, Kun Wang, Xudong Jiang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12095](https://arxiv.org/abs/2511.12095)
- **Reason:** 提出PACE数据集蒸馏框架，通过ST-DSM和PEQ-N模块将大规模事件数据集蒸馏为紧凑合成数据集，大幅减少脉冲神经网络的训练时间（>50×）和存储成本（6000×），同时保持85%的全数据集性能，助力脉冲神经网络的高效训练与边缘部署。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] Sparse by Rule: Probability-Based N:M Pruning for Spiking Neural Networks
- **Authors:** Shuhan Ye, Yi Yu, Qixin Zhang, Chenqi Kong, Qiangqiang Wu, Xudong Jiang, Dacheng Tao
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12097](https://arxiv.org/abs/2511.12097)
- **Reason:** 提出SpikeNM半结构化N:M剪枝框架，通过M-way basis-logit参数化和eligibility-inspired蒸馏从scratch训练稀疏脉冲神经网络，平衡了稀疏性（2:4）和准确性，生成硬件友好的稀疏模式，提升了脉冲神经网络的部署效率，属于高效大模型推理的关键技术。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] OmniSparse: Training-Aware Fine-Grained Sparse Attention for Long-Video MLLMs
- **Authors:** Feng Chen, Yefei He, Shaoxuan He, Yuanyu He, Jing Liu, Lequan Lin, Akide Liu, Zhaoyang Li, Jiyuan Zhang, Zhenbang Sun, Bohan Zhuang, Qi Wu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12201](https://arxiv.org/abs/2511.12201)
- **Reason:** 提出训练感知的细粒度稀疏注意力框架，通过查询选择、KV动态分配和KV缓存瘦身，解决长视频MLLM的高计算复杂度问题，实现2.7倍预填充加速和2.4倍解码内存减少，同时保持性能，属于高效大模型推理的关键优化。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] D$^{3}$ToM: Decider-Guided Dynamic Token Merging for Accelerating Diffusion MLLMs
- **Authors:** Shuochen Chang, Xiaofeng Zhang, Qingyang Liu, Li Niu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12280](https://arxiv.org/abs/2511.12280)
- **Reason:** 提出Decider引导的动态token合并方法，通过前一步生成的token构建视觉token重要性图，合并冗余token以缩短序列长度，加速扩散MLLM的推理过程，实验验证在保持性能的同时减少计算量，属于高效大模型推理的关键技术。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] Fast Reasoning Segmentation for Images and Videos
- **Authors:** Yiqing Shen, Mathias Unberath
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12368](https://arxiv.org/abs/2511.12368)
- **Reason:** 提出FastReasonSeg框架，通过数字孪生表示 decouple 感知与推理，结合蒸馏和强化学习优化，将模型压缩至0.6B参数（性能优于20倍参数模型），同时实现7.79 FPS吞吐量和2.1GB内存消耗，支持资源受限环境的实时推理分割，属于高效大模型部署的重要进展。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] MSLoRA: Multi-Scale Low-Rank Adaptation via Attention Reweighting
- **Authors:** Xu Yang, Gady Agam
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12400](https://arxiv.org/abs/2511.12400)
- **Reason:** 提出多尺度低秩适应方法，通过注意力重加权实现卷积与Transformer模型的参数高效微调，解决现有LoRA跨架构泛化问题，属于高效大模型训练的核心优化。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] VVS: Accelerating Speculative Decoding for Visual Autoregressive Generation via Partial Verification Skipping
- **Authors:** Haotian Dong, Ye Li, Rongwei Lu, Chen Tang, Shu-Tao Xia, Zhi Wang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13587](https://arxiv.org/abs/2511.13587)
- **Reason:** 提出VVS框架加速视觉自回归生成的投机解码，通过部分验证跳过减少目标模型前向传递次数（2.8×），属于高效大模型训练与推理中的推理加速
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] CacheFlow: Compressive Streaming Memory for Efficient Long-Form Video Understanding
- **Authors:** Shrenik Patel, Daivik Patel
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13644](https://arxiv.org/abs/2511.13644)
- **Reason:** 提出CacheFlow用于长视频理解的高效内存管理，通过动态token丢弃和压缩长期记忆，减少87%的token处理量，属于高效大模型训练与推理中的内存优化
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] LLM on a Budget: Active Knowledge Distillation for Efficient Classification of Large Text Corpora
- **Authors:** Viviana Luccioli, Rithika Iyengar, Ryan Panley, Flora Haberkorn, Xiaoyu Ge, Leland Crane, Nitish Sinha, Seung Jung Lee
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11574](https://arxiv.org/abs/2511.11574)
- **Reason:** 提出M-RARU主动学习方法，高效蒸馏LLM，减少80%样本需求，属于高效大模型训练与推理中的知识蒸馏，降低了LLM部署成本
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] The Anatomy of a Triton Attention Kernel
- **Authors:** Burkhard Ringlein, Jan van Lunteren, Radu Stoica, Thomas Parnell
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11581](https://arxiv.org/abs/2511.11581)
- **Reason:** 研究LLM推理核心的Triton注意力 kernel优化，实现跨GPU平台的高效推理，提升LLM infra性能，属于高效大模型推理关键技术。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] EcoSpa: Efficient Transformer Training with Coupled Sparsity
- **Authors:** Jinqi Xiao, Cheng Luo, Lingyi Huang, Cheng Yang, Yang Sui, Huy Phan, Xiao Zang, Yibiao Ying, Zhexiang Tang, Anima Anandkumar, Bo Yuan
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11641](https://arxiv.org/abs/2511.11641)
- **Reason:** 提出EcoSpa框架通过耦合稀疏性实现Transformer高效训练，提升内存利用率和训练速度，属于高效大模型训练的核心技术。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization
- **Authors:** Zhixiong Zhao, Fangxin Liu, Junjie Wang, Chenyang Guan, Zongwu Wang, Li Jiang, Haibing Guan
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11663](https://arxiv.org/abs/2511.11663)
- **Reason:** 提出SpecQuant通过谱分解和自适应截断实现LLM超低比特量化，提升推理效率并减少内存占用，属于高效大模型推理中的量化方向。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] Uncertainty Makes It Stable: Curiosity-Driven Quantized Mixture-of-Experts
- **Authors:** Sebastián Andrés Cajas Ordóñez (unknown), Luis Fernando Torres Torres (unknown), Mackenzie J. Meni (unknown), Carlos Andrés Duran Paredes (unknown), Eric Arazo (unknown), Cristian Bosch (unknown), Ricardo Simon Carbajo (unknown), Yuan Lai (unknown), Leo Anthony Celi (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11743](https://arxiv.org/abs/2511.11743)
- **Reason:** 提出好奇心驱动的量化混合专家框架，提升边缘设备推理效率和稳定性，属于高效大模型训练与推理的high compression方向
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] Conformal Constrained Policy Optimization for Cost-Effective LLM Agents
- **Authors:** Wenwen Si (unknown), Sooyong Jang (unknown), Insup Lee (unknown), Osbert Bastani (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11828](https://arxiv.org/abs/2511.11828)
- **Reason:** 提出conformal约束的策略优化，提升LLM agents的成本效益，属于高效大模型训练与推理的efficient inference方向
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] Scaling Law Analysis in Federated Learning: How to Select the Optimal Model Size?
- **Authors:** Xuanyu Chen, Nan Yang, Shuai Wang, Dong Yuan
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12188](https://arxiv.org/abs/2511.12188)
- **Reason:** 分析联邦学习中的模型缩放律，通过PAC-Bayes上界推导最优模型大小与客户端数量的关系，为联邦场景下的模型选择与训练效率优化提供理论指导，与高效大模型训练与推理方向相关
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] Optimal Self-Consistency for Efficient Reasoning with Large Language Models
- **Authors:** Austin Feng, Marius Alonso, Ambroise Odonnat
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12309](https://arxiv.org/abs/2511.12309)
- **Reason:** 分析自洽性（Self-Consistency）的缩放行为与样本效率，提出Blend-ASC动态分配样本的方法，在减少样本使用的同时保持推理性能，提升LLM推理效率，与高效大模型训练与推理方向相关
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] MACKO: Sparse Matrix-Vector Multiplication for Low Sparsity
- **Authors:** Vladimír Macko, Vladimír Boža
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13061](https://arxiv.org/abs/2511.13061)
- **Reason:** 提出MACKO-SpMV优化低稀疏度下的稀疏矩阵乘法，提升剪枝LLM的推理速度与内存效率，属于高效大模型训练与推理中的基础设施优化方向，对稀疏LLM的实际部署有重要意义。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] OTARo: Once Tuning for All Precisions toward Robust On-Device LLMs
- **Authors:** Shaoyuan Chen, Zhixuan Chen, Dawei Yang, Zhihang Yuan, Qiang Wu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13147](https://arxiv.org/abs/2511.13147)
- **Reason:** 提出OTARo方法实现一次微调支持多量化精度切换，提升On-Device LLM的鲁棒性，属于高效大模型训练与推理中的部署优化方向，对实际场景的精度自适应有重要意义。
Score: 8
Field: 高效大模型训练与推理

### [Score: 8.0/10] ParaDySe: A Parallel-Strategy Switching Framework for Dynamic Sequence Lengths in Transformer
- **Authors:** Zhixin Ou, Peng Liang, Jianchen Han, Baihui Liu, Linbo Qiao
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13198](https://arxiv.org/abs/2511.13198)
- **Reason:** 提出ParaDySe框架自适应切换Transformer的并行策略，解决动态序列的内存与通信瓶颈，属于高效大模型训练与推理中的训练基础设施优化方向，对长序列LLM训练有帮助。
Score: 8
Field: 高效大模型训练与推理

### [Score: 7.0/10] Breaking the Modality Wall: Time-step Mixup for Efficient Spiking Knowledge Transfer from Static to Event Domain
- **Authors:** Yuqi Xie, Shuhan Ye, Yi Yu, Chong Wang, Qixin Zhang, Jiazhen Xu, Le Shen, Yuanbin Qian, Jiangbo Qian, Guoqi Li
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12150](https://arxiv.org/abs/2511.12150)
- **Reason:** 提出时间步混合知识转移框架（TMKT），通过概率时间步混合和模态感知目标函数，减少静态图像到事件域的模态差距，加速脉冲神经网络的跨模态知识转移，实验验证在多个基准和backbone上的性能提升，属于高效大模型训练的跨模态优化。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] ActVAR: Activating Mixtures of Weights and Tokens for Efficient Visual Autoregressive Generation
- **Authors:** Kaixin Zhang, Ruiqing Yang, Yuan Zhang, Shan You, Tao Huang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12893](https://arxiv.org/abs/2511.12893)
- **Reason:** 针对视觉自回归模型的计算成本问题，提出ActVAR动态激活框架，在权重（FFN分解为专家子网络）和token（门控选择高潜力token）上引入双稀疏性，用知识蒸馏保持性能，提升生成效率。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] MCAQ-YOLO: Morphological Complexity-Aware Quantization for Efficient Object Detection with Curriculum Learning
- **Authors:** Yoonjae Seo, Ermal Elbasani, Jaehong Lee
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12976](https://arxiv.org/abs/2511.12976)
- **Reason:** 针对目标检测模型的量化问题，提出形态复杂度感知的空间自适应比特分配方法MCAQ-YOLO，结合课程学习提升量化模型的准确性和收敛效率，属于高效目标检测的量化技术。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] UNSEEN: Enhancing Dataset Pruning from a Generalization Perspective
- **Authors:** Furui Xu, Shaobo Wang, Jiajun Zhang, Chenghao Sun, Haixiang Tang, Linfeng Zhang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12988](https://arxiv.org/abs/2511.12988)
- **Reason:** 针对现有数据集剪枝方法的拟合-centric问题，提出从泛化角度剪枝的UNSEEN框架，通过未见过样本的模型评分提升coreset质量，减少训练数据同时保持性能，属于高效训练的数据集优化技术。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] Parameter-Efficient and Personalized Federated Training of Generative Models at the Edge
- **Authors:** Kabir Khan, Manju Sarkar, Anita Kar, Suresh Ghosh
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11585](https://arxiv.org/abs/2511.11585)
- **Reason:** 提出FedGen-Edge框架，通过LoRA实现生成模型的参数高效联邦训练，解决边缘设备资源限制问题，属于高效大模型训练方向。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] Beyond One-Way Pruning: Bidirectional Pruning-Regrowth for Extreme Accuracy-Sparsity Tradeoff
- **Authors:** Junchen Liu, Yi Sheng
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11675](https://arxiv.org/abs/2511.11675)
- **Reason:** 提出双向剪枝-再生方法提升模型精度-稀疏性权衡，实现极端压缩下的性能保持，属于高效大模型训练中的剪枝方向。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] Stratified Knowledge-Density Super-Network for Scalable Vision Transformers
- **Authors:** Longhua Li, Lei Qi, Xin Geng
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11683](https://arxiv.org/abs/2511.11683)
- **Reason:** 提出分层知识密度超网络实现ViT可扩展训练，支持不同资源约束下的模型生成，属于高效大模型训练中的超网络方向。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] Computation-aware Energy-harvesting Federated Learning: Cyclic Scheduling with Selective Participation
- **Authors:** Eunjeong Jeong (unknown), Nikolaos Pappas (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11949](https://arxiv.org/abs/2511.11949)
- **Reason:** 提出计算感知的能量收集联邦学习框架，提升训练效率，属于高效大模型训练与推理的efficient LLM training方向
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching
- **Authors:** Jiacheng Cheng, Xu Zhang, Guanghui Qiu, Yifang Zhang, Yinchuan Li, Kaiyuan Feng
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13144](https://arxiv.org/abs/2511.13144)
- **Reason:** 提出pFed1BS联邦学习框架，用一比特随机草图压缩双向通信，解决个性化联邦学习的通信开销问题，属于高效大模型训练与推理中的通信优化方向。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs
- **Authors:** Yuxiang Zhang, Zhengxu Yu, Weihang Pan, Zhongming Jin, Qiang Fu, Deng Cai, Binbin Lin, Jieping Ye
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13223](https://arxiv.org/abs/2511.13223)
- **Reason:** 提出TokenSqueeze压缩推理LLM的Token序列，在保持性能的同时减少token使用，解决推理中的 latency 与内存问题，属于高效大模型训练与推理中的推理优化方向。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] KForge: Program Synthesis for Diverse AI Hardware Accelerators
- **Authors:** Taras Sereda, Tom St. John, Burak Bartan, Natalie Serrino, Sachin Katti, Zain Asgar
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13274](https://arxiv.org/abs/2511.13274)
- **Reason:** 提出KForge框架用LLM agents生成跨硬件的GPU内核，解决AI加速器的程序合成问题，属于高效大模型训练与推理中的硬件优化方向，对多硬件平台的AI部署有实用价值。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] Hardware optimization on Android for inference of AI models
- **Authors:** Iulius Gherasim, Carlos García Sánchez
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13453](https://arxiv.org/abs/2511.13453)
- **Reason:** 研究Android平台的AI模型推理优化，比较量化与加速器的使用，属于高效大模型训练与推理中的边缘部署优化方向，对移动设备的AI推理效率提升有实用价值。
Score: 7
Field: 高效大模型训练与推理

### [Score: 7.0/10] Experience-Guided Adaptation of Inference-Time Reasoning Strategies
- **Authors:** Adam Stein, Matthew Trager, Benjamin Bowman, Michael Kleinman, Aditya Chattopadhyay, Wei Xia, Stefano Soatto
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11519](https://arxiv.org/abs/2511.11519)
- **Reason:** 提出EGuR框架，通过积累的经验在推理时动态生成定制化推理策略（涵盖LLM调用、工具、采样参数等），在多个基准测试中实现了准确率提升与计算成本降低，属于高效大模型推理方向的重要探索。
Score: 7
Field: 高效大模型训练与推理

### [Score: 6.0/10] Lightweight Time Series Data Valuation on Time Series Foundation Models via In-Context Finetuning
- **Authors:** Shunyu Wu, Tianyue Li, Yixuan Leng, Jingyi Suo, Jian Lou, Dan Li, See-Kiong Ng
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11648](https://arxiv.org/abs/2511.11648)
- **Reason:** 提出LTSV方法通过in-context finetuning实现时间序列基础模型的轻量级数据 valuation，提升数据利用效率，属于高效大模型训练中的数据方向。
Score: 6
Field: 高效大模型训练与推理

### [Score: 6.0/10] R-Tuning: Wavelet-Decomposed Replay and Semantic Alignment for Continual Adaptation of Pretrained Time-Series Models
- **Authors:** Tianyi Yin, Jingwei Wang, Chenze Wang, Han Wang, Jiexuan Cai, Min Liu, Yunlong Ma, Kun Gao, Yuting Song, Weiming Shen
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11685](https://arxiv.org/abs/2511.11685)
- **Reason:** 提出R-Tuning通过小波分解重放和语义对齐实现时间序列预训练模型持续适应，解决灾难性遗忘，属于高效大模型训练中的持续适应方向。
Score: 6
Field: 高效大模型训练与推理

<h2 id='大模型安全与对齐'>大模型安全与对齐</h2>

### [Score: 9.0/10] Defending Unauthorized Model Merging via Dual-Stage Weight Protection
- **Authors:** Wei-Jia Chen, Min-Yen Tsai, Cheng-Yi Lee, Chia-Mu Yu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11851](https://arxiv.org/abs/2511.11851)
- **Reason:** 提出双阶段权重保护框架MergeGuard，通过分布任务信息与注入结构化扰动防御未授权模型合并，保护模型所有权与安全性，属于大模型安全与对齐中的模型安全方向。
Score: 9
Field: 大模型安全与对齐

### [Score: 9.0/10] Rethinking Deep Alignment Through The Lens Of Incomplete Learning
- **Authors:** Thong Bach, Dung Nguyen, Thao Minh Le, Truyen Tran
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12155](https://arxiv.org/abs/2511.12155)
- **Reason:** 分析大语言模型安全对齐中的不完全学习问题（position-dependent gradient weakening导致的信号衰减），提出针对性的靶向完成方法提升对抗鲁棒性，对大模型安全对齐的机制理解与性能优化有重要意义
Score: 9
Field: 大模型安全与对齐

### [Score: 8.0/10] Suppressing VLM Hallucinations with Spectral Representation Filtering
- **Authors:** Ameen Ali, Tamim Zoabi, Lior Wolf
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12220](https://arxiv.org/abs/2511.12220)
- **Reason:** 提出光谱表示过滤（SRF）方法，通过特征协方差分析识别幻觉模式并衰减，无需修改模型或再训练，有效抑制VLM的幻觉生成，实验验证在LLaVA-1.5、MiniGPT-4等模型上提升faithfulness，属于大模型安全与对齐的核心问题解决。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] Model Inversion Attack Against Deep Hashing
- **Authors:** Dongdong Zhao, Qiben Xu, Ranxin Fang, Baogang Song
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12233](https://arxiv.org/abs/2511.12233)
- **Reason:** 提出首个针对深度哈希的扩散基模型 inversion 框架（DHMI），通过语义哈希中心和surrogate引导去噪优化，在黑盒设置下成功重建高保真图像，揭示深度哈希系统的隐私风险，属于大模型安全的重要研究方向。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] DINO-Detect: A Simple yet Effective Framework for Blur-Robust AI-Generated Image Detection
- **Authors:** Jialiang Shen, Jiyang Zheng, Yunqi Xue, Huajie Chen, Yu Yao, Hui Kang, Ruiqi Liu, Helin Gong, Yang Yang, Dadong Wang, Tongliang Liu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12511](https://arxiv.org/abs/2511.12511)
- **Reason:** 基于知识蒸馏实现模糊鲁棒的AI生成图像检测，解决真实场景下的图像真实性验证问题，属于大模型安全与对齐的核心应用。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] Beyond Pixels: Semantic-aware Typographic Attack for Geo-Privacy Protection
- **Authors:** Jiayi Zhu, Yihao Huang, Yue Cao, Xiaojun Jia, Qing Guo, Felix Juefei-Xu, Geguang Pu, Bin Wang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12575](https://arxiv.org/abs/2511.12575)
- **Reason:** 提出语义感知的排版攻击方法，解决大模型地理推理中的隐私泄露问题，属于大模型安全与对齐的隐私保护研究。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning
- **Authors:** Ankita Raj, Chetan Arora
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12735](https://arxiv.org/abs/2511.12735)
- **Reason:** 首次研究开放词汇目标检测器（OVODs）的后门攻击，提出TrAP多模态提示调优策略，通过轻量级prompt参数优化植入后门，保持模型泛化性的同时实现高攻击成功率，揭示了OVODs的新攻击表面。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] SAGA: Source Attribution of Generative AI Videos
- **Authors:** Rohit Kundu, Vishal Mohanty, Hao Xiong, Shan Jia, Athula Balachandran, Amit K. Roy-Chowdhury
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12834](https://arxiv.org/abs/2511.12834)
- **Reason:** 提出首个针对生成式AI视频的多粒度来源归因框架SAGA，支持真实性、生成任务、模型版本等五层归因，引入T-Sigs解释方法，解决深度伪造视频的溯源问题，属于大模型安全的重要应用方向。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] GrOCE:Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models
- **Authors:** Ning Han, Zhenyu Ge, Feng Han, Yuhua Sun, Chengqing Li, Jingjing Chen
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12968](https://arxiv.org/abs/2511.12968)
- **Reason:** 针对文本到图像扩散模型的概念擦除问题，提出训练-free的图引导在线概念擦除框架GrOCE，通过动态语义图推理实现精准概念移除，解决现有方法依赖微调或语义分离粗糙的问题，提升概念擦除的准确性和稳定性。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] SAGE: Spuriousness-Aware Guided Prompt Exploration for Mitigating Multimodal Bias
- **Authors:** Wenqian Ye, Di Wang, Guangtao Zheng, Bohan Liu, Aidong Zhang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13005](https://arxiv.org/abs/2511.13005)
- **Reason:** 针对CLIP的多模态偏差问题，提出SAGE引导prompt探索方法，通过选择诱导类间语义分离最大的prompt缓解偏差，提升鲁棒性，无需训练或微调，解决VLMs偏差痛点。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] VEIL: Jailbreaking Text-to-Video Models via Visual Exploitation from Implicit Language
- **Authors:** Zonghao Ying (), Moyang Chen (), Nizhang Li (), Zhiqiang Wang (), Wenxin Zhang (), Quanchen Zou (), Zonglei Jing (), Aishan Liu (), Xianglong Liu ()
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13127](https://arxiv.org/abs/2511.13127)
- **Reason:** 提出模块化提示设计的 jailbreak 框架，利用隐式语言线索诱导文本到视频模型生成不安全内容，揭示了多模态大模型的安全漏洞，对大模型安全与对齐研究有重要参考价值。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] VOPE: Revisiting Hallucination of Vision-Language Models in Voluntary Imagination Task
- **Authors:** Xingming Long, Jie Zhang, Shiguang Shan, Xilin Chen
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13420](https://arxiv.org/abs/2511.13420)
- **Reason:** 提出VOPE方法评估LVLM在自愿想象任务中的幻觉，属于大模型安全与对齐中的幻觉检测与评估，填补了想象任务中幻觉研究的空白
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] Output Supervision Can Obfuscate the Chain of Thought
- **Authors:** Jacob Drori, Luke Marks, Bryce Woodworth, Alex Cloud, Alexander Matt Turner
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11584](https://arxiv.org/abs/2511.11584)
- **Reason:** 分析输出监督对CoT的混淆效应，提出缓解方法以提升LLM CoT的可监控性，属于大模型安全与对齐中的关键问题。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] Beyond Superficial Forgetting: Thorough Unlearning through Knowledge Density Estimation and Block Re-insertion
- **Authors:** Feng Guo, Yuntao Wen, Shen Gao, Junshuo Zhang, Shuo Shang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11667](https://arxiv.org/abs/2511.11667)
- **Reason:** 提出KUnBR通过知识密度估计和块重插入实现LLM彻底遗忘，解决隐私与合规问题，属于大模型安全与对齐中的关键技术。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] On the Trade-Off Between Transparency and Security in Adversarial Machine Learning
- **Authors:** Lucas Fenaux (unknown), Christopher Srinivasa (unknown), Florian Kerschbaum (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11842](https://arxiv.org/abs/2511.11842)
- **Reason:** 研究透明度与安全的权衡，涉及对抗机器学习的安全问题，属于大模型安全与对齐的LLM safety方向
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] A Systematic Study of Model Extraction Attacks on Graph Foundation Models
- **Authors:** Haoyan Xu (unknown), Ruizhi Qian (unknown), Jiate Li (unknown), Yushun Dong (unknown), Minghao Lin (unknown), Hanson Yan (unknown), Zhengtao Yao (unknown), Qinghua Liu (unknown), Junhao Dong (unknown), Ruopeng Huang (unknown), Yue Zhao (unknown), Mengyuan Li (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11912](https://arxiv.org/abs/2511.11912)
- **Reason:** 系统研究图基础模型的模型提取攻击，属于大模型安全与对齐的LLM safety方向
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] EARL: Entropy-Aware RL Alignment of LLMs for Reliable RTL Code Generation
- **Authors:** Jiahe Shi (unknown), Zhengqi Gao (unknown), Ching-Yun Ko (unknown), Duane Boning (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12033](https://arxiv.org/abs/2511.12033)
- **Reason:** 提出熵感知的RL对齐框架，提升LLM生成RTL代码的可靠性，属于大模型安全与对齐的alignment方向
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] AlignTree: Efficient Defense Against LLM Jailbreak Attacks
- **Authors:** Gil Goren, Shahar Katz, Lior Wolf
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12217](https://arxiv.org/abs/2511.12217)
- **Reason:** 提出高效的LLM越狱攻击防御方法AlignTree，通过监控模型激活并结合拒绝方向与SVM信号检测错位行为，无需额外提示或辅助模型，提升大模型安全，与大模型安全与对齐方向直接相关
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] The 'Sure' Trap: Multi-Scale Poisoning Analysis of Stealthy Compliance-Only Backdoors in Fine-Tuned Large Language Models
- **Authors:** Yuting Tan, Yi Huang, Zhuo Li
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12414](https://arxiv.org/abs/2511.12414)
- **Reason:** 分析LLM微调中的compliance-only后门攻击，揭示仅通过良性训练数据（如“Sure”响应）即可植入隐式后门的风险，为大模型安全对齐中的数据供应链风险提供警示，与大模型安全与对齐方向相关
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] Incoherent Beliefs & Inconsistent Actions in Large Language Models
- **Authors:** Arka Pal, Teo Kitanovski, Arthur Liang, Akilesh Potti, Micah Goldblum
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13240](https://arxiv.org/abs/2511.13240)
- **Reason:** 研究LLM的信念不一致与行动不一致问题，发现即使准确或校准良好的模型也存在这些问题，属于大模型安全与对齐中的robustness与consistency方向，对理解LLM的实际行为有重要意义。
Score: 8
Field: 大模型安全与对齐

### [Score: 8.0/10] EcoAlign: An Economically Rational Framework for Efficient LVLM Alignment
- **Authors:** Ruoxi Cheng, Haoxuan Ma, Teng Ma, Hongyi Zhang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11301](https://arxiv.org/abs/2511.11301)
- **Reason:** 针对大视觉语言模型（LVLM）的对齐难题，提出将对齐重构为经济理性搜索的EcoAlign框架，解决了安全、效用与计算成本的权衡问题，实验验证其在降低成本的同时提升了安全与效用表现，对大模型安全与对齐研究具有重要参考价值。
Score: 8
Field: 大模型安全与对齐

### [Score: 7.0/10] What Color Is It? A Text-Interference Multimodal Hallucination Benchmark
- **Authors:** Jinkun Zhao, Lei Huang, Wenjun Wu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13400](https://arxiv.org/abs/2511.13400)
- **Reason:** 构建文本干扰的多模态幻觉基准，用于评估MLLM的视觉感知鲁棒性，属于大模型安全与对齐中的幻觉问题研究，有助于揭示MLLM的安全隐患
Score: 7
Field: 大模型安全与对齐

### [Score: 7.0/10] Unlocking the Forgery Detection Potential of Vanilla MLLMs: A Novel Training-Free Pipeline
- **Authors:** Rui Zuo, Qinyue Tong, Zhe-Ming Lu, Ziqian Lu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13442](https://arxiv.org/abs/2511.13442)
- **Reason:** 提出训练-free的MLLM伪造检测 pipeline，利用MLLM的固有泛化能力，属于大模型安全与对齐中的内容伪造检测，提升了MLLM在安全领域的实用性
Score: 7
Field: 大模型安全与对齐

### [Score: 7.0/10] Language-Guided Invariance Probing of Vision-Language Models
- **Authors:** Jae Joong Lee
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13494](https://arxiv.org/abs/2511.13494)
- **Reason:** 提出LGIP基准评估VLMs对语言扰动的鲁棒性，属于大模型安全与对齐中的语言鲁棒性研究，揭示了VLMs在语言处理中的潜在漏洞
Score: 7
Field: 大模型安全与对齐

### [Score: 7.0/10] Robust Defense Strategies for Multimodal Contrastive Learning: Efficient Fine-tuning Against Backdoor Attacks
- **Authors:** Md. Iqbal Hossain, Afia Sajeeda, Neeresh Kumar Perla, Ming Shao
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13545](https://arxiv.org/abs/2511.13545)
- **Reason:** 提出针对多模态对比学习模型的后门攻击防御策略，属于大模型安全与对齐中的后门防御，提升了多模态模型的鲁棒性
Score: 7
Field: 大模型安全与对齐

### [Score: 7.0/10] The Good, The Bad, and The Hybrid: A Reward Structure Showdown in Reasoning Models Training
- **Authors:** Subramanyam Sahoo
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13016](https://arxiv.org/abs/2511.13016)
- **Reason:** 针对推理模型训练的奖励结构设计，比较硬奖励、连续奖励与混合奖励，提出自适应混合奖励调度器，属于大模型安全与对齐中的RLHF与奖励建模方向，对提升推理模型的收敛稳定性有帮助。
Score: 7
Field: 大模型安全与对齐

### [Score: 7.0/10] Synthetic Forgetting without Access: A Few-shot Zero-glance Framework for Machine Unlearning
- **Authors:** Qipeng Song, Nan Yang, Ziqi Xu, Yue Li, Wei Shao, Feng Xia
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13116](https://arxiv.org/abs/2511.13116)
- **Reason:** 针对少样本零glance场景的机器遗忘，提出GFOES框架生成最优擦除样本，属于大模型安全与对齐中的隐私保护方向，对数据受限下的模型遗忘有实用价值。
Score: 7
Field: 大模型安全与对齐

### [Score: 7.0/10] The Second Law of Intelligence: Controlling Ethical Entropy in Autonomous Systems
- **Authors:** Samih Fadli (Unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.10704](https://arxiv.org/abs/2511.10704)
- **Reason:** 提出人工智能第二定律，将AI对齐视为热力学控制问题，定量研究伦理熵控制，属于大模型安全与对齐方向。
Score: 7
Field: 大模型安全与对齐

### [Score: 7.0/10] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping
- **Authors:** Dena Mujtaba, Brian Hu, Anthony Hoogs, Arslan Basharat
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11551](https://arxiv.org/abs/2511.11551)
- **Reason:** 针对预训练Agent的伦理对齐挑战，提出测试时策略塑造方法，无需重新训练即可调整Agent行为以符合伦理准则，在MACHIAVELLI基准上验证了有效性，对大模型安全与对齐的实践应用有积极意义。
Score: 7
Field: 大模型安全与对齐

<h2 id='原生多模态大模型'>原生多模态大模型</h2>

### [Score: 9.0/10] Seeing the Forest and the Trees: Query-Aware Tokenizer for Long-Video Multimodal Language Models
- **Authors:** Siyou Li, Huanan Wu, Juexi Shao, Yinghao Ma, Yujian Gan, Yihao Luo, Yuwei Wang, Dong Nie, Lu Wang, Wengqing Wu, Le Zhang, Massimo Poesio, Juntao Yu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11910](https://arxiv.org/abs/2511.11910)
- **Reason:** 提出Query-aware Token Selector（QTSplus）优化长视频多模态大模型的视觉token处理，实现89%的token压缩与28%的延迟降低，同时保持长视频理解性能，属于原生多模态大模型中的长视频处理与tokenizer方向。
Score: 9
Field: 原生多模态大模型

### [Score: 9.0/10] Part-X-MLLM: Part-aware 3D Multimodal Large Language Model
- **Authors:** Chunshi Wang, Junliang Ye, Yunhan Yang, Yang Li, Zizhuo Lin, Jun Zhu, Zhuo Chen, Yawei Luo, Chunchao Guo
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13647](https://arxiv.org/abs/2511.13647)
- **Reason:** 提出Part-X-MLLM，原生3D多模态大模型，将3D任务统一为结构化程序，支持部分级3D理解与编辑，属于原生多模态大模型的前沿研究
Score: 9
Field: 原生多模态大模型

### [Score: 9.0/10] PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image
- **Authors:** Ziang Cao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, Ziwei Liu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13648](https://arxiv.org/abs/2511.13648)
- **Reason:** 提出PhysX-Anything，从单图生成模拟就绪的物理3D资产，支持接触-rich机器人政策学习，属于原生多模态大模型中的3D生成与理解，填补了物理属性缺失的空白
Score: 9
Field: 原生多模态大模型

### [Score: 9.0/10] Scaling Spatial Intelligence with Multimodal Foundation Models
- **Authors:** Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, Chen Wei, Qingping Sun, Tongxi Zhou, Jiaqi Li, Hui En Pang, Oscar Qian, Yukun Wei, Zhiqian Lin, Xuanke Shi, Kewang Deng, Xiaoyang Han, Zukai Chen, Xiangyu Fan, Hanming Deng, Lewei Lu, Liang Pan, Bo Li, Ziwei Liu, Quan Wang, Dahua Lin, Lei Yang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13719](https://arxiv.org/abs/2511.13719)
- **Reason:** 提出SenseNova-SI家族，多模态基础模型的空间智能，在15/24任务中取得最佳嵌入性能，19/29任务中最佳微调性能，属于原生多模态大模型的空间理解前沿
Score: 9
Field: 原生多模态大模型

### [Score: 8.0/10] Target-Balanced Score Distillation
- **Authors:** Zhou Xu, Qi Wang, Yuxiao Yang, Luyuan Zhang, Zhang Liang, Yang Li
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11710](https://arxiv.org/abs/2511.11710)
- **Reason:** 针对Score Distillation Sampling（SDS）的过饱和与过平滑问题，提出目标平衡的分数蒸馏方法，提升3D资产生成的纹理真实感与形状准确性，属于原生多模态大模型中的图像生成方向。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Image-POSER: Reflective RL for Multi-Expert Image Generation and Editing
- **Authors:** Hossein Mohebbi, Mohammed Abdulrahman, Yanting Miao, Pascal Poupart, Suraj Kothawade
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11780](https://arxiv.org/abs/2511.11780)
- **Reason:** 提出反射式强化学习框架协调多专家文本-图像/图像-图像模型，解决长组合prompt的图像生成与编辑问题，属于原生多模态大模型中的图像生成与编辑方向。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] BeyondFacial: Identity-Preserving Personalized Generation Beyond Facial Close-ups
- **Authors:** Songsong Zhang, Chuanqi Tang, Hongguang Zhang, Guijian Tang, Minglong Li, Xueqiong Li, Shaowu Yang, Yuanxi Peng, Wenjing Yang, Jing Zhao
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11989](https://arxiv.org/abs/2511.11989)
- **Reason:** 提出双路径推理与身份自适应融合策略，突破面部特写限制实现身份保持的场景生成，提升语义一致性与视觉叙事性，属于原生多模态大模型中的个性化图像生成方向。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] LIHE: Linguistic Instance-Split Hyperbolic-Euclidean Framework for Generalized Weakly-Supervised Referring Expression Comprehension
- **Authors:** Xianglong Shi, Silin Cheng, Sirui Zhao, Yunhan Jiang, Enhong Chen, Yang Liu, Sebastien Ourselin
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12020](https://arxiv.org/abs/2511.12020)
- **Reason:** 提出弱监督广义指代表达理解框架LIHE，解决零或多目标场景下的指代表达接地问题，结合双曲-欧几里得空间提升语义区分度，属于原生多模态大模型中的指代表达理解方向。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Improved Masked Image Generation with Knowledge-Augmented Token Representations
- **Authors:** Guotao Liang, Baoquan Zhang, Zhiyuan Wen, Zihao Han, Yunming Ye
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12032](https://arxiv.org/abs/2511.12032)
- **Reason:** 提出知识增强的掩码图像生成框架KA-MIG，利用token级语义依赖知识图提升模型的语义捕捉能力，改进生成质量，属于原生多模态大模型中的掩码图像生成方向。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Calibrated Multimodal Representation Learning with Missing Modalities
- **Authors:** Xiaohao Liu, Xiaobo Xia, Jiaheng Wei, Shuo Yang, Xiu Su, See-Kiong Ng, Tat-Seng Chua
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12034](https://arxiv.org/abs/2511.12034)
- **Reason:** 针对缺失模态的多模态表示学习问题，提出校准框架CalMRL解决锚点偏移问题，提升表示一致性与下游任务性能，属于原生多模态大模型中的缺失模态处理方向。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] SRSplat: Feed-Forward Super-Resolution Gaussian Splatting from Sparse Multi-View Images
- **Authors:** Xinyuan Hu, Changyue Shi, Chuxiao Yang, Minghao Chen, Jiajun Ding, Tao Wei, Chen Wei, Zhou Yu, Min Tan
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12040](https://arxiv.org/abs/2511.12040)
- **Reason:** 提出前馈高分辨率3D重建框架SRSplat，利用多模态大模型（MLLMs）与扩散模型生成的参考图像提升稀疏多视图输入的纹理细节，属于原生多模态大模型中的3D重建方向。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Learning to Hear by Seeing: It's Time for Vision Language Models to Understand Artistic Emotion from Sight and Sound
- **Authors:** Dengming Zhang, Weitao You, Jingxiong Li, Weishen Lin, Wenda Shi, Xue Zhao, Heda Zuo, Junxian Wu, Lingyun Sun
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12077](https://arxiv.org/abs/2511.12077)
- **Reason:** 提出视觉锚定的音视频情感LLM（VAEmotionLLM），通过视觉引导音频对齐实现多模态艺术情感理解，在ArtEmoBenchmark上取得最优性能，属于原生多模态大模型中的音视频情感理解方向。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering
- **Authors:** Seokwon Song, Minsu Park, Gunhee Kim
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12142](https://arxiv.org/abs/2511.12142)
- **Reason:** 构建首个多模态源归因基准MAVIS，包含157K视觉QA实例和事实级引用注释，评估MLLM的多模态证据检索、长文本生成和归因能力，揭示现有模型在图像文档groundedness上的差距，为原生多模态大模型的可靠性评估提供了关键工具。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Mixture of States: Routing Token-Level Dynamics for Multimodal Generation
- **Authors:** Haozhe Liu, Ding Liu, Mingchen Zhuge, Zijian Zhou, Tian Xie, Sen He, Yukang Yang, Shuming Liu, Yuren Cong, Jiadong Guo, Hongyu Xu, Ke Xu, Kam-Woh Ng, Juan C. P\'erez, Juan-Manuel~P\'erez-R\'ua, Tao Xiang, Wei Liu, Shikun Liu, J\"urgen Schmidhuber
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12207](https://arxiv.org/abs/2511.12207)
- **Reason:** 提出MoS融合范式，通过token级可学习路由器融合多模态隐状态，实现多模态扩散模型的灵活交互，实验验证在文本到图像生成和编辑任务上优于大参数模型，属于原生多模态大模型的架构创新。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Reasoning Text-to-Video Retrieval via Digital Twin Video Representations and Large Language Models
- **Authors:** Yiqing Shen, Chenxiao Fan, Chenjia Li, Mathias Unberath
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12371](https://arxiv.org/abs/2511.12371)
- **Reason:** 提出基于数字孪生视频表示的文本-视频检索框架，结合大语言模型实现隐式查询推理，解决多模态融合中的语义对齐问题，属于原生多模态大模型的关键改进。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] DenseAnnotate: Enabling Scalable Dense Caption Collection for Images and 3D Scenes via Spoken Descriptions
- **Authors:** Xiaoyu Lin, Aniket Ghorpade, Hansheng Zhu, Justin Qiu, Dea Rrozhani, Monica Lama, Mick Yang, Zixuan Bian, Ruohan Ren, Alan B. Hong, Jiatao Gu, Chris Callison-Burch
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12452](https://arxiv.org/abs/2511.12452)
- **Reason:** 构建音频驱动的多语言多模态密集标注数据集，支持多模态大模型的细粒度语义学习，属于原生多模态大模型的基础数据支撑。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] TempoMaster: Efficient Long Video Generation via Next-Frame-Rate Prediction
- **Authors:** Yukuo Ma, Cong Liu, Junke Wang, Junqi Liu, Haibin Huang, Zuxuan Wu, Chi Zhang, Xuelong Li
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12578](https://arxiv.org/abs/2511.12578)
- **Reason:** 提出帧率预测的长视频生成框架，解决多模态视频生成中的 temporal coherence问题，属于原生多模态大模型的视频生成优化。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Direct Visual Grounding by Directing Attention of Visual Tokens
- **Authors:** Parsa Esmaeilkhani, Longin Jan Latecki
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12738](https://arxiv.org/abs/2511.12738)
- **Reason:** 针对VLMs中视觉token注意力不足导致的视觉任务性能问题，提出KLAL损失直接监督视觉token注意力分布，引导答案语言token关注相关视觉token，提升几何任务、指向和指代理解等性能，解决了VLMs视觉grounding核心问题。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Explore How to Inject Beneficial Noise in MLLMs
- **Authors:** Ruishu Zhu, Sida Huang, Ziheng Jiao, Hongyuan Zhang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12917](https://arxiv.org/abs/2511.12917)
- **Reason:** 针对MLLMs的跨模态异质性问题，提出MuNG有益噪声注入策略，通过动态分析跨模态关系生成任务自适应噪声，提升跨模态对齐和下游性能，仅需微调1-2%参数，高效解决MLLMs核心对齐问题。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] ViSS-R1: Self-Supervised Reinforcement Video Reasoning
- **Authors:** Bo Fang (), Yuxin Song (), Qiangqiang Wu (), Haoyuan Sun (), Wenhao Wu (), Antoni B. Chan ()
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13054](https://arxiv.org/abs/2511.13054)
- **Reason:** 针对多模态大模型视频推理中视觉信息利用不足的问题，提出自监督强化学习框架ViSS-R1，强制模型处理变换后的视觉输入，提升视频推理的鲁棒性和准确性。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] MMD-Thinker: Adaptive Multi-Dimensional Thinking for Multimodal Misinformation Detection
- **Authors:** Junjie Wu (), Guohong Fu ()
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13242](https://arxiv.org/abs/2511.13242)
- **Reason:** 针对多模态大模型在虚假信息检测中的推理不足问题，提出两阶段框架MMD-Thinker，结合定制化思维模式和强化学习，构造了MMR数据集，提升了检测性能。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Free-Form Scene Editor: Enabling Multi-Round Object Manipulation like in a 3D Engine
- **Authors:** Xincheng Shuai, Zhenyuan Qin, Henghui Ding, Dacheng Tao
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13713](https://arxiv.org/abs/2511.13713)
- **Reason:** 提出FFSE框架，实现3D-aware图像编辑，支持多轮物理一致的物体操作，属于原生多模态大模型中的场景编辑，提升了编辑的直观性和一致性
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation
- **Authors:** Jiayin Zhu (unknown), Linlin Yang (unknown), Yicong Li (unknown), Angela Yao (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11692](https://arxiv.org/abs/2511.11692)
- **Reason:** 研究text-to-3D生成中动态源锚定方法，提升语义一致性，属于原生多模态大模型的image generation方向
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Simple Vision-Language Math Reasoning via Rendered Text
- **Authors:** Matvey Skripkin (unknown), Elizaveta Goncharova (unknown), Andrey Kuznetsov (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11704](https://arxiv.org/abs/2511.11704)
- **Reason:** 通过渲染文本将LaTeX方程转化为图像，结合思维链提示训练视觉语言模型解决数学推理，属于原生多模态大模型的vision-language方向
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Selecting Fine-Tuning Examples by Quizzing VLMs
- **Authors:** Tenghao Ji (unknown), Eytan Adar (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12002](https://arxiv.org/abs/2511.12002)
- **Reason:** 通过测试VLM选择微调示例，提升文本到图像模型的对齐和真实感，属于原生多模态大模型的image generation方向
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] To Align or Not to Align: Strategic Multimodal Representation Alignment for Optimal Performance
- **Authors:** Wanlong Fang, Tianle Zhang, Alvin Chan
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12121](https://arxiv.org/abs/2511.12121)
- **Reason:** 研究多模态学习中explicit representation alignment的策略，探讨其对模型性能的影响，揭示最优对齐强度与模态间冗余度的关系，为多模态表示整合提供理论指导，与原生多模态大模型方向高度相关
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] Uncovering and Mitigating Transient Blindness in Multimodal Model Editing
- **Authors:** Xiaoqi Han, Ru Li, Ran Yi, Hongye Tan, Zhuomin Liang, Víctor Gutiérrez-Basulto, Jeff Z. Pan
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13243](https://arxiv.org/abs/2511.13243)
- **Reason:** 提出多模态模型编辑中的“瞬时失明”现象，设计locality-aware对抗损失缓解该问题，属于原生多模态大模型中的模型编辑方向，对提升多模态模型的编辑鲁棒性有帮助。
Score: 8
Field: 原生多模态大模型

### [Score: 8.0/10] ZeroDexGrasp: Zero-Shot Task-Oriented Dexterous Grasp Synthesis with Prompt-Based Multi-Stage Semantic Reasoning
- **Authors:** Juntao Jian, Yi-Lin Wei, Chengjie Mou, Yuhao Lin, Xing Zhu, Yujun Shen, Wei-Shi Zheng, Ruizhen Hu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13327](https://arxiv.org/abs/2511.13327)
- **Reason:** 整合多模态大语言模型，通过prompt语义推理生成任务对齐的灵巧抓取姿态，实现零样本跨物体类别抓取，属于原生多模态大模型方向的关键应用。
Score: 8
Field: 原生多模态大模型

### [Score: 7.0/10] Point Cloud Quantization through Multimodal Prompting for 3D Understanding
- **Authors:** Hongxuan Li (College of Intelligence and Computing, Tianjin University), Wencheng Zhu (College of Intelligence and Computing, Tianjin University, Haihe Laboratory of Information Technology Application Innovation), Huiying Xu (School of Computer Science and Technology, Zhejiang Normal University), Xinzhong Zhu (School of Computer Science and Technology, Zhejiang Normal University), Pengfei Zhu (College of Intelligence and Computing, Tianjin University)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12079](https://arxiv.org/abs/2511.12079)
- **Reason:** 提出多模态提示驱动的点云量化框架，利用预训练文本嵌入作为原型先验，结合双约束量化空间和Gumbel-Softmax离散化，解决现有点云量化的表示性和可解释性问题，实验验证在3D理解任务上的有效性，属于原生多模态大模型中3D模态融合的关键进展。
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description
- **Authors:** Quanxing Xu, Ling Zhou, Feifei Zhang, Jinyu Tian, Rubing Huang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12131](https://arxiv.org/abs/2511.12131)
- **Reason:** 提出OAD-Promoter框架，通过目标属性描述生成、记忆知识辅助和OAD提示优化，解决LLM在VQA任务中的语言偏见和OOD泛化问题，实验验证在零样本和少样本设置下优于现有方法，属于原生多模态大模型中视觉-语言对齐的关键优化。
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] MOON2.0: Dynamic Modality-balanced Multimodal Representation Learning for E-commerce Product Understanding
- **Authors:** Zhanheng Nie, Chenghan Fu, Daoze Zhang, Junxian Wu, Wanxian Guan, Pengjie Wang, Jian Xu, Bo Zheng
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12449](https://arxiv.org/abs/2511.12449)
- **Reason:** 提出动态模态平衡的多模态表示学习框架，解决电商场景下的模态不平衡与数据噪声问题，属于原生多模态大模型在垂直领域的落地优化。
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] Video Finetuning Improves Reasoning Between Frames
- **Authors:** Ruiqi Yang, Tian Yun, Zihan Wang, Ellie Pavlick
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12868](https://arxiv.org/abs/2511.12868)
- **Reason:** 研究视频微调对多模态LLMs帧间推理的影响，提出vCoT验证视频模型隐含捕捉帧间过渡的能力，提升长视频问答和静态视觉推理性能，属于多模态LLM的训练优化研究。
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] DeepSport: A Multimodal Large Language Model for Comprehensive Sports Video Reasoning via Agentic Reinforcement Learning
- **Authors:** Junbo Zou, Haotian Xia, Zhen Ye, Shengjie Zhang, Christopher Lai, Vicente Ordonez, Weining Shen, Hanjie Chen
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12908](https://arxiv.org/abs/2511.12908)
- **Reason:** 提出针对体育视频推理的MLLM DeepSport，通过强化学习优化主动推理过程，解决体育视频的高动态、复杂规则和长时序推理问题，属于多模态大模型的领域特定应用。
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework
- **Authors:** Diego Ortego (), Marlon Rodr\'iguez (), Mario Almagro (), Kunal Dahiya (), David Jim\'enez (), Juan C. SanMiguel ()
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13189](https://arxiv.org/abs/2511.13189)
- **Reason:** 提出ViXML多模态框架，整合大语言模型和视觉模型处理极端多标签分类，解决了大模型在多模态任务中的应用问题，提升了分类性能。
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] TabFlash: Efficient Table Understanding with Progressive Question Conditioning and Token Focusing
- **Authors:** Jongha Kim, Minseong Bae, Sanghyeok Lee, Jinsung Yoon, Hyunwoo J. Kim
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13283](https://arxiv.org/abs/2511.13283)
- **Reason:** 提出高效的多模态大模型TabFlash用于表格理解，结合渐进式问题条件注入和token聚焦策略，提升性能的同时降低计算成本（减少27% FLOPs和30%内存使用），与原生多模态大模型研究方向高度相关
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] Semantic Document Derendering: SVG Reconstruction via Vision-Language Modeling
- **Authors:** Adam Hazimeh, Ke Wang, Mark Collier, Gilles Baechler, Efi Kokiopoulou, Pascal Frossard
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13478](https://arxiv.org/abs/2511.13478)
- **Reason:** 提出SliDer框架，利用VLMs将光栅文档转换为可编辑的SVG，属于原生多模态大模型中的文档理解与生成，解决了文档编辑性问题
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] Training-Free Multi-View Extension of IC-Light for Textual Position-Aware Scene Relighting
- **Authors:** Jiangnan Ye, Jiedong Zhuang, Lianrui Mu, Wenjie Zheng, Jiaqi Hu, Xingze Zou, Jing Wang, Haoji Hu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13684](https://arxiv.org/abs/2511.13684)
- **Reason:** 提出GS-Light，文本引导的3D场景重照明，利用LVLM解析光照先验，生成多视图一致的重照明结果，属于原生多模态大模型中的场景编辑
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] Small Vocabularies, Big Gains: Pretraining and Tokenization in Time Series Models
- **Authors:** Alexis Roger, Gwen Legate, Kashif Rasul, Yuriy Nevmyvaka, Irina Rish
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11622](https://arxiv.org/abs/2511.11622)
- **Reason:** 研究时间序列的tokenization设计与预训练的结合，证明小词汇量tokenizer的有效性，属于原生多模态大模型中的tokenizer关键方向。
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] Doubly Debiased Test-Time Prompt Tuning for Vision-Language Models
- **Authors:** Fei Song, Yi Li, Rui Wang, Jiahuan Zhou, Changwen Zheng, Jiangmeng Li
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11690](https://arxiv.org/abs/2511.11690)
- **Reason:** 提出双去偏测试时prompt tuning提升视觉语言模型泛化能力，属于原生多模态大模型中的prompt优化方向。
Score: 7
Field: 原生多模态大模型

### [Score: 7.0/10] MMSense: Adapting Vision-based Foundation Model for Multi-task Multi-modal Wireless Sensing
- **Authors:** Zhizhen Li, Xuanhao Luo, Xueren Ge, Longyu Zhou, Xingqin Lin, Yuchen Liu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12305](https://arxiv.org/abs/2511.12305)
- **Reason:** 提出MMSense多模态多任务基础模型，整合图像、雷达、LiDAR等模态并进行跨模态对齐，用于无线感知的多任务学习，与原生多模态大模型方向相关
Score: 7
Field: 原生多模态大模型

<h2 id='多模态智能体'>多模态智能体</h2>

### [Score: 9.0/10] GCAgent: Long-Video Understanding via Schematic and Narrative Episodic Memory
- **Authors:** Jeong Hun Yeo, Sangyun Chung, Sungjune Park, Dae Hoe Kim, Jinyoung Moon, Yong Man Ro
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12027](https://arxiv.org/abs/2511.12027)
- **Reason:** 提出全局上下文感知的智能体框架GCAgent，通过结构化的情节记忆解决长视频理解的长期依赖问题，在Video-MME基准上实现23.5%的 accuracy提升，属于多模态智能体中的长视频理解方向。
Score: 9
Field: 多模态智能体

### [Score: 8.0/10] Hi-Reco: High-Fidelity Real-Time Conversational Digital Humans
- **Authors:** Hongbin Huang, Junwei Li, Tianxin Xie, Zhuang Li, Cekai Weng, Yaodong Yang, Yue Luo, Li Liu, Jing Tang, Zhijing Shao, Zeyu Wang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12662](https://arxiv.org/abs/2511.12662)
- **Reason:** 提出实时对话数字人系统，结合多模态交互与知识接地，属于多模态智能体中的数字人应用创新。
Score: 8
Field: 多模态智能体

### [Score: 8.0/10] REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding
- **Authors:** Jiaze Li, Hao Yin, Wenhui Tan, Jingyang Chen, Boshen Xu, Yuxun Qu, Yijing Chen, Jianzhong Ju, Zhenbo Luo, Jian Luan
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13026](https://arxiv.org/abs/2511.13026)
- **Reason:** 针对长视频理解的文本反思不足问题，提出REVISOR多模态反思框架，通过工具增强的多模态反思提升MLLMs的长视频推理能力，设计DADR奖励机制保证推理与视频证据的因果对齐，解决长视频理解核心痛点。
Score: 8
Field: 多模态智能体

### [Score: 8.0/10] Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents
- **Authors:** Yuan Zhao (Unknown), Hualei Zhu (Unknown), Tingyu Jiang (Unknown), Shen Li (Unknown), Xiaohang Xu (Unknown), Hao Henry Wang (Unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.10705](https://arxiv.org/abs/2511.10705)
- **Reason:** 针对GUI代理的规划与接地能力协同进化提出Co-EPG框架，属于多模态智能体中的GUI Agent方向，与用户高优先级方向高度相关。
Score: 8
Field: 多模态智能体

### [Score: 7.0/10] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching
- **Authors:** Zewei Chang, Zheng-Peng Duan, Jianxing Zhang, Chun-Le Guo, Siyu Liu, Hyungju Chun, Hyunhee Park, Zikun Liu, Chongyi Li
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12998](https://arxiv.org/abs/2511.12998)
- **Reason:** 提出VLM驱动的个性化图像修图代理PerTouch，支持语义级控制和用户反馈，解决图像修图的个性化与语义一致性问题，属于多模态智能体的具体应用。
Score: 7
Field: 多模态智能体

### [Score: 7.0/10] PIGEON: VLM-Driven Object Navigation via Points of Interest Selection
- **Authors:** Cheng Peng, Zhenzhe Zhang, Cheng Chi, Xiaobao Wei, Yanhao Zhang, Heng Wang, Pengwei Wang, Zhongyuan Wang, Jing Liu, Shanghang Zhang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13207](https://arxiv.org/abs/2511.13207)
- **Reason:** 提出VLM驱动的物体导航框架，结合视觉-语言模型与兴趣点选择实现未知环境目标导航，提升决策频率与语义指导能力，属于多模态智能体方向的重要研究。
Score: 7
Field: 多模态智能体

<h2 id='深度学习可解释性'>深度学习可解释性</h2>

### [Score: 9.0/10] Did Models Sufficient Learn? Attribution-Guided Training via Subset-Selected Counterfactual Augmentation
- **Authors:** Yannan Chen, Ruoyu Chen, Bin Zeng, Wei Wang, Shiming Liu, Qunli Zhang, Zheng Hu, Laiyuan Wang, Yaowei Wang, Xiaochun Cao
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12100](https://arxiv.org/abs/2511.12100)
- **Reason:** 将LIMA归因方法融入训练流程，提出子集选择反事实增强（SS-CA）策略，通过生成反事实样本纠正模型的非因果依赖，提升模型的泛化能力和鲁棒性，实验验证在ImageNet变体和OOD基准上优于现有方法，直接关联可解释性与模型优化，属于深度学习可解释性的实践创新。
Score: 9
Field: 深度学习可解释性

### [Score: 9.0/10] Explainable AI-Generated Image Detection RewardBench
- **Authors:** Michael Yang, Shijian Deng, William T. Doan, Kai Wang, Tianyu Yang, Harsh Singh, Yapeng Tian
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12363](https://arxiv.org/abs/2511.12363)
- **Reason:** 构建XAIGID-RewardBench基准，包含3000+注释三元组，评估MLLM对AI生成图像检测解释的判断能力，揭示当前模型与人类的差距（最佳模型88.76% vs 人类98.30%），为可解释AI生成图像检测提供了标准化评估工具，属于深度学习可解释性的基准创新。
Score: 9
Field: 深度学习可解释性

### [Score: 9.0/10] HEDGE: Hallucination Estimation via Dense Geometric Entropy for VQA with Vision-Language Models
- **Authors:** Sushant Gautam, Michael A. Riegler, P{\aa}l Halvorsen
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12693](https://arxiv.org/abs/2511.12693)
- **Reason:** 提出几何熵的幻觉估计框架，解决多模态VQA的可靠性评估问题，属于深度学习可解释性中的幻觉检测研究。
Score: 9
Field: 深度学习可解释性

### [Score: 8.0/10] Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models
- **Authors:** Sanchit Sinha, Guangzhi Xiong, Zhenghao He, Aidong Zhang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11751](https://arxiv.org/abs/2511.11751)
- **Reason:** 提出接地的多Agent神经符号推理框架，通过视觉概念提取与规则生成提升VLMs的可解释性并减少幻觉，属于深度学习可解释性中的神经符号方法方向。
Score: 8
Field: 深度学习可解释性

### [Score: 8.0/10] X-VMamba: Explainable Vision Mamba
- **Authors:** Mohamed A. Mabrok, Yalda Zafari
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12694](https://arxiv.org/abs/2511.12694)
- **Reason:** 针对Vision Mamba等SSMs空间信息处理缺乏透明解释机制的问题，提出基于可控性的可解释性框架，包含Jacobian（适用于所有SSM）和Gramian（适用于对角SSM）两种线性复杂度方法，验证了医疗图像等领域的效果，解决了SSMs可解释性痛点。
Score: 8
Field: 深度学习可解释性

### [Score: 8.0/10] Concept Regions Matter: Benchmarking CLIP with a New Cluster-Importance Approach
- **Authors:** Aishwarya Agarwal, Srikrishna Karanam, Vineet Gandhi
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12978](https://arxiv.org/abs/2511.12978)
- **Reason:** 针对CLIP的背景依赖问题，提出CCI可解释性方法，通过CLIP的patch嵌入聚类评估概念区域的重要性，提升可解释性的faithfulness，解决VLMs背景过依赖痛点。
Score: 8
Field: 深度学习可解释性

### [Score: 8.0/10] Rethinking Saliency Maps: A Cognitive Human Aligned Taxonomy and Evaluation Framework for Explanations
- **Authors:** Yehonatan Elisha (), Seffi Cohen (), Oren Barkan (), Noam Koenigstein ()
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13081](https://arxiv.org/abs/2511.13081)
- **Reason:** 针对显著性图解释的核心问题，提出RFxG分类法和新的忠实性指标，解决现有评估忽略对比推理和语义粒度的局限，对深度学习可解释性的理论和实践有重要价值。
Score: 8
Field: 深度学习可解释性

### [Score: 8.0/10] Accuracy is Not Enough: Poisoning Interpretability in Federated Learning via Color Skew
- **Authors:** Farhin Farhad Riya, Shahinul Hoque, Jinyuan Stella Sun, Olivera Kotevska
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13535](https://arxiv.org/abs/2511.13535)
- **Reason:** 揭示联邦学习中通过颜色偏移攻击模型可解释性的问题，属于深度学习可解释性中的攻击与鲁棒性研究，强调了可解释性本身的安全风险
Score: 8
Field: 深度学习可解释性

### [Score: 8.0/10] Which Sparse Autoencoder Features Are Real? Model-X Knockoffs for False Discovery Rate Control
- **Authors:** Tsogt-Ochir Enkhbayar (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11711](https://arxiv.org/abs/2511.11711)
- **Reason:** 将Model-X knockoffs应用于稀疏自编码器特征选择，控制假阳性率，属于深度学习可解释性的特征解释方向
Score: 8
Field: 深度学习可解释性

### [Score: 8.0/10] FLEX: Feature Importance from Layered Counterfactual Explanations
- **Authors:** Nawid Keshtmand (unknown), Roussel Desmond Nzoyem (unknown), Jeffrey Nicholas Clark (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11891](https://arxiv.org/abs/2511.11891)
- **Reason:** 将counterfactuals转化为局部、区域和全局特征重要性，属于深度学习可解释性的feature importance方向
Score: 8
Field: 深度学习可解释性

### [Score: 8.0/10] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification
- **Authors:** Alan G. Paredes Cetina, Kaouther Benguessoum, Raoni Lourenço, Sylvain Kubler
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13237](https://arxiv.org/abs/2511.13237)
- **Reason:** 提出CONFETTI方法生成多变量时间序列的反事实解释，平衡准确性、proximity与sparsity，属于深度学习可解释性中的反事实解释方向，对时间序列模型的可解释性与决策支持有帮助。
Score: 8
Field: 深度学习可解释性

### [Score: 7.0/10] SAGE: Saliency-Guided Contrastive Embeddings
- **Authors:** Colton R. Crum, Adam Czajka
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12744](https://arxiv.org/abs/2511.12744)
- **Reason:** 针对现有显著性引导训练依赖图像空间机制的问题，提出SAGE框架，将人类显著性整合到潜空间的对比嵌入训练中，通过对比损失引导模型关注显著特征，提升分类性能和泛化性，属于可解释性中的人类先验整合方法。
Score: 7
Field: 深度学习可解释性

### [Score: 7.0/10] Semantic Prioritization in Visual Counterfactual Explanations with Weighted Segmentation and Auto-Adaptive Region Selection
- **Authors:** Lintong Zhang, Kang Yin, Seong-Whan Lee
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12992](https://arxiv.org/abs/2511.12992)
- **Reason:** 针对视觉反事实解释的语义无关问题，提出WSAE-Net框架，通过加权语义图和自适应编辑序列提升解释的语义相关性，改善反事实解释质量，属于深度学习可解释性的反事实解释改进。
Score: 7
Field: 深度学习可解释性

### [Score: 7.0/10] WildfireGenome: Interpretable Machine Learning Reveals Local Drivers of Wildfire Risk and Their Cross-County Variation
- **Authors:** Chenyue Liu, Ali Mostafavi
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11589](https://arxiv.org/abs/2511.11589)
- **Reason:** 用SHAP和ICE/PDP解释野火风险模型的局部驱动因素，连接模型输出与可理解的地理因素，属于深度学习可解释性的实际应用。
Score: 7
Field: 深度学习可解释性

### [Score: 7.0/10] Sound Logical Explanations for Mean Aggregation Graph Neural Networks
- **Authors:** Matthew Morris, Ian Horrocks
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11593](https://arxiv.org/abs/2511.11593)
- **Reason:** 为均值聚合GNN提供可靠的逻辑解释，证明其可表示的单调规则类，提升GNN的白盒解释能力，属于深度学习可解释性中的GNN方向。
Score: 7
Field: 深度学习可解释性

### [Score: 7.0/10] Evaluation of LLM-based Explanations for a Learning Analytics Dashboard
- **Authors:** Alina Deriyeva, Benjamin Paassen
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11671](https://arxiv.org/abs/2511.11671)
- **Reason:** 评估LLM对学习分析仪表板的解释效果，提升教育领域LLM应用的可解释性，属于深度学习可解释性中的LLM解释方向。
Score: 7
Field: 深度学习可解释性

### [Score: 7.0/10] Beyond saliency: enhancing explanation of speech emotion recognition with expert-referenced acoustic cues
- **Authors:** Seham Nasr, Zhao Ren, David Johnson
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11691](https://arxiv.org/abs/2511.11691)
- **Reason:** 将显著性与专家 acoustic 线索结合提升语音情感识别模型解释性，连接模型输出与领域知识，属于深度学习可解释性中的语音模型方向。
Score: 7
Field: 深度学习可解释性

### [Score: 7.0/10] Explainable RL Policies by Distilling to Locally-Specialized Linear Policies with Voronoi State Partitioning
- **Authors:** Senne Deproost, Dennis Steckelmacher, Ann Nowé
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13322](https://arxiv.org/abs/2511.13322)
- **Reason:** 提出将RL政策蒸馏为局部线性政策，用Voronoi划分提升可解释性，属于深度学习可解释性中的RL政策解释方向，对提升RL模型的透明度与 trust 有帮助。
Score: 7
Field: 深度学习可解释性

### [Score: 7.0/10] Weight-sparse transformers have interpretable circuits
- **Authors:** Leo Gao (Unknown), Achyuta Rajaram (Unknown), Jacob Coxon (Unknown), Soham V. Govande (Unknown), Bowen Baker (Unknown), Dan Mossing (Unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13653](https://arxiv.org/abs/2511.13653)
- **Reason:** 通过权重稀疏化使Transformer具有可解释的电路，探索模型的机械可解释性，属于深度学习可解释性方向，符合用户研究重点。
Score: 7
Field: 深度学习可解释性

<h2 id='大模型新技术'>大模型新技术</h2>

### [Score: 9.0/10] Back to Basics: Let Denoising Generative Models Denoise
- **Authors:** Tianhong Li, Kaiming He
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13720](https://arxiv.org/abs/2511.13720)
- **Reason:** 提出JiT，基于Transformer的扩散模型直接预测干净数据，无需tokenizer、预训练或额外损失，在ImageNet上取得 competitive 结果，属于大模型新技术中的扩散模型基础研究
Score: 9
Field: 大模型新技术

### [Score: 9.0/10] Reasoning: From Reflection to Solution
- **Authors:** Zixi Li (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11712](https://arxiv.org/abs/2511.11712)
- **Reason:** 提出推理是状态空间迭代算子应用的定义，设计OpenLM架构解决OpenXOR问题，属于大模型新技术的推理方向
Score: 9
Field: 大模型新技术

### [Score: 9.0/10] Diffusion Models: A Mathematical Introduction
- **Authors:** Sepehr Maleki (unknown), Negar Pourmoazemi (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11746](https://arxiv.org/abs/2511.11746)
- **Reason:** 系统推导扩散生成模型的数学基础，包括正向过程、反向posterior和变分边界，属于大模型新技术的diffusion理论方向
Score: 9
Field: 大模型新技术

### [Score: 8.0/10] Multivariate Diffusion Transformer with Decoupled Attention for High-Fidelity Mask-Text Collaborative Facial Generation
- **Authors:** Yushe Cao, Dianxi Shi, Xing Fu, Xuechao Zou, Haikuo Peng, Xueqi Li, Chun Yu, Junliang Xing
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12631](https://arxiv.org/abs/2511.12631)
- **Reason:** 提出多变量扩散Transformer，解决掩码-文本协同面部生成中的模态冲突问题，属于大模型新技术中的扩散模型架构创新。
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] DensePercept-NCSSD: Vision Mamba towards Real-time Dense Visual Perception with Non-Causal State Space Duality
- **Authors:** Tushar Anand, Advik Sinha, Abhijit Das
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12671](https://arxiv.org/abs/2511.12671)
- **Reason:** 提出Vision Mamba模型，解决密集视觉感知的实时性问题，属于大模型新技术中的Mamba架构应用。
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] Recurrent Autoregressive Diffusion: Global Memory Meets Local Attention
- **Authors:** Taiye Chen, Zihan Ding, Anjian Li, Christina Zhang, Zeqi Xiao, Yisen Wang, Chi Jin
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12940](https://arxiv.org/abs/2511.12940)
- **Reason:** 针对扩散模型长视频生成的遗忘和时空不一致问题，将RNN引入扩散Transformer，提出递归自回归扩散框架，结合全局记忆与局部注意力，提升长视频生成的一致性和记忆能力，属于diffusion模型长序列生成新技术。
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] MeanFlow Transformers with Representation Autoencoders
- **Authors:** Zheyuan Hu, Chieh-Hsin Lai, Ge Wu, Yuki Mitsufuji, Stefano Ermon
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13019](https://arxiv.org/abs/2511.13019)
- **Reason:** 针对MeanFlow生成模型的训练不稳定和计算成本问题，提出MF-RAE框架，利用Representation Autoencoder的语义丰富潜空间提升训练稳定性，减少训练与采样计算量，实现高效生成，属于MeanFlow重要改进技术。
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] Generalized Denoising Diffusion Codebook Models (gDDCM): Tokenizing images using a pre-trained diffusion model
- **Authors:** Fei Kong
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13387](https://arxiv.org/abs/2511.13387)
- **Reason:** 将DDCM扩展到主流扩散模型（DDPM、Score-Based Models等），实现图像token化，属于大模型新技术中的扩散模型扩展研究，提升了扩散模型的通用性和性能
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] Distribution Matching Distillation Meets Reinforcement Learning
- **Authors:** Dengyang Jiang, Dongyang Liu, Zanyi Wang, Qilong Wu, Xin Jin, David Liu, Zhen Li, Mengmeng Wang, Peng Gao, Harry Yang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13649](https://arxiv.org/abs/2511.13649)
- **Reason:** 将强化学习引入扩散模型蒸馏，提升少步生成性能（超过教师模型），属于大模型新技术中的扩散模型优化，创新了蒸馏范式
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] UnSAMv2: Self-Supervised Learning Enables Segment Anything at Any Granularity
- **Authors:** Junwei Yu, Trevor Darrell, XuDong Wang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13714](https://arxiv.org/abs/2511.13714)
- **Reason:** 提出UnSAMv2，自监督提升SAM的粒度控制，支持任意粒度的分割，仅用6K无标签图像和0.02%额外参数增强SAM-2，属于大模型新技术中的分割模型优化
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling
- **Authors:** Aihua Zhu, Rui Su, Qinglin Zhao, Li Feng, Meng Shen, Shibo He
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11688](https://arxiv.org/abs/2511.11688)
- **Reason:** 提出HSO框架优化扩散模型采样调度，实现快速鲁棒的采样，属于大模型新技术中的扩散模型方向。
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] Optimizing Input of Denoising Score Matching is Biased Towards Higher Score Norm
- **Authors:** Tongda Xu (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11727](https://arxiv.org/abs/2511.11727)
- **Reason:** 研究去噪分数匹配的输入优化偏差，涉及diffusion模型核心技术，属于大模型新技术的diffusion LLM方向
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] Better LLM Reasoning via Dual-Play
- **Authors:** Zhengxin Zhang (unknown), Chengyu Huang (unknown), Aochong Oliver Li (unknown), Claire Cardie (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11881](https://arxiv.org/abs/2511.11881)
- **Reason:** 提出双玩框架提升LLM推理能力，属于大模型新技术的推理方向
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] Chain-of-Generation: Progressive Latent Diffusion for Text-Guided Molecular Design
- **Authors:** Lingxiao Li (unknown), Haobo Zhang (unknown), Bin Chen (unknown), Jiayu Zhou (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11894](https://arxiv.org/abs/2511.11894)
- **Reason:** 提出渐进式潜在扩散模型用于文本引导分子设计，属于大模型新技术的diffusion LLM方向
Score: 8
Field: 大模型新技术

### [Score: 8.0/10] Diffusion Model Based Signal Recovery Under 1-Bit Quantization
- **Authors:** Youming Chen, Zhaoqiang Liu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12471](https://arxiv.org/abs/2511.12471)
- **Reason:** 提出Diff-OneBit方法，将扩散模型作为先验用于1位量化的信号恢复（如压缩感知、逻辑回归），通过可微分替代似然函数解决非可微问题，拓展扩散模型的应用场景，与大模型新技术方向相关
Score: 8
Field: 大模型新技术

### [Score: 7.0/10] From Events to Clarity: The Event-Guided Diffusion Framework for Dehazing
- **Authors:** Ling Wang, Yunfan Lu, Wenzong Ma, Huizai Yao, Pengteng Li, Hui Xiong
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11944](https://arxiv.org/abs/2511.11944)
- **Reason:** 提出事件引导的扩散去雾框架，利用事件相机的高动态范围（HDR）信息改进去雾效果，属于大模型新技术中的扩散模型应用方向。
Score: 7
Field: 大模型新技术

### [Score: 7.0/10] Null-Space Diffusion Distillation for Efficient Photorealistic Lensless Imaging
- **Authors:** Jose Reinaldo Cunha Santos A V Silva Neto, Hodaka Kawachi, Yasushi Yagi, Tomoya Nakamura
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12024](https://arxiv.org/abs/2511.12024)
- **Reason:** 提出零空间扩散蒸馏方法，实现无监督的无透镜成像真实感重建，解决传统方法的域偏移问题，属于大模型新技术中的扩散模型应用方向。
Score: 7
Field: 大模型新技术

### [Score: 7.0/10] HiGFA: Hierarchical Guidance for Fine-grained Data Augmentation with Diffusion Models
- **Authors:** Zhiguang Lu, Qianqian Xu, Peisong Wen, Siran Da, Qingming Huang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12547](https://arxiv.org/abs/2511.12547)
- **Reason:** 提出分层引导的扩散模型数据增强方法，解决细粒度任务中的样本保真度问题，属于大模型新技术中的扩散模型优化。
Score: 7
Field: 大模型新技术

### [Score: 7.0/10] Which Way from B to A: The role of embedding geometry in image interpolation for Stable Diffusion
- **Authors:** Nicholas Karris, Luke Durell, Javier Flores, Tegan Emerson
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12757](https://arxiv.org/abs/2511.12757)
- **Reason:** 提出将CLIP嵌入视为Wasserstein空间点云的新视角，用最优传输方法解决Stable Diffusion图像插值的平滑性问题，提升插值结果的一致性，属于diffusion模型的嵌入几何分析与应用新技术。
Score: 7
Field: 大模型新技术

### [Score: 7.0/10] Infinite-Story: A Training-Free Consistent Text-to-Image Generation
- **Authors:** Jihun Park, Kyoungmin Lee, Jongmin Gim, Hyeonseo Jo, Minseok Oh, Wonhyeok Choi, Kyumin Hwang, Jaeyeul Kim, Minwoo Choi, Sunghoon Im
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13002](https://arxiv.org/abs/2511.13002)
- **Reason:** 针对文本到图像生成的多prompt一致性问题，提出训练-free的Infinite-Story框架，通过身份提示替换和统一注意力引导提升身份与风格一致性，实现高效故事生成，属于T2I生成的一致性新技术。
Score: 7
Field: 大模型新技术

### [Score: 7.0/10] TiViBench: Benchmarking Think-in-Video Reasoning for Video Generative Models
- **Authors:** Harold Haodong Chen, Disen Lan, Wen-Jie Shu, Qingyang Liu, Zihan Wang, Sirui Chen, Wenkai Cheng, Kanghao Chen, Hongfei Zhang, Zixin Zhang, Rongjin Guo, Yu Cheng, Ying-Cong Chen
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13704](https://arxiv.org/abs/2511.13704)
- **Reason:** 构建TiViBench基准评估视频生成模型的推理能力，涵盖结构、空间、符号和动作规划四大维度，属于大模型新技术中的视频生成推理研究
Score: 7
Field: 大模型新技术

### [Score: 7.0/10] Segment Anything Across Shots: A Method and Benchmark
- **Authors:** Hengrui Hu, Kaining Ying, Henghui Ding
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13715](https://arxiv.org/abs/2511.13715)
- **Reason:** 提出SAAS模型和Cut-VOS基准，解决多镜头视频分割的 shot 不连续问题，属于大模型新技术中的视频分割研究，提升了分割的泛化性
Score: 7
Field: 大模型新技术

### [Score: 7.0/10] EL3DD: Extended Latent 3D Diffusion for Language Conditioned Multitask Manipulation
- **Authors:** Jonas Bode, Raphael Memmesheimer, Sven Behnke
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13312](https://arxiv.org/abs/2511.13312)
- **Reason:** 提出融合视觉与文本输入的3D扩散模型框架，用于生成机器人多任务操作轨迹，属于大模型新技术中的扩散模型应用，提升了多任务操作长程成功率。
Score: 7
Field: 大模型新技术

<h2 id='深度学习理论'>深度学习理论</h2>

### [Score: 9.0/10] Decoupling Positional and Symbolic Attention Behavior in Transformers
- **Authors:** Felipe Urrutia, Jorge Salas, Alexander Kozachinskiy, Cristian Buc Calderon, Hector Pasten, Cristobal Rojas
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11579](https://arxiv.org/abs/2511.11579)
- **Reason:** 分析Transformer中位置与符号注意力的分离，提出行为定义和量化指标，揭示RoPE频率与注意力行为的关联，属于深度学习理论中的网络架构研究
Score: 9
Field: 深度学习理论

### [Score: 9.0/10] Understanding InfoNCE: Transition Probability Matrix Induced Feature Clustering
- **Authors:** Ge Cheng, Shuo Wang, Yun Zhang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12180](https://arxiv.org/abs/2511.12180)
- **Reason:** 分析对比学习核心目标函数InfoNCE的理论基础，提出Transition Probability Matrix诱导特征聚类的机制，改进得到Scaled Convergence InfoNCE（SC-InfoNCE），为对比学习的理论理解与性能优化提供重要支持，与深度学习理论方向高度相关
Score: 9
Field: 深度学习理论

### [Score: 8.0/10] LE-CapsNet: A Light and Enhanced Capsule Network
- **Authors:** Pouya Shiri, Amirali Baniasadi
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11708](https://arxiv.org/abs/2511.11708)
- **Reason:** 提出LE-CapsNet改进Capsule Network的性能与效率，在CIFAR-10和AffNIST数据集上实现更高准确率与更快推理速度，属于深度学习理论中网络架构方向的研究。
Score: 8
Field: 深度学习理论

### [Score: 8.0/10] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding
- **Authors:** Yuan Zhou, Litao Hua, Shilong Jin, Wentao Huang, Haoran Duan
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12530](https://arxiv.org/abs/2511.12530)
- **Reason:** Reinforced Causal Search with Information Bottleneck for Video Understanding
Authors: Yuan Zhou, Litao Hua, Shilong Jin, Wentao Huang, Haoran Duan
Published: 2025-11-18
Link: https://arxiv.org/abs/2511.12530
Reason: 提出强化因果搜索框架，结合信息瓶颈实现视频关键帧选择，解决多模态视频理解中的因果推理问题，属于深度学习理论中的因果表征学习。
Score: 8
Field: 深度学习理论

### [Score: 8.0/10] Seg-VAR: Image Segmentation with Visual Autoregressive Modeling
- **Authors:** Rongkun Zheng, Lu Qi, Xi Chen, Yi Wang, Kun Wang, Hengshuang Zhao
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12594](https://arxiv.org/abs/2511.12594)
- **Reason:** 提出视觉自回归建模的分割框架，将分割转化为条件掩码生成问题，属于深度学习理论中的自回归模型应用创新。
Score: 8
Field: 深度学习理论

### [Score: 8.0/10] Denoising Vision Transformer Autoencoder with Spectral Self-Regularization
- **Authors:** Xunzhi Xiang, Xingye Tian, Guiyu Zhang, Yabo Chen, Shaofeng Zhang, Xuebo Wang, Xin Tao, Qi Fan
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12633](https://arxiv.org/abs/2511.12633)
- **Reason:** 提出光谱自正则化的去噪ViT自动编码器，解决高维 latent空间的噪声问题，属于深度学习理论中的自动编码器优化。
Score: 8
Field: 深度学习理论

### [Score: 8.0/10] Softmax as a Lagrangian-Legendrian Seam
- **Authors:** Christopher R. Lee-Jenkins
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11573](https://arxiv.org/abs/2511.11573)
- **Reason:** 从微分几何角度分析softmax，将其建模为Lagrangian-Legendrian seam，揭示了softmax的几何属性，属于深度学习理论中的模型组件分析
Score: 8
Field: 深度学习理论

### [Score: 8.0/10] Mind Your Entropy: From Maximum Entropy to Trajectory Entropy-Constrained RL
- **Authors:** Guojian Zhan, Likun Wang, Pengcheng Wang, Feihong Zhang, Jingliang Duan, Masayoshi Tomizuka, Shengbo Eben Li
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11592](https://arxiv.org/abs/2511.11592)
- **Reason:** 提出轨迹熵约束的RL框架TECRL，解决最大熵RL的非平稳Q值估计问题，提升RL训练稳定性，属于深度学习理论中的RL优化方向。
Score: 8
Field: 深度学习理论

### [Score: 8.0/10] Coordinate Descent for Network Linearization
- **Authors:** Vlad Rakhlin (unknown), Amir Jevnisek (unknown), Shai Avidan (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11781](https://arxiv.org/abs/2511.11781)
- **Reason:** 使用坐标下降优化网络线性化，涉及优化器和网络架构，属于深度学习理论的optimizer方向
Score: 8
Field: 深度学习理论

### [Score: 8.0/10] Variation-Bounded Loss for Noise-Tolerant Learning
- **Authors:** Jialiang Wang, Xiong Zhou, Xianming Liu, Gangfeng Hu, Deming Zhai, Junjun Jiang, Haoliang Li
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12143](https://arxiv.org/abs/2511.12143)
- **Reason:** 提出Variation-Bounded Loss（VBL）家族，基于Variation Ratio理论分析其鲁棒性，为噪声标签下的监督学习提供新的鲁棒损失函数设计方法，与深度学习理论方向相关
Score: 8
Field: 深度学习理论

### [Score: 8.0/10] Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs
- **Authors:** Jeongwhan Choi, Seungjun Park, Sumin Park, Sung-Bae Cho, Noseong Park
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13010](https://arxiv.org/abs/2511.13010)
- **Reason:** 聚焦图神经网络架构对比，提出分形节点改进MPNN的长距离信息传递，解决Graph Transformers的效率问题，属于深度学习理论中的网络架构研究，对图模型的效率与表现力提升有重要意义。
Score: 8
Field: 深度学习理论

### [Score: 8.0/10] Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression
- **Authors:** Tingkai Yan, Haodong Wen, Binghui Li, Kairong Luo, Wenguang Chen, Kaifeng Lyu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13421](https://arxiv.org/abs/2511.13421)
- **Reason:** 理论分析线性回归的多epoch缩放规律，提出“有效复用率”量化数据重复的价值，属于深度学习理论中的scaling laws研究，对理解有限数据下的模型训练有理论贡献。
Score: 8
Field: 深度学习理论

### [Score: 8.0/10] AdamX: An Adam improvement algorithm based on a novel exponential decay mechanism for the second-order moment estimate
- **Authors:** Meng Zhu (Unknown), Quan Xiao (Unknown), Weidong Min (Unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13465](https://arxiv.org/abs/2511.13465)
- **Reason:** 针对Adam优化器易收敛到非平坦 minima的问题，提出AdamX改进其二阶矩估计的指数衰减机制，属于深度学习理论中的optimizer研究方向，与用户高优先级方向高度相关。
Score: 8
Field: 深度学习理论

### [Score: 7.0/10] Toward bilipshiz geometric models
- **Authors:** Yonatan Sverdlov, Eitan Rosen, Nadav Dym
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11735](https://arxiv.org/abs/2511.11735)
- **Reason:** 研究点云模型的双Lipschitz等价性，分析现有不变网络的几何性质并提出改进方法，属于深度学习理论中的几何模型研究方向。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Calibrated Decomposition of Aleatoric and Epistemic Uncertainty in Deep Features for Inference-Time Adaptation
- **Authors:** Divake Kumar, Patrick Poggi, Sina Tayebati, Devashri Naik, Nilesh Ahuja, Amit Ranjan Trivedi
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12389](https://arxiv.org/abs/2511.12389)
- **Reason:** 提出深度特征空间中 aleatoric与epistemic不确定性的校准分解方法，用于推理时自适应模型选择，属于深度学习理论中的不确定性估计与泛化研究。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] MFI-ResNet: Efficient ResNet Architecture Optimization via MeanFlow Compression and Selective Incubation
- **Authors:** Nuolin Sun, Linyuan Wang, Haonan Wei, Lei Li, Bin Yan
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12422](https://arxiv.org/abs/2511.12422)
- **Reason:** 将生成模型的MeanFlow流场思想引入ResNet架构优化，通过压缩-孵化策略提升参数效率与判别性能，属于深度学习理论中的网络结构创新。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] MaskAnyNet: Rethinking Masked Image Regions as Valuable Information in Supervised Learning
- **Authors:** Jingshan Hong, Haigen Hu, Huihuang Zhang, Qianwei Zhou, Zhao Li
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12480](https://arxiv.org/abs/2511.12480)
- **Reason:** 提出将掩码区域作为监督学习的有效信息源，通过掩码语义多样性增强特征表达，属于深度学习理论中的训练策略创新。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Clustering-Based Weight Orthogonalization for Stabilizing Deep Reinforcement Learning
- **Authors:** Guoqing Ma, Yuhan Zhang, Yuming Dai, Guangfu Hao, Yang Chen, Shan Yu
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11607](https://arxiv.org/abs/2511.11607)
- **Reason:** 提出COWM层通过聚类和正交化稳定RL模型训练，解决非平稳环境下的样本效率问题，属于深度学习理论中的RL架构优化。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Enhancing PINN Accuracy for the RLW Equation: Adaptive and Conservative Approaches
- **Authors:** Aamir Shehzad
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11638](https://arxiv.org/abs/2511.11638)
- **Reason:** 提出自适应和保守的PINN方法，提升RLW方程的求解精度，解决物理驱动模型的误差问题，属于深度学习理论中的优化方向。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] On the Probabilistic Learnability of Compact Neural Network Preimage Bounds
- **Authors:** Luca Marzari, Manuele Bicego, Ferdinando Cicalese, Alessandro Farinelli
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11656](https://arxiv.org/abs/2511.11656)
- **Reason:** 研究神经网络前像边界的概率可学习性，提出RF-ProVe方法，属于深度学习理论中的可学习性分析方向。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Clifford Algebraic Rotor Embeddings : Maybe embeddings should start to CARE
- **Authors:** Sameeksha Sriram, Ayush Paliwal, Alexander S. Ecker, Chase van de Geijn
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11665](https://arxiv.org/abs/2511.11665)
- **Reason:** 提出Clifford代数转子嵌入改进位置编码的 commutative 性，提升Transformer性能，属于深度学习理论中的嵌入架构方向。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Adaptive Stepsizing for Stochastic Gradient Langevin Dynamics in Bayesian Neural Networks
- **Authors:** Rajit Rajpal, Benedict Leimkuhler, Yuanhao Jiang
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11666](https://arxiv.org/abs/2511.11666)
- **Reason:** 提出SA-SGLD方法通过自适应步长提升BNN后验采样精度，解决优化器步长选择问题，属于深度学习理论中的optimizer方向。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Do traveling waves make good positional encodings?
- **Authors:** Chase van de Geijn, Ayush Paliwal, Timo Lüddecke, Alexander S. Ecker
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11668](https://arxiv.org/abs/2511.11668)
- **Reason:** 研究行波作为位置编码的有效性，提出RollPE方法，提升Transformer位置编码性能，属于深度学习理论中的位置编码方向。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Learning with Preserving for Continual Multitask Learning
- **Authors:** Hanchen David Wang, Siwoo Bae, Zirong Chen, Meiyi Ma
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11676](https://arxiv.org/abs/2511.11676)
- **Reason:** 提出LwP框架通过保持表示空间几何结构提升持续多任务学习性能，解决灾难性遗忘问题，属于深度学习理论中的持续学习方向。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Regularized Schrödinger: Alleviating Distortion and Exposure Bias in Solving Inverse Problems
- **Authors:** Qing Yao, Lijian Gao, Qirong Mao, Dong Ming
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11686](https://arxiv.org/abs/2511.11686)
- **Reason:** 提出正则化薛定谔桥方法缓解逆问题中的失真和暴露偏差，提升模型鲁棒性，属于深度学习理论中的逆问题优化方向。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Tighter Truncated Rectangular Prism Approximation for RNN Robustness Verification
- **Authors:** Xingqi Lin (unknown), Liangyu Chen (unknown), Min Wu (unknown), Min Zhang (unknown), Zhenbing Zeng (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11699](https://arxiv.org/abs/2511.11699)
- **Reason:** 提出更紧的截断长方体近似方法用于RNN鲁棒性验证，涉及网络架构的鲁棒性分析，属于深度学习理论方向
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] FSC-Net: Fast-Slow Consolidation Networks for Continual Learning
- **Authors:** Mohamed El Gorrim (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11707](https://arxiv.org/abs/2511.11707)
- **Reason:** 提出快慢整合网络解决持续学习中的灾难性遗忘，涉及网络架构设计，属于深度学习理论方向
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] KAN/H: Kolmogorov-Arnold Network using Haar-like bases
- **Authors:** Susumu Katayama (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11736](https://arxiv.org/abs/2511.11736)
- **Reason:** 提出使用Haar-like基的Kolmogorov-Arnold网络，改进网络架构，属于深度学习理论的network architecture方向
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Sumudu Neural Operator for ODEs and PDEs
- **Authors:** Ben Zelenskiy (unknown), Saibilila Abudukelimu (unknown), George Flint (unknown), Kevin Zhu (unknown), Sunishchal Dev (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11762](https://arxiv.org/abs/2511.11762)
- **Reason:** 提出Sumudu神经算子用于ODE和PDE求解，涉及神经算子架构设计，属于深度学习理论的network architecture方向
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Learning Fair Representations with Kolmogorov-Arnold Networks
- **Authors:** Amisha Priyadarshini (unknown), Sergio Gago-Masague (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11767](https://arxiv.org/abs/2511.11767)
- **Reason:** 结合Kolmogorov-Arnold网络学习公平表示，涉及网络架构和公平性，属于深度学习理论的network architecture方向
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Robust Bidirectional Associative Memory via Regularization Inspired by the Subspace Rotation Algorithm
- **Authors:** Ci Lin (unknown), Tet Yeap (unknown), Iluju Kiringa (unknown), Biwei Zhang (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11902](https://arxiv.org/abs/2511.11902)
- **Reason:** 提出正则化策略提升双向联想记忆的鲁棒性，涉及网络架构和正则化，属于深度学习理论的network architecture方向
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Batch Matrix-form Equations and Implementation of Multilayer Perceptrons
- **Authors:** Wieger Wesselink (unknown), Bram Grooten (unknown), Huub van de Wetering (unknown), Qiao Xiao (unknown), Decebal Constantin Mocanu (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11918](https://arxiv.org/abs/2511.11918)
- **Reason:** 推导MLP的批量矩阵形式方程，涉及网络架构的数学基础，属于深度学习理论的network architecture方向
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Beyond the Laplacian: Interpolated Spectral Augmentation for Graph Neural Networks
- **Authors:** Ziyao Cui (unknown), Edric Tam (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11928](https://arxiv.org/abs/2511.11928)
- **Reason:** 提出插值拉普拉斯嵌入用于GNN特征增强，涉及网络架构的谱增强，属于深度学习理论的network architecture方向
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] A Systematic Analysis of Out-of-Distribution Detection Under Representation and Training Paradigm Shifts
- **Authors:** C. César Claros Olivares (unknown), Austin J. Brockmeier (unknown)
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11934](https://arxiv.org/abs/2511.11934)
- **Reason:** 系统分析表示和训练范式变化下的OOD检测，涉及训练范式和表示学习，属于深度学习理论的training paradigm方向
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Self-Organization of Attractor Landscapes in High-Capacity Kernel Logistic Regression Hopfield Networks
- **Authors:** Akira Tamamori
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13053](https://arxiv.org/abs/2511.13053)
- **Reason:** 分析核Hopfield网络的吸引子景观，提出“优化脊”现象与自组织机制，属于深度学习理论中的经典网络架构研究，对理解Hopfield网络的记忆容量与稳定性有理论贡献。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Laplace Learning in Wasserstein Space
- **Authors:** Mary Chriselda Antony Oliver, Michael Roberts, Carola-Bibiane Schönlieb, Matthew Thorpe
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13229](https://arxiv.org/abs/2511.13229)
- **Reason:** 研究Wasserstein空间中的拉普拉斯学习，扩展半监督学习到无限维空间，属于深度学习理论中的manifold hypothesis与半监督学习理论，对高维数据的表征学习有理论贡献。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Tab-PET: Graph-Based Positional Encodings for Tabular Transformers
- **Authors:** Yunze Leng, Rohan Ghosh, Mehul Motani
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.13338](https://arxiv.org/abs/2511.13338)
- **Reason:** 提出Tab-PET用图生成表格Transformer的位置编码，提升泛化性能，属于深度学习理论中的表格模型架构研究，对Tabular Transformer的改进有帮助。
Score: 7
Field: 深度学习理论

### [Score: 7.0/10] Decoupled Action Head: Confining Task Knowledge to Conditioning Layers
- **Authors:** Jian Zhou, Sihao Lin, Shuai Fu, Qi WU
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.12101](https://arxiv.org/abs/2511.12101)
- **Reason:** 针对行为克隆（BC）中Diffusion Policy的数据稀缺问题，提出解耦训练策略，预训练通用动作头并通过条件层适配新任务，提升了训练效率与模型泛化能力，属于深度学习理论中网络架构与训练策略的重要研究。
Score: 7
Field: 深度学习理论

### [Score: 6.0/10] Toward Better Generalization in Few-Shot Learning through the Meta-Component Combination
- **Authors:** Qiuhao Zeng
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11632](https://arxiv.org/abs/2511.11632)
- **Reason:** 提出元组件组合的元学习算法，提升少样本学习的泛化能力，属于深度学习理论中的元学习方向。
Score: 6
Field: 深度学习理论

### [Score: 6.0/10] H-Model: Dynamic Neural Architectures for Adaptive Processing
- **Authors:** Dmytro Hospodarchuk
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11669](https://arxiv.org/abs/2511.11669)
- **Reason:** 提出H-Model动态神经网络架构实现自适应计算，探索可解释的动态模型方向，属于深度学习理论中的动态架构方向。
Score: 6
Field: 深度学习理论

### [Score: 6.0/10] A neural optimization framework for free-boundary diffeomorphic mapping problems and its applications
- **Authors:** Zhehao Xu, Lok Ming Lui
- **Published:** 2025-11-18
- **Link:** [https://arxiv.org/abs/2511.11679](https://arxiv.org/abs/2511.11679)
- **Reason:** 提出SBN-Opt神经优化框架解决自由边界微分同胚映射问题，属于深度学习理论中的优化框架方向。
Score: 6
Field: 深度学习理论

