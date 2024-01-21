# 神经网络对抗攻击基础

----

### 材料

- 根论文：[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- AdvAttack工具箱
  - torchattack: [https://github.com/Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch)
  - art: [https://github.com/Trusted-AI/adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
  - robustbench: [https://github.com/RobustBench/robustbench](https://github.com/RobustBench/robustbench)


### 概念基础

⚪ 知识要点

- 对抗攻击的目的
  - 找出模型的泛化缺陷
- 对抗样本存在的原因
  - 过度线性化 over-linearility
- 万恶之源 FGSM/PGD 的原理
  - 相似性度量: Linf/L2

⚪ 对抗攻击时所要考虑的

- 模型/损失函数/训练数据已知？
  - 白盒
    - 梯度
  - 黑盒
    - 查询 (估计梯度)
    - 迁移攻击
- 诱导？
  - 无目标
  - 有目标
- 修改数据量/幅度？
  - Linf/L2: ≈ 数字攻击
  - [AdvPatch](https://arxiv.org/pdf/1712.09665.pdf): ≈ 物理攻击


### 实验任务

ℹ 熟悉基本的 git 操作和 [Fork](https://git-fork.com/) 的使用

#### 入门

- 训练一个基础任务模型
  - 预训练 Cifar10 的推理环境: [https://github.com/Kahsolt/PyTorch_CIFAR10](https://github.com/Kahsolt/PyTorch_CIFAR10)
- 用 torchattacks 跑通基本攻击
  - 攻击方法：FGSM/PGD/MI-FGSM/...
  - 调参配置 steps/alpha/eps 观察影响
  - 跑单张图，记录结果
    - 原图 X 
    - 对抗图 AX
    - 扰动 DX = AX - X
    - 扰动指标: Linf/L1/L2
  - 跑全数据集，记录指标
    - 原始精度 acc: clean样本的精度
    - 残存精度 racc: adv样本的精度
    - 攻击成功率 asr := 1 - racc
    - 预测稳定率 psr: clean和adv预测一致
    - 严格攻击成功率 sasr: clean样本分类正确，但adv样本分类错误
- 整理 torchattacks 中实现的所有攻击的发展谱系
  - 分类
  - 谁的做法继承自谁

#### 进阶

⚪ 基于 torchattack 框架实现一个玩具 Attack 类

- 熟练背诵 FGSM/PGD/MI-FGSM/PGDL2 的写法
- 如果有自己的想法：自行实现，或基于任何已有的类进行魔改、反复实验
- 如果没有自己的想法：从别的库搬迁一个新的攻击算法，跑通流程

⚪ 实现两种基本的迁移攻击

- 跨样本: 样本 X 在模型 M 上产生对抗样本 AX, 将 DX=AX-X 应用到同分布的另一个样本 X' 上看是否有攻击效果
- 跨模型: 样本 X 在模型 M 上产生对抗样本 AX，用另一个模型 M' 预测 AX 看是否有攻击效果

⚪ 复现 Universal Adversarial Perturbations: [https://arxiv.org/abs/1610.08401](https://arxiv.org/abs/1610.08401)

- 可以仅借用其思想，直接改造朴素 PGD
  - 在单个模型上，为所有样本寻找一个公共的 DX
  - 在单个模型上，为各类样本分别寻找一个公共的 DX
  - 在单个模型上，为指定的一组样本寻找一个公共的 DX

⚪ 复现 Common Weakness Attack: [https://arxiv.org/abs/2303.09105](https://arxiv.org/abs/2303.09105)

- 可以仅借用其思想，直接改造朴素 PGD
  - 在多个模型上，为单个数据样本寻找一个公共的 DX

#### 高级

⚪ **探究对抗样本的性质**

- 稳定性
  - 线性地放缩 DX，观察模型预测的变化
  - 给 AX 加一定强度的高斯噪声，观察模型预测的变化
- 线性性
  - 在 X 和 AX 之间线性插值，观察模型预测的变化
  - 在两个同类 AX 之间线性插值，观察模型预测的变化
  - 在两个不同类 AX 之间线性插值，观察模型预测的变化

⚪ General PGD: 基于 scipy 实现基于传统数值优化方法的对抗攻击

- 用 sklearn 中的任意模型 (不局限于nn) 训练一个 iris 数据集三分类模型
  - 因为传统数值优化通常只能有效处理100-元函数，所以我们不能用 cifar10 了 :(
- 用 scipy.optimze.minimize
  - 尝试各种优化方法：BFGS/COBYLA/Nelder-Mead/CG/...
  - 用 scipy.optimize.approx_fprime 估计梯度 (虽然很慢


### 拓展思考

⚪ 对抗攻击的玩具应用

- 调研下列工具如何通过对抗攻击使所得产生的样本难以在目标 StableDiffusion 模型上正确地微调
  - Mist: [https://github.com/mist-project/mist-v2](https://github.com/mist-project/mist-v2)
  - Glaze: [https://glaze.cs.uchicago.edu/](https://glaze.cs.uchicago.edu/)
  - NightShade: [https://nightshade.cs.uchicago.edu/](https://nightshade.cs.uchicago.edu/)
- 调研下述工作如何通过对抗攻击使所得产生的样本在目标模型上无法学习
  - Unlearnable-Examples: [https://github.com/HanxunH/Unlearnable-Examples](https://github.com/HanxunH/Unlearnable-Examples)

⚪ 理论延拓

- 对抗攻击和对抗生成网络 (GAN) 在何种意义上、什么方面是一回事
- 了解非神经网络类模型的对抗攻击思路
  - 尤其是kNN、NaiveBayes、决策树等没有线性层结构的模型
- 过度线性化: https://zhuanlan.zhihu.com/p/42667844
 
----
by Armit
2024/01/21 
