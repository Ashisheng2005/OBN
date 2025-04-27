# OBN

## 基于ospf和gbp的网络数据实现自主决策网络模型

1. 安装第三方库

```bash
pip install -r requirements.txt
```



已经训练好的模型文件

分类模型：

```bash
.\model\best_path_classification_model.weights.h5
```

回归模型：

```bash
.\model\best_path_regression_model.weights.h5
```

回归模型采用R2评分。

模型训练的结果将会保存在./chart目录中，

ROC曲线（真阳性率-假阳性率）

**8000数量级**

![](https://github.com/Ashisheng2005/OBN/blob/main/chart/Classification_17077.png)

![PR曲线（精确率-召回率）](https://github.com/Ashisheng2005/OBN/blob/main/chart/Classification_pr_roc.png)

![](https://github.com/Ashisheng2005/OBN/blob/main/chart/Packet_Loss_Boxplot.png)

![](https://github.com/Ashisheng2005/OBN/blob/main/chart/Packet_Loss_Distribution.png)

![](https://github.com/Ashisheng2005/OBN/blob/main/chart/Regression_10134.png)

训练数据通过ospf和bgp结果通过算法模拟生成，如果想要更加接近现实可添加更复杂的数据。

ospf数据存放位置：

```bash
.\data\ospf_data.txt
```

bgp数据存放位置：

```bash
.\data\bgp_data.txt
```

数据示例：

```bash
# BGP Data:
BGP table version is 12, local router ID is 192.168.1.1
Status codes: s suppressed, d damped, h history, * valid, > best, i - internal
Origin codes: i - IGP, e - EGP, ? - incomplete

   Network          Next Hop            Metric LocPrf Weight Path
*> 10.1.1.0/24      192.168.2.2             100    100      0 65001 i
*  10.1.2.0/24      192.168.2.3             200    100      0 65002 65003 i
*> 172.16.1.0/24    192.168.2.4             150    200      0 65004 i
*  10.1.3.0/24      192.168.2.5             120    80       0 65005 65006 65007 i
*> 192.168.3.0/24   192.168.2.6             110    120      0 65008 i
*  10.1.4.0/24      192.168.2.7             130    90       0 65009 65010 i
*> 10.1.5.0/24      192.168.2.8             140    100      0 65001 65002 i
*  172.16.2.0/24    192.168.2.9             160    150      0 65003 i
*> 10.1.6.0/24      192.168.2.10            170    100      0 65004 65005 65006 i
*  10.1.7.0/24      192.168.2.11            180    80       0 65007 i
*> 192.168.4.0/24   192.168.2.12            190    200      0 65008 65009 i
*  10.1.8.0/24      192.168.2.13            200    90       0 65010 i
*> 10.1.9.0/24      192.168.2.14            210    100      0 65001 65002 65003 i
*  172.16.3.0/24    192.168.2.15            220    120      0 65004 i
*> 10.1.10.0/24     192.168.2.16            230    150      0 65005 i

# OSPF Data:
Neighbor ID     Pri   State           Dead Time   Address         Interface
10.0.0.2        1     FULL/DR         00:00:38    192.168.1.2     GigabitEthernet0/0
10.0.0.3        1     FULL/BDR        00:00:35    192.168.1.3     GigabitEthernet0/1
10.0.0.4        1     2WAY/DROTHER    00:00:40    192.168.1.4     GigabitEthernet0/2
10.0.0.5        0     DOWN/-          00:00:00    192.168.1.5     GigabitEthernet0/3
10.0.0.6        1     FULL/DR         00:00:37    192.168.1.6     GigabitEthernet0/4
10.0.0.7        1     FULL/BDR        00:00:36    192.168.1.7     GigabitEthernet0/5
10.0.0.8        1     2WAY/DROTHER    00:00:39    192.168.1.8     GigabitEthernet0/6
10.0.0.9        1     FULL/DR         00:00:38    192.168.1.9     GigabitEthernet0/7
10.0.0.10       0     DOWN/-          00:00:00    192.168.1.10    GigabitEthernet0/8
10.0.0.11       1     FULL/BDR        00:00:35    192.168.1.11    GigabitEthernet0/9
```



## 训练模型

在开始训练模型之前需要进行一些必要的配置，训练模型的文件为

```bash
.\main.py
```



**真实环境获取数据进行训练**

训练中的数据默认全部为模拟数据，如果你需要根据真实数据训练，可以对文件中21行如下设置：

```python
    def main(self):
        training_data = Collect_network_data(mode="real").main()	# 设置mode为真实，默认为虚拟
        ...
```

当设置模式为真实后，会启动脚本自动连接交换机，所以你需要正确配置链接信息：

```yaml
# ./config.yaml
  real:
    devices:
      - device_type: "cisco_ios"
        host: "192.168.1.1"
        username: "admin"
        password: "password"
      - device_type: "huawei"
        host: "router2"
        username: "admin"
        password: "password"

```

device中可包含多台设备的配置，支持多设备批量获取数据。



**虚拟环境通过自定义数据进行训练**

通过向**.\data\ospf_data.txt** 和 **.\data\bgp_data.txt **中添加网络协议数据进行训练，默认提供一些数据。



**调整配置文件**

```yaml
# ./config.json
...
data_collection:
  mode: "virtual"			# 模式
  virtual:
    num_samples: 5000		# 数据集大小
    congestion_lambda: 2	# 链路拥堵系数λ的大小
    packet_loss_scale: 2	# 数据包丢包规模
    edge_case_prob: 0.1		# 边缘概率
    ospf_data_path: "./data/ospf_data.txt"	# ospf数据文件路径
    bgp_data_path: "./data/bgp_data.txt"	# bgp数据文件路径
...
```

实际训练中可能跑不到设定的步数，因为模型为了防止过拟合引入了**早停**机制。

```yaml
# ./config.json
...

model:
  classification:
    epochs: 200			# 分类模型最大步数
    batch_size: 128		# 批大小
    capacity_unit:		# 不同层的单元大小
      large: 512
      medium: 256
      small: 128
  regression:			# 回归模型设置，同上
    epochs: 200
    batch_size: 128
    capacity_unit:
      large: 512
      medium: 256
      small: 128

logging:					# 日志设置
  level: "INFO"
  file: "./logs/app.log"	# 日志保存位置

output:
  model_dir: "./model/"		# 模型保存位置
  plot_dir: "./chart/"		# 结果图保存位置
```



配置文件修改完成后即可启动训练文件：

```bash
python main.py
```

# 更新内容



## 2025.4.17

数据问题：

1. 目前特征集包括了（' as_path_length ', ' local_pref ', ' ospf_state ', ' bandwidth ', ' latency ', ' packet_loss ', ' bandwidth_utilization '），但实际指标远不止这些，可以进行扩展以捕获更多的网络动态，例如：抖动、往返长度（RTT）、路由器的CPU/内存利用率或队列长度等等。（**已实现**）
2. 时态特征：可以从上次ospf状态变化或BGP如有更新到现在的时间来捕获网络稳定性。
3. 派生特征：计算路径稳定性（路由振荡频率）或归一化指标（延迟/带宽的比率）等特征来丰富数据集（**已实现**）
4. 分类编码：对于分类变量，例如“接口”类型，使用单热编码或目标编码，而不是完全依赖于模拟带宽值。目前接口类型的信息完全通过带宽值间接表示，丢失了接口类型的固有类别特性，可能导致模型无法直接捕获接口类型之间的非线性差异。将 interface 转换为三个二进制列（is_FastEthernet, is_GigabitEthernet, is_TenGigabitEthernet），每列表示是否为对应接口类型（1 或 0）。用目标变量（is_best）的统计信息（例如均值）替换接口类型，例如将 FastEthernet 替换为 is_best 在 FastEthernet 样本中的平均值。（**已实现**）
   1. 单热编码的优点：单热编码直接表示接口类型，避免通过带宽值间接推导。例如，FastEthernet 和 GigabitEthernet 之间的差异不仅是带宽（100 vs 1000 Mbps），还可能涉及硬件特性、稳定性等隐含因素，单热编码能保留这些类别特性。其次对于非线性模型（如决策树、神经网络），单热编码允许模型学习每种接口类型的独特模式，而不假设带宽的线性关系。
   2. 单热编码允许模型独立学习每种接口类型对 is_best 的贡献，而无需假设带宽值完全代表接口性能。例如，TenGigabitEthernet 可能因高带宽通常有高 is_best 概率，但单热编码允许模型发现例外情况（如高带宽但高延迟的路径）。后续使用分类模型（如随机森林、XGBoost）预测 is_best，单热编码是标准做法，能直接输入到模型中，无需额外归一化。
   3. 缺点：单热编码将为每种接口类型添加一列，当前有 3 种接口类型，会增加 2-3 列（取决于是否丢弃一列以避免多重共线性）。对于 较大的数据集影响不大，但如果接口类型增加（如添加更多接口），维度可能显著膨胀。维度增加可能导致模型训练时间变长，尤其对于线性模型（如逻辑回归）或深度学习模型的影响会更加显著。
   4. 带宽值已经部分捕获了接口类型的特性（FastEthernet=100, GigabitEthernet=1000, TenGigabitEthernet=10000），单热编码可能引入冗余信息，增加模型过拟合风险。但可以通过特征选择或正则化（如 L1/L2 正则）减少冗余。
   5. 如果完全移除带宽值，仅使用单热编码，模型将丢失带宽的连续性信息（如 100 Mbps 和 1000 Mbps 的量级差异），可能降低模型对路径性能的敏感性。所以选择保留带宽，结合单热编码使用。
5. 在生成的模拟数据中，合成数据可能无法完全代表现实世界的场景，但可以使用**统计分布**来更现实的模拟拥塞和数据包丢失，或者**基于现实网络拓扑**模拟数据，以考虑多跳依赖关系和路由循环。而对于**数据增强**方面，可以引入噪声或者扰动来模拟边缘情况（例如：链路故障，突然流量峰值等）。（**已实现**）
   1. 具体决策：统计分布 + 数据增强，对于拓扑模拟方案，缺点是实现复杂、数据需求高、计算成本大且需要真实拓扑数据或者需建模模拟多跳。列入后续更新考虑，但暂时放弃实现该方案。
   2. 使用**泊松分布**（拥塞）和**指数分布**（丢包）优化现有特征生成，添加边缘情况（流量峰值、多链路故障），确保模型学习到罕见场景。
   3. 在非拥塞场景下，添加微小噪声（如 0-0.5 的丢包率），模拟现实网络中即使无拥塞也可能存在的微小丢包。
6. 对于训练时进行的SMOTE过采样，虽然解决了is_best的类不平衡问题，但也可能引入人工合成痕迹。为解决这个问题，后续我们会通过替代技术尝试其他不平衡处理方法，例如**类加权损失函数**或对**多数类进行欠采样**。**阈值调整**也是一种不错的选择，目前采用固定的is_best阈值的方案并不靠谱，后续会改为使用验证集来动态调整决策边界。汇总方案为将SMOTE与**ADASYN**或**borderline-SMOTE**等其他技术相结合，以生成更强大的合成样本。（**已实现**）
   1. ASASYN（Adaptive Synthetic Sampling）
      1. ADASYN是SMOTE的改进版本，它根据样本的“困难程度”（即少数类样本靠近多数类边界的程度）自适应地生成更多合成样本。相比SMOTE，ADASYN更关注那些难以分类的边界样本，能生成更具代表性的合成样本。可以使用imblearn.over_sampling.ADASYN库，替换现有的SMOTE实现。调整sampling_strategy参数，控制合成样本的比例。可以在simulate_training_data后添加ADASYN处理，确保生成的is_best少数类样本更贴近真实网络场景。
   2. Borderline-SMOTE
      1. Borderline-SMOTE只对少数类中靠近决策边界的样本进行过采样，而忽略那些“安全”（远离边界的）样本。这种方法能减少合成样本的冗余，降低人工痕迹。可以使用imblearn.over_sampling.BorderlineSMOTE，设置kind='borderline-1'或borderline-2。在生成df后，针对is_best标签应用该方法，结合您的logger记录合成样本的数量和分布（如is_best的均值、标准差）。但是需要注意检查合成样本的packet_loss、latency等关键特征是否符合真实网络分布。
   3. SMOTEENN（SMOTE + Edited Nearest Neighbors）
      1. SMOTEENN结合了过采样和欠采样，先用SMOTE生成少数类样本，再通过Edited Nearest Neighbors（ENN）清理那些与多数类样本过于相似的合成样本。这种方法能有效减少人工痕迹，同时提高样本质量。可以使用imblearn.combine.SMOTEENN，在DataProcessor中，生成df后应用SMOTEENN，确保is_best分布更平衡。通过logger记录清理后的样本数量和特征统计，验证path_cost、route_stability等衍生特征的分布是否合理。
   4. Random Undersampling
      1. 随机删除多数类样本，简单高效，但可能丢失重要信息。使用imblearn.under_sampling.RandomUnderSampler，设置sampling_strategy为目标比例（如0.5表示1:1平衡）。在simulate_training_data后，应用欠采样，并通过plotter.plot_edge_cases检查欠采样后packet_loss、jitter等特征的分布是否仍具代表性。记录logger中欠采样前后is_best的比例变化。
   5. Tomek Links
      1. Tomek Links识别并移除多数类中靠近少数类的样本，这些样本可能是噪声或冗余的。使用imblearn.under_sampling.TomekLinks。结合SMOTE或ADASYN，先过采样少数类，再用Tomek Links清理边界样本。验证latency_bandwidth_ratio、path_cost等衍生特征的分布，确保欠采样后数据仍能反映网络的拥塞和故障场景。
   6. Cluster Centroids
      1. 通过将多数类样本聚类并替换为质心，减少多数类样本量，同时保留分布特性。使用imblearn.under_sampling.ClusterCentroids。针对is_best=0的样本，基于bandwidth、latency等关键特征进行聚类。检查route_stability和queue_length的统计信息，确保质心样本能代表网络动态。
   7. 类加权损失函数
      1. 通过在模型训练时为少数类（is_best=1）分配更高的权重，可以在不改变数据集的情况下解决类不平衡问题。这种方法避免了合成样本和数据丢失的风险，很适合网络数据特征复杂、真实性要求高的场景。
      2. 在模型训练时（假设使用如XGBoost、LightGBM或神经网络），设置class_weight参数。例如，在sklearn中可设置class_weight='balanced'，或手动指定权重（如{0:1, 1:5}，根据is_best分布调整）。
      3. 结合现有的is_best动态阈值逻辑，计算训练集中is_best=1和is_best=0的比例，动态设置权重。
      4. 在logger中记录加权后的损失值，验证模型对少数类的预测性能（如F1分数、PR-AUC）。
      5. 需要注意的是：类加权可能导致模型对多数类欠拟合，需通过交叉验证监控is_best=0的性能。
   8. 动态阈值调整
      1. 基于验证集的阈值优化
         1. 使用验证集的PR曲线（Precision-Recall Curve）或ROC曲线，动态选择最优阈值。
         2. 在simulate_training_data后，将df拆分为训练集和验证集，使用sklearn.metrics.precision_recall_curve计算不同阈值的精确率和召回率，选择F1分数最高的阈值。在logger中记录最优阈值及其对应的性能指标（如F1、AUC），定期更新阈值，适应bandwidth_utilization、route_stability等特征的分布变化。
      2. 基于特征的阈值调整
         1. 考虑bandwidth、latency等关键特征的影响，动态调整is_best的阈值。例如，在高拥塞场景（bandwidth_utilization>80%）下降低阈值，以增加is_best=1的比例。
         2. 在is_best评分逻辑中，添加基于bandwidth_utilization或queue_length的权重调整。通过plotter.plot_edge_cases验证调整后is_best分布是否更合理。
   9. 基于物理约束的生成模型
      1. 目前simulate_training_data方法已经通过物理约束（如bandwidth、latency的分布）生成逼真的网络数据。针对类不平衡问题，可以进一步引入生成模型（如GAN或VAE）生成少数类样本，同时确保样本符合网络物理特性。
      2. 条件GAN（Conditional GAN）：训练一个条件GAN，输入is_best=1的条件，生成符合bandwidth、packet_loss等特征分布的样本。使用PyTorch或TensorFlow实现条件GAN，输入特征包括is_FastEthernet、is_GigabitEthernet等，约束生成样本的path_cost和route_stability分布，参考simulate_training_data中的逻辑，最后通过plotter.plot_edge_boxplot比较生成样本与真实样本的分布。
      3. 变分自编码器（VAE）：VAE可以学习少数类样本的潜在分布，生成更自然的样本。针对is_best=1的样本训练VAE，可以选择重点建模latency、jitter等关键特征。并且在生成样本时，添加后处理步骤（如裁剪packet_loss到0-50），确保物理合理性。
   10. 验证和监控
       1. 每次对于任何方面的改动都需要进行多种验证和监控
       2. 分布一致性检查：比较处理前后packet_loss、route_stability等特征的分布，使用KS检验或KL散度。使用scipy.stats.ks_2samp检查合成样本与原始样本的分布差异且在logger中记录检验结果。
       3. 模型性能监控：使用多种指标（如F1、PR-AUC、ROC-AUC）评估模型在is_best上的性能，尤其关注边缘案例（高packet_loss或queue_length）。在plotter中添加PR曲线和ROC曲线的绘制功能且记录不同策略（如SMOTE vs. ADASYN vs. 类加权）的性能对比
   11. 策略优先级
       1. **尝试ADASYN + Tomek Links**：ADASYN生成更高质量的少数类样本，Tomek Links清理冗余样本，减少人工痕迹。
       2. **结合类加权损失函数**：在模型训练时为is_best=1分配更高权重，弥补数据分布的不足。
       3. **动态阈值调整**：基于验证集的PR曲线优化is_best阈值，适应不同网络场景。
       4. **监控分布和性能**：通过logger和plotter持续验证特征分布和模型性能。
7. 目前的对于数据有效性的保证是通过基本断言来实现的。但对真实世界的数据没有全面的清理。解决方案一是进行离群检测，通过实现**离群检测**（例如**IQR**或者**孤立森林**算法）来过滤’延迟‘、'pack_loss'或'带宽'中的异常值。而对于部分数据可能存在的数据缺失所造成的ospf/bgp数据不完整。后续需要补充额外的缺失值处理函数。对于特征关联分析，可以使用关联矩阵或互信息评分来识别冗余特征，降低维数。（**已实现**）

模型架构问题

1. 目前使用的分类模型和回归模型使用三层隐藏层和L2正则化的固定架构，这不是所有数据集的最佳选择。

   1. 解决方案：

   2. **动态架构调整**：根据数据集大小动态调整层数和神经元数量，允许更灵活的模型架构。

      **BatchNormalization**：在每层后添加BatchNormalization以稳定训练。

      **跳跃连接**：引入残差连接（Residual Connections）以缓解深层网络的梯度消失问题。

      实现思路：修改RegressionModel和ClassificationModel的build方法，允许通过配置动态设置层数和神经元数量。

      添加BatchNormalization层和残差连接。

2. 关于超参数调优问题，后续可以使用自动化工具，例如：Keras Tuner或者Optuna来优化层大小、辍学率、学习率和L2正则化强度。其次，可以通过添加"BatchNormalization" 层来实现稳定训练并提高收敛性，特别是对于更深的网络。而对于较大的数据集，可以通过合并跳过连接以缓解更深层次架构中的梯度消失问题。特别的，对于回归模型，实现自定义损失函数来处理 path_cost 中的异常值。

   1. **解决方案**：

      - **超参数调优**：集成Optuna进行自动化超参数搜索，优化层大小、Dropout率、学习率和L2正则化强度。
      - **自定义损失函数**：为回归模型实现Huber损失函数，以更好地处理path_cost中的异常值。
      - **BatchNormalization**：已在上一步实现。
      - **跳跃连接**：同上。

      **实现思路**：

      - 在trainer.py中添加Optuna超参数调优逻辑，搜索最优参数。
      - 在RegressionModel中实现Huber损失函数。

3. 当前的代码库完全依赖于密集的神经网络，但实际上它可能不像其他模型那样有效地捕获网络数据中的复杂模式。由于网络路由数据本质上是基于图的（路由器作为节点，链路作为边），gnn可以更有效地构建拓扑模型。对于实时预测可以使用rnn或者lstm来捕获路由数据中的时间依赖性。对于较小的数据集，梯度增强模型例如：XGBoost、LightGBM等，通常优于神经网络，，并且计算强度较低。而在最后，可以将神经网络于基于树的模型结合在一起，以利用他们的互补优势。

   1. **解决方案**：

      - **图神经网络（GNN）**：使用PyTorch Geometric实现简单的GNN模型，捕获网络拓扑。
      - **时间序列模型（LSTM）**：为实时预测添加LSTM模型，处理时间依赖性。
      - **梯度增强模型**：集成XGBoost作为备选模型，适用于小型数据集。
      - **模型集成**：结合神经网络和基于树的模型，采用加权平均或Stacking方法。

      **实现思路**：

      - 添加新的GNNModel类，使用PyTorch Geometric实现。
      - 在trainer.py中添加XGBoost训练逻辑。
      - 实现模型集成逻辑。

4. 当前模型使用L2正则化和dropout，但是dropout率相对较高，可能导致欠拟合。为解决该问题，后续可能会使用蒙特卡罗退出的不确定性估计等技术。通过实现L2正则化的衰减调度，以减少其在后期训练阶段的影响。

   1. **解决方案**：

      - **降低Dropout率**：将Dropout率从0.5-0.7降低到0.2-0.4，并通过Optuna优化。
      - **L2正则化衰减**：实现L2正则化的学习率调度，逐渐降低正则化强度。
      - **蒙特卡罗Dropout**：在预测时启用MC Dropout以估计不确定性。

      **实现思路**：

      - 修改build方法中的Dropout率，并通过Optuna搜索最佳值。
      - 在trainer.py中实现L2正则化衰减调度。
      - 在predict_optimal_path中添加MC Dropout逻辑。

5. 目前模型使用早停和降低学习率的机制，但训练稳定性可以提高，例如：应用梯度裁剪，以防止爆炸梯度，特别是对于较大的数据集。指定’热身计划‘：使用学习率热身阶段来稳定初始训练。通过实现k-fold 交叉验证，以确保跨不同数据分割的稳健性能。

   1. **解决方案**：

      - **梯度裁剪**：在模型训练时启用梯度裁剪，限制梯度范数。
      - **学习率热身**：实现学习率热身调度，初始阶段逐渐增加学习率。
      - **K折交叉验证**：在trainer.py中实现K折交叉验证，确保模型稳健性。

      **实现思路**：

      - 在RegressionModel和ClassificationModel的训练逻辑中添加梯度裁剪。
      - 使用tf.keras.callbacks实现学习率热身。
      - 在trainer.py中添加K折交叉验证逻辑。

6. 目前，batch大小和epoch是根据数据大小改动的，但是其他超参数是固定的。对于这种问题会使用网格搜索或者贝叶斯优化来找到学习率，batch大小和层数的最优值。也可以尝试像RMSprop或者AdamW这样的优化器，他们可能会更快的收敛于某些数据集。还有动态的调整epoch数，而不是使用固定的数值。

   1. **解决方案**：

      - **贝叶斯优化**：使用Optuna进行贝叶斯优化，搜索学习率、Batch大小和层数。
      - **动态Epoch**：根据验证集损失动态调整Epoch数。
      - **优化器选择**：测试AdamW和RMSprop优化器。

      **实现思路**：

      - 在trainer.py中集成Optuna进行超参数优化。
      - 添加动态Epoch逻辑，基于早停机制。
      - 在模型编译时支持AdamW和RMSprop。

评价指标问题

1. 当前模型中，分类模型使用精度，回归模型使用MSE、MAE和R²,但这些可能都无法完全捕获模型的性能。而对于这一方面的修改，计划添加进度、召回率，f1得分和ROC-AUC来评估分类模型，特别是对于不平衡的数据集。至于回归模型，添加RMSE或者MAPE等指标，以提供更全面的的回归性能视图。最后是自定义指标，例如：路径选择准确性、平均延迟减少等等，但其根本依然是与网络优化目标保持一致。

部署和可伸缩性问题

1. 虽然在Netmiko中进行设备轮询，但这可能无法扩大到大型网络或频繁更新的环境中，而可以使用异步轮询的方式，使用 asyncio 或 concurrent 在多个设备上并行收集数据。或者集成遥测协议，如gNMI 或 NETCONF实时时间驱动的数据收集，而不是定期轮询。缓存访问频繁的OSPF/BGP数据，减少冗余查询，提高相应速度。
2. start.py 为每个推理加载模型和所放弃，但这对于实时应用程序是低效的，应用训练后量化（如使用TensorFlow Lite）来减少模型大小和推理延迟。或者将模型转换为ONNX格式，以实现更快的推理和跨平台兼容性。start.py后续将支持同时对多条路径进行批处理预测，提高吞吐量。
3. 代码库的框架单一，缺乏模块化部署框架，因为目前只是实验性测试，后续将会采用多种方案进行比对测试例如：微服务，将数据采集、预处理和推理分离到单独的微服务中，以获得更好的可扩展性和故障隔离。Containerization：使用Docker容器化应用程序，确保环境一致，更容易部署。而对于大规模网络，可部署在Kubernetes上，实现大规模网络的负载均衡和自动伸缩。



## 2025.4.13

分类模型问题：

**过拟合**：在小数据集上，模型可能记住训练数据，导致高准确率，但在大数据集上泛化能力下降。

**模型复杂度不足**：如果模型结构较简单（如浅层神经网络），可能无法捕捉大数据集的复杂模式。

**数据质量或分布问题**：大数据集可能引入噪声或类别不平衡，导致准确率下降。

**超参数未优化**：学习率、批量大小、优化器等可能未针对不同数据量和Epoch进行调整。



回归模型问题：

**回归任务的损失函数**：回归模型的“准确率”可能不是最佳评估指标，均方误差（MSE）或平均绝对误差（MAE）可能更能反映模型性能。

**过拟合**：与分类模型类似，小数据集高准确率可能源于过拟合。

**特征选择或预处理不足**：回归任务对特征的缩放和分布敏感，可能需要更强的特征工程。

**模型容量不足**：回归任务可能需要更深的网络或更复杂的模型（如树模型或集成模型）。



解决方法：

**正则化**：在模型中加入Dropout、L2正则化或者Batch Normalization

**早停机制**：监控验证集损失，当验证损失不在下降时停止训练

**数据增强**：增加模拟数据的变化，引入多个网络质量指标，动态模拟网络状态。增加训练数据的多样性，减少过拟合。

**增加模型深度或宽度**：根据训练数据体量动态调配神经网络层和隐藏单元层

**尝试其他模型**：对于分类任务，可以尝试随机森林或XGBoost；对于回归任务，可以尝试梯度提升树。（暂留后续投票机制中使用）

**检查数据分布**：确保训练集和测试集分布一致，检查是否存在类别不平很或异常值

**特征工程**：对回归任务，确保特征标准化，对分类任务，检查特征重要性



