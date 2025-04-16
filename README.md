# AiNet基于ospf和gbp的网络数据实现自决策

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

模型训练的结果将会保存在./chart目录中，一下是不同数量的训练结果(前者为分类模型，后者为回归模型，共三个数量级测试)：

**100数量级**

![](https://github.com/Ashisheng2005/OBN/blob/main/chart/Classification_100.png)

![](https://github.com/Ashisheng2005/OBN/blob/main/chart/Regression_100.png)

**1000数量级**

![](https://github.com/Ashisheng2005/OBN/blob/main/chart/Classification_1000.png)

![](https://github.com/Ashisheng2005/OBN/blob/main/chart/Regression_1000.png)

**8000数量级**

![Classification_8000](https://github.com/Ashisheng2005/OBN/blob/main/chart/Classification_8000.png)

![](https://github.com/Ashisheng2005/OBN/blob/main/chart/Regression_8000.png)



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
.\Training_model.py
```



**真实环境获取数据进行训练**

训练中的数据默认全部为模拟数据，如果你需要根据真实数据训练，可以对文件中21行如下设置：

```python
    def main(self):
        training_data = Collect_network_data(mode="real").main()	# 设置mode为真实，默认为虚拟
        ...
```

当设置模式为真实后，会启动脚本自动连接交换机，所以你需要正确配置链接信息：

```json
# ./config.json
"real": {
        "device": [
            {
                "device_type": "cisco_ios",
                "host": "192.168.1.1",
                "username": "admin",
                "password": "password"
            },
            {
                "device_type": "huawei",
                "host": "router2",
                "username": "admin",
                "password": "password"
            }
        ],
    	......
    },

```

device中的每个哈希表都是一台设备的配置，支持多设备批量获取数据。



**虚拟环境通过自定义数据进行训练**

通过向**.\data\ospf_data.txt** 和 **.\data\bgp_data.txt **中添加网络协议数据进行训练，默认提供一些数据。



**调整配置文件**

```json
# ./config.json
...

    "Classification_model": {
        "epochs": 500	# 分类模型训练步数
    },
    "Regression_model": {
        "epochs": 500	# 回归模型训练步数
    },
...
"virtual": {
        "num_samples": 8000		# 虚拟环境生成的数据集大小
    }

```

实际训练中可能跑不到设定的步数，因为模型为了防止过拟合引入了**早停**机制。



配置文件修改完成后即可启动训练文件：

```bash
python Training_model.py
```



复杂环境设置：

模型实习了动态批量大小，根据数据量动态调整batch_size大小（小数据量用小批量，大数据量用大批量）

```python
batch_size = 32 if data_size < 1000 else 64 if data_size < 5000 else 128
```

在训练文件**Training_model.py**中，分类模型和回归模型**都存在各自的语句**，可根据实际需求修改：

# 更新内容



## 2024.4.13

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



