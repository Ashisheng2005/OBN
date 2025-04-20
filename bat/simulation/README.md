# 仿真网络模型



​		该模型构建了一个中等规模的仿真网络，包含多个自治系统(AS)和OSPF区域，确保高度的网络复杂性，同时提供关键设备的配置命令

## 网络拓扑设计

我设计了一个包含3个自治系统(AS 100、AS 200、AS 300)的网络，每个AS内部都有多个路由器和OSPF区域，具体如下：

**AS 100**：

- 包含3个路由器：R1（Area 0）、R2（Area 0）、R3（Area 1）。
- R1和R2在Area 0（骨干区域），R2和R3在Area 1。
- R1作为ASBR（Autonomous System Boundary Router），与AS 200互联。

**AS 200**：

- 包含4个路由器：R4（Area 0）、R5（Area 0）、R6（Area 2）、R7（Area 2）。
- R4和R5在Area 0，R5和R6、R7在Area 2。
- R4与AS 100的R1通过eBGP互联，R5与AS 300的R8通过eBGP互联。

**AS 300**：

- 包含3个路由器：R8（Area 0）、R9（Area 0）、R10（Area 3）。
- R8和R9在Area 0，R9和R10在Area 3。
- R8与AS 200的R5通过eBGP互联。

**终端设备**：

- 每个AS内至少有一个PC，用于测试连通性和数据采集。

**链路**：

- 使用GigabitEthernet接口，带宽设为1000 Mbps（可在模拟器中调整）。
- AS之间链路带宽可设置为100 Mbps（FastEthernet），模拟较低带宽。



## 拓扑图

```bash
AS 100                         AS 200                          AS 300
+-------+                      +-------+                       +-------+
|  R1   |-------(eBGP)---------|  R4   |--------(eBGP)---------|  R8   |
| (A0)  |                      | (A0)  |                       | (A0)  |
+---+---+                      +---+---+                       +---+---+
    |                              |                               |
    |                              |                               |
+---+---+                      +---+---+                       +---+---+
|  R2   |                      |  R5   |                       |  R9   |
| (A0)  |                      | (A0)  |                       | (A0)  |
+---+---+                      +---+---+                       +---+---+
    |                              |                               |
    |                              |                               |
+---+---+                      +---+---+------+                +---+---+
|  R3   |                      |  R6   |  R7  |                |  R10  |
| (A1)  |                      | (A2)  | (A2) |                | (A3)  |
+---+---+                      +---+---+------+                +---+---+
    |                              |   |                           |
+---+---+                      +---+---+-----+                 +---+---+
|  PC1  |                      |  PC2  | PC3 |                 |  PC4  |
+-------+                      +-------+-----+                 +-------+
```

**AS 100**：R1-R2-R3，R1和R2在Area 0，R2和R3在Area 1。

**AS 200**：R4-R5-R6-R7，R4和R5在Area 0，R5和R6、R7在Area 2。

**AS 300**：R8-R9-R10，R8和R9在Area 0，R9和R10在Area 3。

**eBGP**：R1-R4（AS 100-AS 200），R5-R8（AS 200-AS 300）。

**iBGP**：AS内部路由器之间（如R4-R5、R8-R9）。



## IP地址规划

为简化配置，使用以下IP地址段：

- AS 100：
  - R1-R2：192.168.1.0/24
  - R2-R3：192.168.2.0/24
  - R1-R4（eBGP）：10.0.12.0/30
  - PC1：192.168.3.0/24
- AS 200：
  - R4-R5：192.168.4.0/24
  - R5-R6：192.168.5.0/24
  - R5-R7：192.168.6.0/24
  - R4-R1（eBGP）：10.0.12.0/30
  - R5-R8（eBGP）：10.0.45.0/30
  - PC2：192.168.7.0/24
  - PC3：192.168.8.0/24
- AS 300：
  - R8-R9：192.168.9.0/24
  - R9-R10：192.168.10.0/24
  - R8-R5（eBGP）：10.0.45.0/30
  - PC4：192.168.11.0/24

具体IP分配：

- R1：192.168.1.1/24（to R2），10.0.12.1/30（to R4）

- R2：192.168.1.2/24（to R1），192.168.2.1/24（to R3）

- R3：192.168.2.2/24（to R2），192.168.3.1/24（to PC1）

- PC1：192.168.3.2/24（默认网关192.168.3.1）

- R4：192.168.4.1/24（to R5），10.0.12.2/30（to R1），10.0.45.1/30（to R8）

- R5：192.168.4.2/24（to R4），192.168.5.1/24（to R6），192.168.6.1/24（to R7），10.0.45.2/30（to R8）

- R6：192.168.5.2/24（to R5），192.168.7.1/24（to PC2）

- R7：192.168.6.2/24（to R5），192.168.8.1/24（to PC3）

- PC2：192.168.7.2/24（默认网关192.168.7.1）

- PC3：192.168.8.2/24（默认网关192.168.8.1）

- R8：192.168.9.1/24（to R9），10.0.45.2/30（to R5）

- R9：192.168.9.2/24（to R8），192.168.10.1/24（to R10）

- R10：192.168.10.2/24（to R9），192.168.11.1/24（to PC4）

- PC4：192.168.11.2/24（默认网关192.168.11.1）

  

## 配置命令

以下为关键路由器的配置命令，基于Cisco IOS（适用于Packet Tracer或GNS3）。我将提供R1（AS 100）、R4（AS 200）和R8（AS 300）的配置，其他路由器可类似配置。

### R1（AS 100）配置

#### 	**接口配置**：

```bash
enable
configure terminal
hostname R1
interface GigabitEthernet0/0
 ip address 192.168.1.1 255.255.255.0
 no shutdown
interface GigabitEthernet0/1
 ip address 10.0.12.1 255.255.255.252
 no shutdown
```

####  **OSPF配置**（Area 0）：

```bash
router ospf 1
 router-id 1.1.1.1
 network 192.168.1.0 0.0.0.255 area 0
```

####  **BGP配置**（eBGP with AS 200, iBGP with R2）：

```bash
router bgp 100
 neighbor 10.0.12.2 remote-as 200
 neighbor 192.168.1.2 remote-as 100
 neighbor 192.168.1.2 update-source GigabitEthernet0/0
 network 192.168.3.0 mask 255.255.255.0  # 宣告PC1的网段
```



### R4（AS 200）配置

#### **接口配置**：

```bash
enable
configure terminal
hostname R4
interface GigabitEthernet0/0
 ip address 192.168.4.1 255.255.255.0
 no shutdown
interface GigabitEthernet0/1
 ip address 10.0.12.2 255.255.255.252
 no shutdown
interface GigabitEthernet0/2
 ip address 10.0.45.1 255.255.255.252
 no shutdown
```

#### **OSPF配置**（Area 0）：

```bash
router ospf 1
 router-id 4.4.4.4
 network 192.168.4.0 0.0.0.255 area 0
```

#### **BGP配置**（eBGP with AS 100 and AS 300, iBGP with R5）：

```bash
router bgp 200
 neighbor 10.0.12.1 remote-as 100
 neighbor 10.0.45.2 remote-as 300
 neighbor 192.168.4.2 remote-as 200
 neighbor 192.168.4.2 update-source GigabitEthernet0/0
 network 192.168.7.0 mask 255.255.255.0  # 宣告PC2的网段
 network 192.168.8.0 mask 255.255.255.0  # 宣告PC3的网段
```



### R8（AS 300）配置

#### **接口配置**：

```bash
enable
configure terminal
hostname R8
interface GigabitEthernet0/0
 ip address 192.168.9.1 255.255.255.0
 no shutdown
interface GigabitEthernet0/1
 ip address 10.0.45.2 255.255.255.252
 no shutdown
```

#### **OSPF配置**（Area 0）：

```bash
router ospf 1
 router-id 8.8.8.8
 network 192.168.9.0 0.0.0.255 area 0
```

#### **BGP配置**（eBGP with AS 200, iBGP with R9）：

```bash
router bgp 300
 neighbor 10.0.45.1 remote-as 200
 neighbor 192.168.9.2 remote-as 300
 neighbor 192.168.9.2 update-source GigabitEthernet0/0
 network 192.168.11.0 mask 255.255.255.0  # 宣告PC4的网段
```



### R2（AS 100）配置（多区域OSPF示例）

#### **接口配置**：

```bash
enable
configure terminal
hostname R2
interface GigabitEthernet0/0
 ip address 192.168.1.2 255.255.255.0
 no shutdown
interface GigabitEthernet0/1
 ip address 192.168.2.1 255.255.255.0
 no shutdown
```

#### **OSPF配置**（Area 0 和 Area 1）：

```bash
router ospf 1
 router-id 2.2.2.2
 network 192.168.1.0 0.0.0.255 area 0
 network 192.168.2.0 0.0.0.255 area 1
```

#### **BGP配置**（iBGP with R1）：

```bash
router bgp 100
 neighbor 192.168.1.1 remote-as 100
 neighbor 192.168.1.1 update-source GigabitEthernet0/0
```



### PC配置（以PC1为例）

- 在Packet Tracer中手动设置：
  - IP：192.168.3.2
  - 子网掩码：255.255.255.0
  - 默认网关：192.168.3.1（R3接口）



## 数据采集方法

### OSPF数据采集

#### **查看OSPF邻居状态**（ospf_state）：

```bash
show ip ospf neighbor
```

记录邻居状态（如FULL、DOWN），可模拟链路故障（shutdown接口）后重新采集。

#### **查看OSPF路由表**：

```bash
show ip route ospf
```

记录OSPF学习的路由，用于分析路径选择。



### BGP数据采集

#### **查看BGP邻居状态**：

```bash
show ip bgp summary
```

记录BGP邻居状态和路由更新。

#### **查看BGP路由表**：

```bash
show ip bgp
```

记录BGP路径属性（如AS路径长度、local_pref）。

### **路径属性**：

- as_path_length：从show ip bgp中提取AS路径。

- local_pref：通过配置调整（默认100）：

- ```bash
  router bgp 100
   neighbor 10.0.12.2 route-map SET_LOCAL_PREF in
  route-map SET_LOCAL_PREF permit 10
   set local-preference 150
  ```

  

### 动态数据模拟

#### 模拟拥塞：

- 在模拟器中降低带宽（Packet Tracer可能不支持直接调整，可通过接口类型模拟，例如将GigabitEthernet改为FastEthernet）。

- 或者通过流量生成（如PC之间发送大量数据）模拟拥塞。

  

#### 模拟链路故障

```bash
interface GigabitEthernet0/0
 shutdown
```

关闭接口，观察OSPF邻居状态和BGP路由变化。



**采集延迟和丢包**：

- 使用ping测试延迟：

- ```bash
  ping 192.168.11.2
  ```

  Packet Tracer中无法直接模拟丢包，可通过外部工具（如GNS3）添加丢包。



### 数据导出

#### **日志导出**：

在路由器上启用日志：

```bash
logging 192.168.3.2  # 假设PC1为日志服务器
```

使用模拟器自带功能导出日志。

#### **脚本采集**：

如果使用GNS3，可以通过脚本（Python+Netmiko）自动采集：

```python
from netmiko import ConnectHandler

device = {
    "device_type": "cisco_ios",
    "ip": "192.168.1.1",
    "username": "admin",
    "password": "cisco",
}

connection = ConnectHandler(**device)
ospf_data = connection.send_command("show ip ospf neighbor")
bgp_data = connection.send_command("show ip bgp")
print(ospf_data)
print(bgp_data)
connection.disconnect()
```



## 结尾：

**网络规划**：

- 设计了一个包含3个AS（AS 100、AS 200、AS 300）的网络，共10个路由器，多个OSPF区域（Area 0、1、2、3）。
- AS之间通过eBGP互联，AS内部使用iBGP和OSPF。
- IP地址规划清晰，支持扩展。

**配置命令**：

- 提供了R1、R4、R8和R2的关键配置，包括接口、OSPF和BGP。
- 其他路由器可参考类似配置。

**数据采集**：

- OSPF：show ip ospf neighbor、show ip route ospf。
- BGP：show ip bgp summary、show ip bgp。
- 动态模拟：链路故障、拥塞、流量生成。