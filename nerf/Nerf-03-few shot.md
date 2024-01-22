# Nerf-03 ：解决视角稀疏问题

**1.Depth-Supervised NeRF (DS-NeRF): Deng K, Liu A, Zhu J Y, et al. Depth-supervised nerf: Fewer views and faster training for free[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 12882-12891.**

***

##### 2.Info-NeRF：Kim M, Seo S, Han B. Infonerf: Ray entropy minimization for few-shot neural volume rendering[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 12912-12921.

***

**3.Free-NeRF：Yang J, Pavone M, Wang Y. FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 8254-8263.**

***

**4.RegNeRF：Niemeyer M, Barron J T, Mildenhall B, et al. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 5480-5490.**

***

**理解如何在视角稀疏（也即问题欠定）情况下更好地进行网络优化。**

读完了领域内四篇 state-of-the-art的论文，总结出如下大致用来解决视角稀疏的方法：

​		1.信息补充：从未观测视角去采样，利用未观察视角的信息去进行训练，进行内容补充。或者采用 SFM 技术（如 COLMAP）去额外获得信息。

​		2. 从位置信息的维度/频率入手：频率正则化限制模型的输出在频域上的能量分布；退火算法来避免早期训练发散。

​		3. 损失函数【最常用】：通过对损失函数的优化或者限制来训练整个网络，如加入深度信息、射线熵最小化、遮挡正则化、几何正则化和颜色正则化等方法，都是通过向损失函数中加入限制来约束网络。

***

##### 第一篇：Depth-supervised nerf: Fewer views and faster training for free：加入深度监督

<font color=#9D1420>回答如下问题</font>【带着问题去思考】：

- DSNeRF如何渲染深度图？【在你的 NeRF 复现中实现深度图输出。<font color=#9D1420>(后续试图加入复现中) </font>】
- 理解深度监督是如何加入的？

<font color=#3F5965>阅读过程中笔记</font>:

#### 1.核心：

<u>1）解决了什么问题</u>：

​		1.Nerf 训练过程代价太高：昂贵的ray-casting操作 和 冗长的优化过程【拟合体渲染结果】。且需要大量的数据和相机姿势才能优化出较好的结果，且在view较少的情况下Nerf会过拟合。

​		2.在缺乏视野的一些情景下：一些方法采用了预测或者从训练集里面来抽取信息来填补，而本文的深度监督DSNerf只需要已经存在的3D关键点。

<u>2）主要贡献</u>：

​		1.使用structure-from-motion(SFM)技术：使用该技术替代Nerf获取图片和相机姿势的过程，该技术还将额外返回一些深度信息3D point clouds及其reprojection error，使用这些深度信息引导NeRF学习场景几何信息。

​		2.由深度监督的射线终止位置：加入深度监督模型和损失函数（depth-supervised distribution loss）来鼓励**射线终止位置的分布**去匹配3D关键点。

<u>3）实验效果如何</u>：	

​		该论文的两个目标：1.少量的图片/视角就能渲染出更好的收敛效果。2.达到更快的训练速度。

**1.少量的图片/视角就能渲染出更好的收敛效果：**

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231104105553657.png" alt="image-20231104105553657" style="zoom: 33%;" />

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231104105800106.png" alt="image-20231104105800106" style="zoom: 50%;" />

**2.达到更快的训练速度：**

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231104105912560.png" alt="image-20231104105912560" style="zoom: 67%;" />

有提速，在5-view时，达到Nerf的同效果可以提速3倍。10-view时是2倍。

**3.深度误差：**

![image-20231104110706491](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231104110706491.png)从表上可以看到，DSNerf恢复出来的深度的误差范围在10%左右，而 w/RGBD的情况下，深度的误差范围可以降低到5%。

#### 2. 具体method细节

**1.深度监督模型：：**

​		1）通过 SFM 包（如 COLMAP），得到相机矩阵 $(P_1,P_2,....)$. 以及捆绑得到的一系列 3D 关键点$\{X: x_1, x_2,.... \in R^3 \}$  和 可见性标志 $ j$  (该可见性标志所对应的关键点都可以被相机所看见： $X_j \subset  X$)。

​		2）对于给定的图片 $I_j$ 以及其相机矩阵 $P_j$，对于每个visible keypoints $x_i \in X_j$ **<u>估计</u>**其深度值 $D_{ij}$ : 将$ x_i $投影在 $P_j $上，则重投影值就是该关键点的深度值 。

​			1.由此可见：深度值$D_{ij}$ 是一个恒有噪音的估计量，所以对关键点 $x_i$ 的可靠性需要靠平均重投影误差 $\hat{\sigma}_i$ 来衡量。**这也是本文的一个局限性：效果上限取决于得到的深度准不准。**

​			2.具体做法：定义ray termination distribution，利用KL散度使ray termination distribution逼近COLMAP 估计的$D_{ij}$ 的分布。

➊ 对光线首次遇到的表面位置建模为一个随机变量$\mathbb D_{ij}$ ，该随机变量在COLMAP所估计的深度$D_{ij}$周围服从正态分布，方差为 $\hat{\sigma}_i$ ：$\mathbb D_{ij} \sim \mathbb{N}(D_{ij} ,\ \hat{\sigma}_i)$

➋首先，明确一下定义：**Ray termination distribution**

​		**Ideal Distribution**：假设一条光线发射出去，它会在光线的方向向量上的某一个点上终止，且有且仅有一个点，因此理论上来说是符合 $\delta$ 脉冲分布，本文认为这个分布应该是靠近<u>离光线最近的表面D</u>，所以这个分布应该是$\delta (t-D)$.  【$\delta$ 脉冲分布：特点是在除了零点以外的值都为零，而其在整个定义域上的积分等于1。】

​		**Ray termination distribution**：
$$
h(t) = T(t) \sigma(t) = e^{-\int_{0}^{t}\sigma( \boldsymbol s)ds }  \sigma(t)
$$

其中  $ \sigma(s)$ 是实数非负的函数，用来描述**密度 s** （density）从相机中心的分布。$T(t)$  是从0到t的积分，【详解见Nerf-01】  所以分布 $h(t)$ 是描述了沿射线采样的辐射对最终渲染值的加权贡献。

➌本文使用正态分布模拟深度的分布，目标是使ray distribution的分布 $h(t)$ 跟深度的分布（$\mathbb D_{ij} $）逼近，使用**KL散度**计算两个分布之间的距离及loss。

​				KL散度：又称为相对熵，衡量两个概率分布的相似性。表达式如下：[离散型 or 连续型]
$$
\begin{align}
D_{KL}(P||Q) &= H(P,Q) - H(P) \\
&=\sum_{i} P(x_i)log \frac{P(x_i)}{Q(x_i)} \\
&or = \int_x P(x)* [log \frac{P(x_i)}{Q(x_i)}] dx
\end{align}
$$
表示用概率分布Q来拟合真实分布P时，产生的信息损耗，其中P表示真实分布，Q表示P的拟合分布。

所以：$ \mathbb E_{  \mathbb  D_{ij}}KL[ \delta(t-  \mathbb  D_{ij}) || h_{ij}(t)]  = KL[  \mathbb  N_{ij}(D_{ij}, \hat{\sigma}_i) || h_{ij}(t)]+const$   

****

​	【什么意思？？如何理解这个期望？？？】

​	  在讲被渲染的color时，作者认为 rendered color 可以被表示成一个期望值：
$$
\hat{C} = \int_0^{\infty }h(t) \boldsymbol c(t)dt = \mathbb E_{h(t)}[\boldsymbol c(t)]
$$

​	  可以从连续随机变量的期望来理解，概率密度函数为$f(x)$的随机变量$X$，其函数的期望为$E(g(x)) = \int g(x)f(x) dx$. 其中积分区域由上文分析，是非负的，所以从0积分到无穷。

<font color=#9D1420>推导如下： </font>（论文经典跳步。。花了大量篇幅证明了$\int_0^{\infty }h(t)dt =1$, 也不愿说下这个KL怎么化简的。。 ）

令$KL[ \delta(t-  \mathbb  D_{ij})|| h_{ij}(t)] = g(t)$ , 且$\mathbb  D_{ij} $ 服从正态分布：$\mathbb D_{ij} \sim \mathbb{N}(D_{ij} ,\ \hat{\sigma}_i)$，设其概率密度函数为$P(D) $，【这是关于位置的函数，指分布在终点粒子附近表面可能终止的概率】则

$ \mathbb D_{ij} $  可以看做是随机变量X，来求其期望，所以相对于是对$P(D )$ 作为概率密度求积分
$$
\begin{align}
\mathbb E_{  \mathbb  D_{ij}}KL[ \delta(t-  \mathbb  D_{ij}) || h_{ij}(t)]  &= \mathbb E_{  \mathbb  D_{ij}}(g(t)) \\
&= \int_{-\infty}^{+\infty} g(t) P(D) dt

\end{align}
$$
其中$g(t)$ 由KL散度公式可以得为：
$$
g(t)=  \int_x P(x)* [log \frac{P(x_i)}{Q(x_i)}] dx \\
P(X) = \delta(t-  \mathbb  D_{ij}) , \ Q(x) =  h_{ij}(t)
$$

？？然后呢，查阅说 P(X) 与t无关可以省略 ？？？为啥？？

因为【冲激函数的“筛分”性质】

对于任意一个$t= t_0$ 的连续函数$f(t)$ ，有：$ \int_{-\infty}^{+ \infty} f(t) \delta(t-t_0) dt = f(t_0)$

所以并不是$P(X)$与t无关可以省略，而是该积分$\int P(x)dx$可以化为与t不相关的。


***

​	3）总体的loss为：$\mathfrak{L} = \mathfrak{L}_{color} + \lambda_{D}  \mathfrak{L}_{Depth}$ , 其中：
$$
\begin{align}
\mathfrak{L} _{Depth} &= \mathbb  E_{x_{i} \in X_j} \int logh(t) \  e^{- \frac{(t- D_{ij})^2}{2 \hat{ \sigma} _i^2}}dt \\
&\approx  \mathbb  E_{x_{i} \in X_j} \sum_k logh_k \  e^{- \frac{(t- D_{ij})^2}{2 \hat{ \sigma} _i^2}}\triangle t_k \\



\end{align}
$$


$$
\mathfrak{L} _{color} = \sum_{r \in R}||\hat{C}(r)-C(r)||_2^2
$$

​		$\lambda_{D}$ 是用于平衡两种损失的超参数。

然后把loss用于训练网络。

***

##### 第二篇：Ray entropy minimization for few-shot neural volume rendering :基于射线熵最小化的少视角神经辐射场

<font color=#3F5965>阅读过程中笔记</font>:

#### 1.核心：

<u>1）解决了什么问题</u>：

​	解决Nerf在视角稀疏的情况下，产生的过拟合、退化问题。

<u>2）主要贡献（创新点）</u>：

​	1.提出了规范正则化：射线熵最小化。即对每条射线密度施加熵约束来最小化稀疏视角带来的潜在重建不一致性，并且利用了不可见视图的光线抽样点。

​	2. 使用KL散度去约束相近的条射线之间的密度，以防止相似视角带来的过拟合。

<u>3）实验效果如何</u>：	

![image-20231109143959243](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231109143959243.png)

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231109144145337.png" alt="image-20231109144145337" style="zoom:50%;" />

射线熵和KL都带来了一些提升，但是很明显相对熵的作用更明显一些，两个熵共同作用下效果更好。

#### 2.具体模型：

​	核心思路：通过使用额外的正则项来最小化每条射线密度函数的熵。

1）归一化射线密度公式：
$$
p(\bold r_i) = \frac{\alpha_i}{\sum_j \alpha_j} = \frac{1-e^{-\sigma_i \delta_i}}{\sum_j 1-e^{-\sigma_j \delta_j}}
$$
其中 $\bold r_i \ (i=1,...,N)$是射线上的抽样点，$\sigma _i$ 是观测密度，也就是指不透明度。$\delta _i$是指的抽样间隔。

2）射线熵 ：
$$
H(\bold r) = - \sum_{i=1}^{N} p(\bold r_i) log \ p(\bold r_i) 
$$
3）在射线熵最小化中，最关键的一个问题是：那些没有击中任何物体的射线仍有最低熵的强制存在。为了防止潜在的伪影影响，本文简单的忽略了这些密度低的射线：

定义累计射线密度为：
$$
Q( \bold r) = \sum_{i=1}^N 1- e^{-\sigma_i \delta_i}
$$
所以，定义了a mask variable $M(·)$：
$$
M(\bold r) = \begin{cases}
 1 & \text{ if } Q( \bold r) > \epsilon , \\
 0 & \text{ otherwise} 
\end{cases}
$$
4）损失函数为：
$$
\mathcal{L}_{entropy} = \frac{1}{|\mathcal R_s|+|\mathcal R_u|} \sum_{r \in \mathcal R_s \cup \mathcal R_u} M( \bold r)\odot H( \bold r)
$$
$ \mathcal R_s $ 表示来自训练集的可视射线，$\mathcal R_u$ 表示从不可视图片中的随机抽样射线集，$\odot$ 表示逐元素相乘。

因为Nerf不能使用不可视图片的光线，因为缺乏对像素RGB的真实标定。但是熵正则化不需要真实标定，所以可以使用不可视的射线。

5）由于训练图像若有比较相似的视角，则会导致过拟合和退化问题。为了避免这个问题，采取KL散度去约束相近的条射线之间的密度。

给定一个观察光线 $r $ , 获取其围绕镜头旋转 $-5^{\circ} $ 到 $5^{\circ} $ 范围之间的光线 $\widetilde{r} $，得到：
$$
\mathcal{L}_{KL} = D_{KL}(P(r) || P(\widetilde{r} ))
$$
KL散度是一种统计学度量，表示的是一个概率分布相对于另一个概率分布的差异程度。若两者差异越小，KL散度越小，反之亦反。当两分布一致时，其KL散度为0。

6）总loss：
$$
\mathcal{L}_{total} = \mathcal{L}_{RGB } +\lambda_1\mathcal{L}_{entropy} +\lambda_2 \mathcal{L}_{KL}
$$

***

##### 第三篇：Improving Few-shot Neural Rendering with Free Frequency Regularization：自由频率正则化

<font color=#3F5965>阅读过程中笔记</font>:

#### 1.核心：

<u>1）解决了什么问题</u>：

​	1.解决了在视角稀疏的情况下，Nerf处理位置信息是采用的高维映射这种方式带来的过拟合、细节丢失等影响。

​	2.解决了遮挡情况下nerf中相机附近密集浮动物体。

<u>2）主要贡献</u>：

​	1.提出了频率正则化，NeRF模型可以在保持场景细节的同时减少过度拟合的风险，从而提高模型的泛化能力和表达能力。具体实现方式可以是通过限制频率范围或对频率进行约束，将模型的输出频率成分被限制在一定的范围内。

​	2.提出了遮挡正则化：用于处理遮挡现象，即场景中某些物体遮挡了其他物体的情况。它通过引入遮挡相关的损失函数，如遮挡一致性损失或深度遮挡损失，来约束模型生成的图像中的遮挡关系。

<u>3）实验效果如何</u>：	

![image-20231110155508019](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231110155508019.png)

![image-20231110155537123](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231110155537123.png)

#### 2.具体模型：

核心思路：

1）在Nerf里有一个核心关键：输入进mlp的位置信息是进行了高维的映射，因为神经网络很难从低纬度的信息里学习信息。而在视角稀疏的场景中，由于对相关几何图像学习较少，NeRF对易感噪声更敏感，高纬度的信息可能会导致过度拟合和细节丢失的问题。因此这是高频分量是少镜头神经渲染中产生误差的主要原因。 

**2）引入：频率正则化。**

 频率正则化的核心思想是通过限制模型的输出在频域上的能量分布，减少高频成分的影响。

 在NeRF中，频率正则化通过引入一个频率掩码来实现，其中在训练过程中逐渐减小高频部分的权重，从而限制模型在高频区域的输出变化。通过调整频率掩码的形状和衰减速度，可以灵活地控制模型的输出在不同频率上的平滑程度。

【$ \alpha$ 的定义是increasing frequency mask：其中mask的含义：掩膜、掩码。是指一个类似遮挡板的功能，图像矩阵和另外一个“遮挡板”矩阵进行乘积运算，从而可以得到想要的结果。可以用来提取特征或者屏蔽相关的像素。】

这个频率掩码是一个与模型参数相关的函数，它在训练的不同阶段逐渐减小高频部分的权重。掩码 $ \bold \alpha_i$ 的表达式如下：
$$
\pmb{ \alpha} _i(t,T,L) = \begin{cases}
1  & \text{ if } i \le \frac{t·L}{T} +3 \\
\frac{t·L}{T} - \left \lfloor \frac{t·L}{T}   \right \rfloor   & \text{ if } \frac{t·L}{T}+3 < i \le \frac{t·L}{T}+6 \\
0  & \text{ if } i > \frac{t·L}{T} +6
\end{cases}
$$

其中：i表示位置，t 表示当前训练迭代，T表示频率正则化的最终迭代。由公式可得，随着每次训练的开始，可见频率线性增加3bit。

$\gamma $  函数是用来映射空间$R$ 到高维的 $R^{2L}$ :
$$
\gamma(x) = (sin(2^{0}\pi x),cos(2^{0}\pi x),....,sin(2^{L-1}\pi x),cos(2^{L-1}\pi x)).
$$

所以本论文中应用到的实际的输入为：
$$
\gamma'_L(t,T;x) = \gamma_L(x) \odot \pmb \alpha  (t,T,L)
$$


**3）引入：遮挡正则化**

由于训练视角有限以及问题的不适定性（如当相机附近存在少量训练样本未覆盖到的区域时），新视角中可能仍会存在某些特征性伪影。这些错误信息通常表现为位于相机极近位置的“墙壁”或“浮动物”。

遮挡正则化：用于减少神经辐射场模型中相机附近密集浮动物体。

具体思路： 

​	遮挡正则化引入了一个二进制掩码向量，用于确定哪些点需要受到惩罚。在这个向量中，我们将与相机距离较近的前几个点（即正则化范围内的点）设置为1，其余点设置为0。这样，模型在训练过程中会被约束，尽量减少相机附近的密集浮动物体的生成。
$$
\mathcal{L}_{occ} = \frac{\sigma^T_K ·m_k}{K} = \frac{1}{K} \sum_{K} \sigma_K ·m_k
$$
其中 ：$ \sigma_K $表示沿射线采样的$K$个点的密度值，按照靠近原点（从近到远）的顺序。$m_k$是一个二值mask向量，决定一个点是否会被penalized， 为了减少相机附近的固体漂浮物，我们将 $m_k$的值设置为索引 $ M $，称为正则化范围，为 1，其余为 0。
遮挡正则化的实现方法简单且易于计算，在训练过程中可以直接应用该损失函数，从而提高神经辐射场模型的性能。


***

##### 第四篇：Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs：正则化

<font color=#3F5965>阅读过程中笔记</font>:

#### 1.核心：

<u>1）解决了什么问题</u>：

​	当可用视角稀疏的时候，NeRF由于估计场景几何形状中的错误和训练开始时的发散行为造成不好的效果影响。

<u>2）主要贡献</u>：

​	1.一个patch-based的正则化器，用于从不可视的视角渲染的深度，这些补丁可以被正则化以产生平滑的几何图形和高性能颜色。并提出了几何正则化和颜色正则化，来约束生成的图形和色彩。
​	2.一种annealing策略，用于沿着光线采样点，首先在一个小范围内采样场景内容，然后扩展到完整的场景边界，以防止在训练早期发散。

<u>3）实验效果如何</u>：	

​	有提升，但没有很显著。

![image-20231110200921962](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231110200921962.png)

#### 2.具体模型：

核心思路：

​	**1）模型：**

​		清晰明了的算法流程图：

![image-20231110200804161](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231110200804161.png)

​	我们提出采样未观测到的视图(红色摄像机)，并将这些视图渲染的petch通过几何正则化、颜色正则化来优化外观。

​	1.如何选取未观测的点？

​			1）先定义未观测点集的边界：

​					一组已知的目标姿势为：$ \{ \pmb P_{target}^i   \}:P_{target}^i = [R_{target}^i|t_{target}^i]  $

​			2）空间位置和旋转角的采样区间取值：

​					$S_t = \{ t\in R^3 | t_{min} \le t \le t_{max} \}$ ，其中$t_{min}$ 和$t_{max}$ 是上面已知边界的$\{ \pmb t_{target}^i \}$ 的最大值和最小值。

​					$S_R |t = \{ R(\overline{p}_u, \overline{p}_f+ \epsilon,t) |\epsilon \sim N(0,0.125)     \}$ ，其中，$R(·)$ 表示相机旋转的样本空间，$\epsilon $ 表示抖动信号值，$\overline{p}_u $表示目标姿势的归一化计算出的平均上限轴心，$\overline{p}_f$表示与目标姿势的轴具有最小平方距离的点。

​				so：未观测的点取值区间为：
$$
S_{P} = \{[R|t] | R \sim S_R|t , t \sim S_t  \}
$$
​	2.几何正则化

​		1）几何正则化：通过在训练过程中引入几何相关的损失函数，来约束生成的几何结构，这样可以防止模型在生成过程中出现不合理的几何形状。

​		2）采用：深度平滑损失函数。它通过计算深度图的梯度，并对梯度进行正则化损失的最小化，以确保深度值的连续性和一致性。这意味着在相邻的像素之间，深度值的变化应该是平滑的，不应该出现剧烈的跳变或不连续性。

​	3.颜色正则化

​		1）颜色正则化：关键思想在于估计渲染patch的可能性，并在优化期间最大化它。利用现成的非结构化2D图像数据集。

​		2）最大似然估计：具体而言，它使用一个训练好的归一化流模型对预测的RGB图像进行编码，并通过最大化预测的对数似然来优化外观的一致性。

**2）抽样空间模拟退火**

​	1.该退火算法的思路同时用于早期的观测到和未观测到的数据训练，用于有效避免梯度消失等现象。

​	2.思路：将场景采样空间限制为所有输入图像定义的较小区域，首先在一个小范围内采样场景内容，然后扩展到完整的场景边界。
