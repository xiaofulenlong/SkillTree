## **01的论文代码实现**

**Mildenhall B, Srinivasan P P, Tancik M, et al. Nerf: Representing scenes as neural radiance fields for view synthesis[J]. Communications of the ACM, 2021, 65(1): 99-106.** 

***

#### 主要逻辑框架

​		1）数据读入：

​				1.数据集类型：'llff'、'blender'、'LINEMOD'、'deepvoxels' 四种类型的数据集。

​					先实现：属于'blender'类型的lego数据集。

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231030101216103.png" alt="image-20231030101216103" style="zoom: 50%;" />

​				 2.读文件数据：

​					1）传入：文件夹路径

其中数据的json文件内容：

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231111151320190.png" alt="image-20231111151320190" style="zoom: 33%;" />

camera_angle_x ：相机水平视场角

frames ：若干字典组成的列表，存放了若干个：

```text
{
    "file_path": "./train/r_0",  对应的图片的地址
    "rotation": 0.012566370614359171, 【？？干什么的？】
    "transform_matrix": [...   pose信息
    ]
}
```

​				2）传出：

1. .png ：由给出的file_path得到，是图像文件
2. pose：由transform_matrix得到
3. 对应每个png的对应的H、W、camera_angle_x的值

​			2）向量构造：读入的数据数组，重建为 5D 向量 

​			3）高维映射，位置编码：对$X $（x,y,z）Position进行 3-> 60 的高维编码，对 viewing direction($\theta, \phi $)向量 进行抬到24维的映射。

​		实际演变过程：

公式为：$\gamma $  函数是用来映射空间$R$ 到高维的 $R^{2L}$.

$$
\gamma(p) = (sin(2^{0}\pi p),cos(2^{0}\pi p),....,sin(2^{L-1}\pi p),cos(2^{L-1}\pi p)).
$$
p 是 3维的，【sin，cos】是扩展了两个维度，从0~L-1 是扩展了L次。所以总扩展维度为 2 * 3 * L。

如：p = **torch**.**rand**([3,4,5]) ，那么在经过如上的扩展之后应该变成什么形状的结构呢？

​			4）如何转化为真实渲染的图片的？：**<u>体渲染</u>**

生成光线的步骤是 NeRF 代码中最为关键的一步，实际上我们模拟的光线就是三维空间中在指定方向上的一系列离散的点的坐标。有了这些点坐标，我们将其投入到 NeRF 的 MLP 神经网络中，计算这个点的密度值以及颜色值。

​				<u>1.如何采样？【两次采样】</u>

​					1）粗采样：先随机均匀的抽取样本 $N_c$ 个，然后归一化，可以得到概率密度函数$w_i$ ,便于细采样时得到更精确的采样。

​					2）细采样：在得到的精确的概率密度上再次进行采样，获得更精细的 $N_f$ 个采样点。	

​				<u>2.如何渲染？【mlp】</u>

​				Nerf的核心贡献：mlp，采用深度学习神经网络来学习光线渲染。

1）Input： 5D vector-valued function : 3D location $X = (x,y,z)$  and  2D viewing direction ($\theta, \phi $) .

2）Output:  an emitted color $C=(r,g,b)$  and volume density $\sigma$ [体积密度，或者说透明度].

将以上处理好的映射数据送入mlp：使用MLP网络得到$ F_{\Theta}:(X,d) \rightarrow (c,\sigma)$ . 其中：通过限制神经网络，只让 location $X=(x,y,z)$ 控制 volume density $\sigma$ 的预测，让 location $X $  和 viewing direction ($\theta, \phi $)一起预测color C.    <u>MLP网络具体步骤如下：</u>

![截屏2023-10-19 14.42.08](/Users/hurenrong/Library/Application Support/typora-user-images/截屏2023-10-19 14.42.08.png)

​	（1）使用8个全连接层来处理输入的3D  location $X=(x,y,z)$ 【使用ReLU激活和每层256个通道】，然后输出σ和256维特征向量；

​	（2）将该特征向量与相机2D viewing direction ($\theta, \phi $) 拼接，然后使用全连接层【使用ReLU激活和128通道】进行处理，得到输出的RGB颜色值。

​	（3）从mlp中得到的颜色 $c$ 和体密度 $\sigma$. 这两个参数就是一条光线上一个采样点对应的粒子密度和颜色值。

​	（4）在整个训练过程中，收集所有光线上所有采样点的粒子密度和颜色，渲染出图像。

​				5）再根据体渲染得到预测的投影视图，然后和真实的投影视图计算 loss 来训练网络。【神经网络train的过程】				

***

#### 遇到的问题

​	1）数据集格式是什么？

​			step1.查阅博客：创建llff格式的数据集，以llff格式的数据输入，那么这个格式的数据具体内容是什么？

​			step2.不对，数据的格式应为json，结构如图： 

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231030101216103.png" alt="image-20231030101216103" style="zoom: 50%;" />

​	2）粗采样和细采样之间的衔接是怎么样的？细采样是在粗采样训练mlp之后结束后进行的吗？还是细采样和粗采样是同步进行，一起喂给mlp？mlp和采样之间的关系？

​		 解决：1.mlp网络输出的 $\sigma $和 $\hat{c}$ 是一条光线上一个采样点对应的粒子密度和颜色值。在整个训练过程中，会收集所有光线上所有采样点的粒子密度和颜色，再根据体渲染得到预测的投影视图。

​					2.根据论文粗采样、细采样和mlp之间的训练关系应如下所示：

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231102170239469.png" alt="image-20231102170239469" style="zoom: 50%;" />

***

（3）关于生成光线、相机参数

​			我们要做的是：生成每个方向下的像素点到光心的单位方向( z轴为单位1)，通过这个单位方向，可以通过调整z轴的坐标来生成空间中每一个点坐标，借此模拟出一条光线。这个射线是怎么构造的。**给定一张图像的一个像素点，我们的目标是构造以相机中心为起始点，经过相机中心和像素点的射线。**该像素点就是位于成像平面的像素点。

​	==1.为了唯一地描述每一个空间点的坐标以及相机的位置和朝向，我们需要先定义一个世界坐标系。==

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231124160447282.png" alt="image-20231124160447282" style="zoom:50%;" />

​		其中：

​		1）**相机的位置和朝向：由外参决定【外参 World-To-camera，w2c】。**

​			外参：4*4的矩阵M，作用是将世界坐标系的点$P_{world} = [x,y,z,1]$变换到相机坐标系$P_{camera}$下：
$$
P_{camera} = MP_{world}
$$
​			相机外参的==逆矩阵==被称为**camera-to-world (c2w)矩阵**，左上角3x3是旋转矩阵R，右上角的3x1向量是平移向量T：
$$
c2w = \begin{bmatrix}
R & T \\
0 & 1
\end{bmatrix}
=\begin{bmatrix}  \begin{array}{ccc|c}
 r_{11} & r_{12} & r_{13} & t_1\\
 r_{21} & r_{22} & r_{23} & t_2\\
 r_{31} & r_{32} & r_{33} & t_3\\
 											\hline
  0& 0 & 0  &1
\end{array}
\end{bmatrix}
=\begin{bmatrix}
X & Y & Z & O \\
\hline
0 & 0 & 0 & 1
\end{bmatrix}
$$
​			旋转矩阵**R**的第一列到第三列分别表示了相机坐标系的X, Y, Z轴在世界坐标系下对应的方向；

​			平移向量**T**表示的是相机原点O在世界坐标系的对应位置。

​			<u>外参由数据集中['transform_matrix']得到。</u>

​		2）**投影属性：由内参决定。**

​			内参：3*3的矩阵K，作用是将相机坐标系下的3D坐标映射到2D的图像平面：
$$
K = \begin{bmatrix}
f_x &0 &c_x \\
0 &f_y &c_y \\
0 &0 &1
\end{bmatrix}
$$
​			$f_x$和$f_y$是相机的水平和垂直**焦距**（对于理想的针孔相机，$f_x = f_y$）。焦距的物理含义是相机中心到成像平面的距离，长度以像素为单位。

​			$c_x$和$c_y$是图像原点相对于相机光心的水平和垂直偏移量。可以用图像宽和高的1/2近似。

​			<u>内参由焦距、图像的宽、高得到。</u>		

​	==2.Nerf：NeRF所做的是在相机坐标系下构建射线，然后再通过camera-to-world (c2w)矩阵将射线变换到世界坐标系。==

​		step1 ：写出相机中心、像素点在相机坐标系下的3D坐标

​		step2 ：使用c2w矩阵变换到世界坐标系上去

过程如下：

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/草稿纸-4.jpg" alt="草稿纸-4" style="zoom:25%;" />

***

（4）关于体渲染的离散公式推导与代码实现

将$[t_{n},t_{f}]$ 均分成为N份，然后从每份里面随机均匀的抽取样本。
$$
t_{i} \sim u[t_n+\frac{i-1}{N}(t_{f}-t_n),t_n+\frac{i}{N}(t_{f}-t_n)]
$$
所以上（2）式就可以离散化为：
$$
\hat{C}(r) =  \sum_{i=1}^{N}T_{i}(1-e^{-\sigma_{i} \delta_{i} })\bold c_{i} \\
T_i= e^{- \sum_{j=1}^{i-1}\sigma_{j} \delta_{j}} \\
$$
![image-20231129104231623](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231129104231623.png)
