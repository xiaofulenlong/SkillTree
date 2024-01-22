# Nerf-04：动态场景NeRF（dynamic）

​		**1.D-NeRF：Pumarola A, Corona E, Pons-Moll G, et al. D-nerf: Neural radiance fields for dynamic scenes[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 10318-10327.**

​		**2. Hdr-nerf：Huang X, Zhang Q, Feng Y, et al. Hdr-nerf: High dynamic range neural radiance fields[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 18398-18408.**

​		**3. NR-NeRF：Tretschk E, Tewari A, Golyanik V, et al. Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 12959-12970.**

***

**第一篇：D-Nerf :  Neural radiance fields for dynamic scenes** （引入宏观时间信息）

<font color=#9D1420>回答如下问题</font>【带着问题去思考】：

+ 理解 deformation 网络如何建模物体的移动。你认为此文章的存在什么问题？

<font color=＃0000FF >阅读过程中笔记</font>:

#### 1.核心：

<u>1）解决了什么问题</u>：

​	NeRF只能重建静态场景，论文提出的方法可以把神经辐射场扩展到动态领域，可以在单相机围绕场景旋转一周的情况下重建物体的刚性和非刚性运动。

<u>2）主要贡献</u>：

​	把时间作为附加维度加到输入中，同时把学习过程分为两个阶段：第一个把场景编码到规范空间，另一个在特定时间把这种规范表达形式map到变形场景中。两个map都用mlp学习，训练完后可以通过控制相机视角和时间变量达到物体运动的效果。

<u>3）实验效果如何</u>：

![image-20231126202720399](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231126202720399.png)

<u>4）存在的问题：</u>：

​	引入时间t作为维度，而t在其官方提供的数据集中取值范围是[0,1]的浮点数，对于每一个相机视角而言需要新增一个时间t参数，那么该参数应该如何标注？官方并没有给出回答。

​	经查阅，有人提到是手工标注，啊这

#### 2. 具体method细节

**1.规范网络canonical Network和变形网络deformation Network：**拆分$(x,y,z, \theta,\phi,t) \to (rgb,\sigma )$ 为：

​		1）deformation Network $\Psi_t$ :
$$
\Psi_t(x,t) = \begin{cases}
\triangle x,  & \text{ if } t\ne  0 \\
 0, & \text{ if } t=0
\end{cases}
$$
​			可以认为是在不同时刻将 $\Psi_t:(x,t) \to \triangle x $

​		2)  canonical Network  $\Psi_x$ : 可以理解为原始nerf

​			将由上变形网络得到的$(\triangle x,\triangle y,\triangle z)$送入网络，即：$\Psi_x: (x+\triangle x,y+\triangle y,z+\triangle z, \theta,\phi) \to (rgb,\sigma )$.

2.加入时间t参数的体渲染 :

<div align=center><img src=" /Users/hurenrong/Library/Application Support/typora-user-images/image-20231126204331341.png" width=400"  "></div>

3.损失函数
$$
\mathfrak{L}  =\frac{1}{N_s} \sum_{i=1}^{N_s}||\hat{C}(p,t)-C(p,t)||_2^2
$$


***

**第二篇：Hdr-nerf : High dynamic range neural radiance fields**

<font color=＃0000FF >阅读过程中笔记</font>:

#### 1.核心：

<u>1）解决了什么问题</u>：

​	1.任务：从一组具有不同曝光（**曝光定义为曝光时间和辐射度的乘积**）的LDR（低动态范围Low Dynamic Range）图像中恢复高动态范围神经辐射场。

​	2.目前nerf中所获得的辐射动态范围仍在一个低动态范围里(0,255)，而现实的物理世界范围更高$(0,+\infty) $.本论文解决了该问题，从辐射场中渲染恢复了高动态范围。

​	3.解决了传统重建HDR由固定姿势产生伪影的问题。

<u>2）主要贡献</u>：

​	1.提出了对物理成像过程的建模：相机响应函数建模。

​	2.提出了端到端方法Hdr-nerf：场景点的辐射度通过两个隐式函数（辐射场、色调映射器）转换为LDR图像中的像素值。

<u>3）实验效果如何</u>：	

![image-20231126222111873](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231126222111873.png)

#### 2. 具体method细节

​	核心：是对捕获场景亮度并将它们映射到像素值的过程进行建模。

​	1.将场景点的辐射度通过两个隐式函数（辐射场、色调映射器）转换为LDR图像中的像素值：

​		1）**辐射场**：对场景辐射度进行编码(值从0到+∞变化)，通过给出 相应的射线源和射线方向 输出射线的密度和辐射度。

​		2）**色调映射器**：对映射过程进行建模，射线照射到相机传感器上成为像素值。通过将**辐射度**和**相应的曝光时间**输入到色调映射器来预测光线的颜色。

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231126223153158.png" alt="image-20231126223153158" style="zoom:50%;" />

​	2.体渲染：使用传统体渲染技术。

​	3.损失函数

​		仅使用输入的LDR的图像作为监督。

​		总的loss 为 : $\mathfrak{L} = \mathfrak{L}_{color} + \lambda_{D}  \mathfrak{L}_{u}$

***

第三篇：NR-NeRF：Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video

<font color=＃0000FF >阅读过程中笔记</font>:

#### 1.核心：

<u>1）解决了什么问题</u>：

​	1.仅需要动态场景的单目视频作为输入（不需要多视图输入）

​	2.可以渲染动态体积的变形或非变形部分。

<u>2）主要贡献</u>：

1. 使用两个组件表示非刚性场景：（1）用于捕获几何和外观的规范神经辐射场；（2）场景变形场。
2. 场景变形是通过光线弯曲实现的，光线弯曲由MLP进行建模。
3. 为每个点分配刚度得分，允许变形不影响场景的静态区域。
4. 引入多个正则化项作为额外的软约束：（1）限制体积内变形的大小；（2）保持局部形状。

<u>3）实验效果如何</u>：	

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231126232441255.png" alt="image-20231126232441255" style="zoom:60%;" />

#### 2. 具体method细节

​	不同于传统nerf经典流程的特点是：

1）Deformation Model

2）loss函数：引入了多个正则化项来额外约束体积内变形的大小，以此来保持局部形状。
