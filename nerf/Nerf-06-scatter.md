## 散射与去雾

​	1.**SeaThru-NeRF：Levy D, Peleg A, Pearl N, et al. SeaThru-NeRF: Neural Radiance Fields in Scattering Media[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 56-65.**

​	2.**Dehaze NeRF： Chen W T, Yifan W, Kuo S Y, et al. Dehazenerf: Multiple image haze removal and 3d shape reconstruction using neural radiance fields[J]. arXiv preprint arXiv:2303.11364, 2023.**

​	3.**Dehazing NeRF：Li T, Li L U, Wang W, et al. Dehazing-NeRF: Neural Radiance Fields from Hazy Images[J]. arXiv preprint arXiv:2304.11448, 2023.**

​	4.**NeRF‐Tex：Baatz H, Granskog J, Papas M, et al. NeRF‐Tex: Neural Reflectance Field Textures[C]//Computer Graphics Forum. 2022, 41(6): 287-301.**

​	5.**Ultra-NeRF：Wysocki M, Azampour M F, Eilers C, et al. Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging[J]. arXiv preprint arXiv:2301.10520, 2023.**

​	6.**NeRF-Det**：**Xu C, Wu B, Hou J, et al. Nerf-det: Learning geometry-aware volumetric representation for multi-view 3d object detection[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 23320-23330.**

​	7.**Nerrf：Chen X, Liu J, Zhao H, et al. Nerrf: 3d reconstruction and view synthesis for transparent and specular objects with neural refractive-reflective fields[J]. arXiv preprint arXiv:2309.13039, 2023.**

​	**8. DoF-NeRF：Wu, Z., Li, X., Peng, J., Lu, H., Cao, Z., & Zhong, W. (2022). DoF-NeRF: Depth-of-Field Meets Neural Radiance Fields. Proceedings of the 30th ACM International Conference on Multimedia.** 

​	**9.Zhang H, Lin Y, Teng F, et al. Circular SAR Incoherent 3D Imaging with a NeRF-Inspired Method[J]. Remote Sensing, 2023, 15(13): 3322.**

​	**10.Ronen R, Holodovsky V, Schechner Y Y. Variable Imaging Projection Cloud Scattering Tomography[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.**

***

​	了解如何考虑散射介质？

***

**笼统总结**：处理散射介质的方法一般是将介质和场景单独设置通道进行参数优化，分开渲染分别优化，然后再将结果合并统一做体渲染。并且会在损失函数上设置一些正则化来进行对颜色、不透明度的控制。其中正则化或者对介质的优化多采用物理去雾模型（DCP、大气散射模型等）。

***

**第一篇：SeaThru-NeRF：Neural Radiance Fields in Scattering Media**

##### 1.总结：

​	<u>1）创新点？</u>

​		1.提供了 NeRF 的重要扩展，可以渲染在雾霾、雾气和水下等散射介质中获取的场景。【出发点：nerf提供了体渲染的框架，但是没有考虑到介质】

​		2.SeaThru-NeRF目的在于对退化进行建模，恢复其参数，并且重建干净的底层场景和新颖视图。首先在nerf框架内为**场景**和**介质**分配单独的参数（颜色和通道）来在渲染 。【这样色彩恢复就好像它们不是通过介质成像一样，因为建模允许将对象外观与介质效果完全分离。】

<u>	2）效果</u>

![image-20231213023844044](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231213023844044.png)



​	<u>3）局限性</u>

1.需要预先提取相机姿势，所以在能见度较差的情况下比较难还原，并且nerf的优势来自于介质的建模，当在不符合模型假设的场景时就表现不佳。

2.没有考虑多重散射或者人工照明。

#### 2. 核心：method

​	**1.散射介质中的图像形成**

雾、霾或水下的图像形成与晴朗空气中的图像形成有两个主要方面的不同。

​	1）首先，从物体发出的直接信号会随着距离和波长的变化而衰减。

​	2）其次，该信号被反向散射（也称为路径辐射或遮蔽光）遮挡。

​		反向散射是由于沿视线的粒子的内散射而增加的辐射。 遮挡后向散射层的强度和颜色与场景内容无关，其强度沿着视距累积，随着距离的增加而增加。结果就会使更远物体的可见度和对比度显着降低并且颜色扭曲。

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231213030254929.png" alt="image-20231213030254929" style="zoom: 67%;" />

​	**2.nerf模型结合**

​		如下公式所示，seaThru-Nerf是整体思路其实与基础Nerf没什么差别，就是对物体和介质使用单独的颜色和密度参数，分别进行预测、积分求和渲染。

​	<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231213152139586.png" alt="image-20231213152139586" style="zoom: 50%;" />



1.<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231213102503731.png" alt="image-20231213102503731" style="zoom: 50%;" />



<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231213152813376.png" alt="image-20231213152813376" style="zoom: 67%;" />

***

**第二篇：Dehazenerf: Multiple image haze removal and 3d shape reconstruction using neural radiance fields**

#### 1.总结

​	<u>1）创新点：</u>

​	1.通过基于物理的 3D 雾霾图像形成模型扩展 NeRF 的体积渲染方程，以准确模拟雾霾条件下普遍存在的散射现象。也就是指在场景的3D形状和雾参数之间建立了联合学习。

​	2.引入多个物理启发的归纳偏差和优化正则化器来有效地消除表面外观的歧义，从而仅使用模糊图像作为输入实现准确的清晰视图外观和几何重建。

​	<u>2）效果：</u>

**去雾成像结果如下所示：**

![image-20231213164224284](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231213164224284.png)

整体质量也不错：

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231213164924708.png" alt="image-20231213164924708" style="zoom:50%;" />

####  2. 核心：method

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231217142201656.png" alt="image-20231217142201656" style="zoom:60%;" />

如图为整个DehazeNeRF architecture。由我自己的理解，根据论文公式来推理一下如何由输入的雾图得到清晰图的。

架构由两个通道：haze module (yellow) 和 surface module (gray)

​	1.黄色通道是基于物理的模型，根据渲染公式来生成结果，然后用来训练和优化$c_s$ 和 $\sigma_s$ ,这两个参数用于公式计算$C_{haze}$. 公式就不推了，和体渲染的公式没什么本质上的区别。

​	2.在灰色的通道（现有的Nerf通道）上，先采用有符号距离函数SDF对光线$r(t)$进行了一个参数化，然后随后和原始Nerf相同的方式送入神经网络，得到$c_{surface}$和$\sigma$.这两个参数也会在训练阶段在黄色通道进行反向的优化，用来计算$C_{surface}$.

​	3.得到回复后的图像的RGB值 $C_{clear}$，根据体渲染公式（ Eq(9)）：

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231217150726212.png" alt="image-20231217150726212" style="zoom:50%;" />

​	而这其中的颜色c是由两部分构成的 : $C = C_{haze}+C_{surface}$.

​	4.优化：采用多个物理启发的归纳偏差和优化正则化器

（1）采用了Koschmider model，这个模型广泛应用于基于图像的单视和多视点去雾方法。

​			引入了深度值：$D(r) =  \sum_{n=1}^{N} T_\sigma^n \alpha^n t^n$.

​			$C(r) = C_{\text{clear}}(r) \exp\left(-\bar{\sigma}_s D(r)\right) + \bar{c}_s \left(1 - \exp\left(-\bar{\sigma}_s D(r)\right)\right)$

​			$C_{\text{surface}}(r) \approx C_{\text{clear}}(r) \exp(-\bar{\sigma}_s D(r)) = \widetilde{C}_{\text{surface}}(r) \quad $

​			$C_{\text{haze}}(r) \approx \bar{c}_s(1 - \exp(-\bar{\sigma}_s D(r))) = \widetilde{C}_{\text{haze}}(r)$

所以得到**一项损失loss**：$L_{2D} = \left\| C_{\text{surface}}(r) - \widetilde{C}_{\text{surface}}(r) \right\|_1 + \left\| C_{\text{haze}}(r) - \widetilde{C}_{\text{haze}}(r) \right\|_1 + \left\| C - \widetilde{C}_{\text{surface}}(r) - \widetilde{C}_{\text{haze}}(r) \right\|_1$

（2）基于DCP（Dark Channel Prior），提出了另一项正则化：

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231217154923275.png" alt="image-20231217154923275" style="zoom:50%;" />

​		 DCP用于估计场景的清晰视图颜色RGB值(Koschmieder定律)，因为模糊图像中严重衰减的颜色可以用雾来解释，也可以用暗淡的表面颜色来解释。 



***

**第三篇：Dehazing NeRF： Neural Radiance Fields from Hazy Images** 

#### 1.总结

<u>1.贡献/创新点：</u>

- 提出了一种新的无监督视图合成框架，该框架可以从模糊输入中恢复清晰的NeRF。利用三维场景的深度信息来补充ASM的不确定参数，解决了单幅图像去雾的不适定问题。
- 为了在保证重建图像一致性的同时，缓解模糊图像量化带来的信息丢失问题，提出了一种软边缘一致性正则化、大气一致性和对比度判别损失等方法。

<u>2.实验结果：</u>

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231217160302452.png" alt="image-20231217160302452" style="zoom:50%;" />

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231217160324792.png" alt="image-20231217160324792" style="zoom:40%;" />

#### 2.method 核心

**（1）大气散射模型（AMS模型）**

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231217160715233.png" alt="image-20231217160715233" style="zoom:50%;" />

​	雾霾天气图像退化的两个原因：

​		首先，物体反射的光被悬浮粒子吸收和散射并衰减。

​		其次，阳光等环境光被悬浮粒子散射形成背景光。

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20231217160914325.png" alt="image-20231217160914325" style="zoom:50%;" />

**I(x)为观测到的模糊图像，J(x)为待恢复的清晰图像**

t(x)为透射图，A为大气光或空气光，β为大气散射系数，d(x)为目标物与相机之间的距离。其中(x)表示逐像素计算。

其中： 未知参数有三个，分别是大气散射系数β、大气光强 A和深度图 $ \hat{D} $。
 **（2）pipeline**

 ![image-20231217161623663](/Users/hurenrong/Library/Application Support/typora-user-images/image-20231217161623663.png)

(a) 分支是估计 ASM 参数的朦胧图像重建分支。

​	分支(a)的核心是**图像退化的物理模型**。在本文中，ASM用于重建模糊图像的退化过程。它结合了来自新视图合成分支的清晰图像和深度图来生成重建的模糊图像。其中ASM模型中的三个未知数，A为大气光，β为描述介质散射能力的散射系数，都由预先训练的DNN来预测。

(b)分支是Nerf分支 ，生成估计清晰图像的新型视图合成分支。

​	Nerf的正常流程，然后将生成ASM中的未知数深度图，送到分支a上去帮助优化。与传统Nerf不同的一点就是，这里采样不再随机采样，而是以相等的间隔进行下采样，便于大规模得到采样结果对齐的数据。

***

**第四篇：NeRF‐Tex: Neural Reflectance Field Textures**

**分类：**表面散射，与材质、纹理相关

**目的：**

​	1.使用神经场来模拟不同的中尺度结构，如皮毛、织物和草等，从真实图像中提取复杂的、难以建模的外观。

​	2.增加了可以建模的外观范围，并提供了对抗重复纹理伪影的解决方案。

​	3.展示了NeRF-Tex能够在场景中进行连续且一致的细节渲染。

**方法：**

​	1.与只使用经典的图形基元来建模相比，提出了采用一种由神经辐射场(NeRF-Tex)表示的多功能体积基元，它可以联合建模材料的几何形状及其对光线的响应。NeRF-Tex基元可以在一个基本网格上实例化，用所需的中观和微观尺度外观来构建纹理。

​	2.参数化NeRF纹理，一个的NeRF纹理可以捕获整个空间的反射场，而不是一个特定的结构。 NeRF-Tex通过将光照作为条件输入，而不是将其固化在神经表示中，从而实现更灵活的材料建模。这种方法允许在不同的光照条件下渲染一个已训练的模型。

**结果show：**

参数化控制下，可以调整不同区域的密度和反射场。这可以用来实现例如从直发到卷发的过渡，或在不同区域内空间变化颜色。

![image-20240117105439003](/Users/hurenrong/Library/Application Support/typora-user-images/image-20240117105439003.png)

***

**第五篇：Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging**

**分类：**nerf的应用，医学超声成像方面。 

**目的：**超声波在不同种类的组织中表现出不同的行为，导致了视角依赖的差异，为了消除这种差异。

**方法：**使用nerf + 基于物理的渲染公式，为由视图依赖差异导致的模糊区域生成几何精确的b超图像。

***

**第六篇：Nerf-det: Learning geometry-aware volumetric representation for multi-view 3d object detection**

**分类**：多视角，室内三维物体的感知，结合目标检测。

**目的**：该方法利用NeRF从RGB图像中学习几何感知的体积表示，以改善室内三维物体检测的性能。

**方法**：NeRF-Det通过提取图像特征并投影为3D体积网格，进行基于图像的3D目标检测。使用NeRF推断场景几何，并使用共享MLP连接3D目标检测与NeRF，以使用NeRF中的多视图约束来增强几何估计。

​	1.NeRF-Det通过共享的MLP（多层感知器）将多视角几何约束融入三维检测中【联合训练NeRF分支和3D检测网络】；

​	2.通过增加图像特征的先验信息来提高NeRF的泛化能力，并使用**高分辨率图像特征**来代替体积以解决高分辨率需求的问题。

​	3.在NeRF-Det中，通过将密度场转换为不透明度场，进一步提高了体积表示的几何感知性能，从而显著改善了室内三维物体检测的性能。

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20240117124559757.png" alt="image-20240117124559757" style="zoom: 33%;" />

***

**第七篇：Nerrf: 3d reconstruction and view synthesis for transparent and specular objects with neural refractive-reflective fields**

**分类**：使用nerf对**透明、镜面反射等材质**的物体进行三维重建和视图合成；融合了**菲涅尔方程**的用法。

**目的**：提出了一种新的折射-反射场NeRRF，它明确地将反射方程和折射定律整合到NeRF的光线追踪过程中。由于菲涅尔方程确定了反射的相对比例，NeRRF使用菲涅尔项统一地对折射和反射进行建模。

**方法**：![image-20240117131243632](/Users/hurenrong/Library/Application Support/typora-user-images/image-20240117131243632.png)

 (a) 几何估计。仅使用对象掩码作为先验，然后使用 Deep Marching Tetrahedra，这是一种混合形状表示来重建对象的几何形状。其中我们逐步编码多层感知器 (MLP) 以预测网格上的符号距离场和每个顶点偏移，用于平滑度和高频细节。

(b) 辐射估计。提出了一种方向无关的 NGP 模型，用于辐射估计。当光线与物体相交时，使用菲涅耳方程来计算光的出方向和辐射。同时使用超级采样去除高频伪影。【instant-NGP全称为Instant Neural Graphics Primitives，用于加速训练nerf收敛】

**结果：**

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20240117133144352.png" alt="image-20240117133144352" style="zoom: 50%;" />

(a) 重新照明：展示了改变镜面反光兔子和透明奶牛背景的能力。新的背景也使用NeRRF渲染器渲染。

(b) 材质编辑：通过将镜面反光马转换为透明马，改变透明兔子的折射率 (IOR) 来展示材质编辑。这些变化是通过编辑 NeRRF 中的材料属性来实现的。

(c) 对象替换：通过将镜面反光兔子转换为镜面反光奶牛，将透明球分别转换为透明兔子。

(d) 对象插入：通过将兔子和马添加到两个不同的场景中来演示对象插入。

***

**第八篇： DoF-NeRF: Depth-of-Field Meets Neural Radiance Fields. **

**分类**：优化nerf，加入了对**景深**的模拟。

**目的**：NeRF常基于针孔相机模型，并假设全焦输入。然而从现实世界捕获的图像通常具有有限的景深 (DoF)，而nerf合成的视觉中缺少对不同焦距下的效果生成。

【景深：**景深**是指在摄像机镜头前方的一段范围内，我们获得的图像清晰度是可以接受的。理想状态下，光线从物体处通过透镜完美地聚焦在成像平面上，此时我们就能看到一个锐利的成像。但更多的时候，由于物体过远或过近，光线都无法汇聚到成像平面上，形成了一个模糊圆或弥散圆（CoC），弥散圆在一定范围内，清晰度都是可以被人眼所接受的。那么这段范围所对应的物体距离范围，就是景深。在景深范围外，图像呈现模糊状态。】 

<img src="/Users/hurenrong/Library/Application Support/typora-user-images/image-20240117210614343.png" alt="image-20240117210614343" style="zoom:33%;" />

**方法：** 提出了DoF-NeRF，引入了Concentrate-and-Scatter技术，这是一种可以处理浅自由度输入的神经渲染方法，可以模拟DoF效果。它根据几何光学原理模拟透镜的孔径。这样的物理保证允许DoF-NeRF操作具有不同焦点配置的视图。 

![image-20240117211521228](/Users/hurenrong/Library/Application Support/typora-user-images/image-20240117211521228.png)

***

**第九篇：Circular SAR Incoherent 3D Imaging with a NeRF-Inspired Method**

**分类**：nerf的应用，结合雷达+三维重建场景

**目的**：提出了基于神经网络的循环合成孔径雷达（CSAR）三维成像方法，该方法利用MLP模型解决CSAR场景三维成像问题，实现了从单通道单次数据中进行三维成像，同时避免了对高精度运动记录的需求，从而提高了无人机SAR系统的成像质量和数据采集效率。

**方法**：多视角nerf成像，传统方法，与其专业【雷达+成像】做了一个结合，并没有什么对nerf本身的创新点。

***

**第十篇：Variable Imaging Projection Cloud Scattering Tomography**

分类：nerf的引申使用，借用了nerf的思想，本质还是DNN，关系不大













