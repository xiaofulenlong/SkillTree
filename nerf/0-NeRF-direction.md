### NeRF 资料

---

### 基础

##### （1）原始方法		

​		原始 NeRF 论文（精读）：Mildenhall B, Srinivasan P P, Tancik M, et al. Nerf: Representing scenes as neural radiance fields for view synthesis[J]. Communications of the ACM, 2021, 65(1): 99-106. 回答如下问题：

（1）输入具体有什么？每个模块的具**<u>体输入输出有什么</u>**，每个**<u>模块是如何工作的</u>**？分为以下几个部分回答：

- 光线生成模块
- 光线点采样模块
- 粗网络（coarse network）
- 重要性采样模块
- 精网络（fine network）
- 渲染 - loss evaluation 模块

（2）复现此方法。基于 pytorch，可参考：

- NeRF-Pytorch: https://github.com/yenchenlin/nerf-pytorch
- NeRF（我的实现，原始NeRF实现是 tensorflow）：https://github.com/Enigmatisms/NeRF

​		不到写不出来时不要看这些代码逻辑。基于自己的理解实现，实现过程中写不出来的就是没理解透的点。注意做好记录。

---

##### （2）视角稀疏引起的欠定问题		

​		Depth-Supervised NeRF (DS-NeRF): Deng K, Liu A, Zhu J Y, et al. Depth-supervised nerf: Fewer views and faster training for free[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 12882-12891.

- 回答如下问题：
  - NeRF如何渲染深度图？在你的 NeRF 复现中实现深度图输出。
  - 理解深度监督是如何加入的？

​		Info-NeRF：Kim M, Seo S, Han B. Infonerf: Ray entropy minimization for few-shot neural volume rendering[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 12912-12921.

​		Free-NeRF：Yang J, Pavone M, Wang Y. FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 8254-8263.

​		Reg-Nerf：Niemeyer M, Barron J T, Mildenhall B, et al. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 5480-5490.

​		理解如何在视角稀疏（也即问题欠定）情况下更好地进行网络优化。

##### （3）动态场景NeRF（引入宏观时间信息）【dynamic】

​		D-NeRF：Pumarola A, Corona E, Pons-Moll G, et al. D-nerf: Neural radiance fields for dynamic scenes[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 10318-10327.

​		理解 deformation 网络如何建模物体的移动。你认为此文章的存在什么问题？

​		Huang X, Zhang Q, Feng Y, et al. Hdr-nerf: High dynamic range neural radiance fields[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 18398-18408.

​		Knodt J. Continuous Dynamic-NeRF: Spline-NeRF[J]. arXiv preprint arXiv:2203.13800, 2022.

​		NR-NeRF：Tretschk E, Tewari A, Golyanik V, et al. Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 12959-12970.

---

### 进阶

##### (1) 完全显式建模

​		Plenoctree（略读了解方法即可）：Yu A, Li R, Tancik M, et al. Plenoctrees for real-time rendering of neural radiance fields[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 5752-5761.

​		**Plenoxel（精读）: Fridovich-Keil S, Yu A, Tancik M, et al. Plenoxels: Radiance fields without neural networks[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 5501-5510.**

​		理解球面谐波函数如何建模场景。

##### （2）瞬态 NeRF  【transient】

​		Transient NeRF：Malik A, Mirdehghan P, Nousias S, et al. Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction[J]. arXiv preprint arXiv:2307.09555, 2023.

​		ToRF：Attal B, Laidlaw E, Gokaslan A, et al. Törf: Time-of-flight radiance fields for dynamic scene view synthesis[J]. Advances in neural information processing systems, 2021, 34: 26289-26301.

​		Martin-Brualla R, Radwan N, Sajjadi M S M, et al. Nerf in the wild: Neural radiance fields for unconstrained photo collections[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 7210-7219.

​		Sabour S, Vora S, Duckworth D, et al. RobustNeRF: Ignoring Distractors with Robust Losses[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 20626-20636.

​		Marí R, Facciolo G, Ehret T. Sat-nerf: Learning multi-view satellite photogrammetry with transient objects and shadow modeling using rpc cameras[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 1311-1321.

​		了解如何考虑时间信息 ？

**（3）散射与去雾 NeRF** 【**scatter**】

​	SeaThru-NeRF：Levy D, Peleg A, Pearl N, et al. SeaThru-NeRF: Neural Radiance Fields in Scattering Media[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 56-65.

​	Wang Y, Yang S, Hu Y, et al. NeRFocus: Neural Radiance Field for 3D Synthetic Defocus[J]. arXiv preprint arXiv:2203.05189, 2022.

​	Dehaze NeRF： Chen W T, Yifan W, Kuo S Y, et al. Dehazenerf: Multiple image haze removal and 3d shape reconstruction using neural radiance fields[J]. arXiv preprint arXiv:2303.11364, 2023.

​	Dehazing NeRF：Li T, Li L U, Wang W, et al. Dehazing-NeRF: Neural Radiance Fields from Hazy Images[J]. arXiv preprint arXiv:2304.11448, 2023.

​	了解如何考虑散射介质？

##### （4）3D Gaussian Splatting

​		如何用 3D 高斯对场景进行表征？

​		SIGGRAPH 23 best paper: Kerbl B, Kopanas G, Leimkühler T, et al. 3d gaussian splatting for real-time radiance field rendering[J]. ACM Transactions on Graphics (ToG), 2023, 42(4): 1-14. （有精力就精读）

​		动态3D高斯喷溅：Luiten J, Kopanas G, Leibe B, et al. Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis[J]. arXiv preprint arXiv:2308.09713, 2023. （了解方法）

​		Yang Z, Gao X, Zhou W, et al. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction[J]. arXiv preprint arXiv:2309.13101, 2023.（了解方法）

##### （5）物理描述（精读）

​		如何将基于物理的散射模型融入到 NeRF 类方法中？microflake microfacet

​		Mai A, Verbin D, Kuester F, et al. Neural Microfacet Fields for Inverse Rendering[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 408-418.

​		回答如下问题：

- microfacet 模型如何使用？
- 场景的光传输是如何建模的？

***







