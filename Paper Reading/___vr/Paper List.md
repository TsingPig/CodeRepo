# Paper List

## VR Software Reliability and Security

### [x] [How developers optimize virtual reality applications A study of optimization commits in open source unity projects](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/_How%20developers%20optimize%20virtual%20reality%20applications%20A%20study%20of%20optimization%20commits%20in%20open%20source%20unity%20projects.pdf)

- 开源VR APP项目性能优化方法（方法分类学、优化的代价、大小厂APP的区别）

	- git commits messages 手动分析（两位研究者独立分析，合并、讨论确定最终结果，通过Cohen's-Kappa值达成一致）

	- call graph 调用图分析（确定优化具体应用于哪个生命周期）

	- Static Analysis（srcML解析C#代码，GumTree解析抽象语法树，通过metaID检测unity依赖文件）

	- AR/VR软件内存泄漏检测

	- AR/VR自动化性能优化检测工具

		-  AR/VR中可视化脚本开发的检测

### [x] [A Study of User Privacy in Android Mobile AR Apps](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/_A%20Study%20of%20User%20Privacy%20in%20Android%20Mobile%20AR%20Apps.pdf)

- 安卓端AR APP的隐私和数据安全问题

	- Androguard提取 .apk的申请权限

	- Static Taint Analysis（使用Flowdroid，通过SUSI对敏感数据源分类，SEEKER和SUSI 对高危数据汇分类）

	- 手动检测Google APP Store Date Safety部分可用性

### [x] [An Empirical  Study on Oculus Virtual Reality Applications Security and Privacy Perspectives](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/An_Empirical_Study_on_Oculus_Virtual_Reality_Applications_Security_and_Privacy_Perspectives.pdf)

- Oculus APP安全漏洞和隐私问题

### [x] [Automated Usability Evaluation of Virtual Reality Application](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Automated%20Usability%20Evaluation%20of%20Virtual%20Reality%20Application.pdf)

- VR APP自动化可用性测试方法，User Study 验证可行性（用户可以自行探索场景，自动记录任务树，可用性异味检测）

  Approach类的文章是如何组织展开方法的：
  在foundations中， effective actions 、 ineffective actions、task、subtask、task model、task tree、event list的定义（可以借用的定义），用户在需要执行task的时候会有若干action，APP对action的响应是event，记录下来构成event list存储在数据库或者日志文件中，用户独有。
  在Related work中，介绍了可用性定义（是用户体验的子集），可用性工程：
  用户导向的可用性工程，需要用户在专家观察下，执行特定的任务。
  专家导向的可用性工程，专家执行一些列预定义的任务，进行正式的评估。
  两种方法需要大量人力时间。
  在方法中，说明了任务树的特点：
  多个用户记录生成的任务树能够反映使用共性。
  用户行为越一致，任务树越简单。
  接着，说明了可用性异味检测，列举了他们在之前的桌面端和网站上的可用性异味检测的异味。然后，说明在VR可用性异味的选择上，哪些异味保留、修改以及新提出，以及说明这些异味具有代表性的原因。然后，阐述了可用性异味检测能够说明可用性问题的概率指标。
  阐述了5种可用性异味，包括它们的选择理由、在任务树中的特点、如何在任务树中选择等。
  Case study部分，首先说明了选择了两种VR场景以及合理性、任务流程，以及为什么使用传送机制作为移动交互方法；接着介绍了VR 手柄，以及自己定义的四种常见VR交互方式，考虑这些所有交互方式是为了更好的验证普适性，并展现了四种交互模式对要记录的action的实现方式。
  User study介绍了选择的被试者的年龄分布、身份分布。说明了实验是如何展开的。
  针对实验结果，如可用性异味检测分布，检查真阳性结果是否准确反映了可用性问题。
  
  
  
	- 扫描场景中的所有可交互VR物体，找到其中具有event handling脚本的，并扩展使其能够在event log的时候，自动化用户行为记录（保存到log file）：物体抓取、释放、物体使用、物体不使用、头部移动；将这些记录保存到中央服务器

	- 提出了任务树生成算法，任务树能够代表记录的用户行为。

	- 分析任务树，Usability Smell 可用性异味检测

	- User Study、假设检验

### [x] [A Peek into the Metaverse  Detecting 3D Model Clones in Mobile Games](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/_A%20Peek%20into%20the%20Metaverse%20%20Detecting%203D%20Model%20Clones%20in%20Mobile%20Games.pdf)

- 3D模型克隆检测框架和实证研究

  模型克隆的定义：
  简单的拉伸、挤压、噪音，视为同一个模型
  模型有结构变化，视为不同
  独立创建的相同结构的简单模型
  购买相同的模型
  静态模型噪声：简单的拉伸、挤压、噪声
  动态模型噪声：IDE对动画关键帧的修改、遗漏、优化
  
  为了解决静态模型噪声，方法是对顶点去重排序，通过有序顶点ID编码面（而不是绝对的坐标值）可以更好的表示模型的结构关系；对编码后的面排序，对面序列哈希加密计算模型哈希值。
  
  为了解决动态模型（包括静态模型、骨骼和动画，即关键帧序列
  ）噪声，方法是对带有动画的模型首先按照形状信息分组，然后按照骨骼信息分组，检查同组内的动态模型克隆。
  
  
	- [模型提取 AssetStudio](https://github.com/Perfare/AssetStudio)

		- 从APK文件中解码得到3D模型（静态模型转为obj，动态模型转为fbx），提取形状（顶点和面）、骨骼（顶点和面绑定骨骼结构）、动画（关键帧序列）

	- 解决静态模型噪声

		- 顶点去重排序，通过有序顶点ID编码面（而不是绝对的坐标值）可以更好的表示模型的结构关系；对编码后的面排序，对面序列哈希加密计算模型哈希值。

	- 解决动态模型噪声（动画关键帧噪声）

		- 对带有动画的模型首先按照静态方法形状信息分组，然后按照骨骼信息分组（对骨骼进行递归计算哈希值，用递归哈希值代表根骨骼节点，相似的哈希值归为同一嫌疑组）

		- 检查同嫌疑组内的动态模型克隆（Pairwase-Comparison）：通过对动画关键帧中变换、旋转、缩放的数据离散点进行拟合，通过拟合曲线相似度检测算法（欧拉距离、最长上升子序列）评估相似性。

	- 提到了重打包游戏和原始版本的重叠率问题

	- AR/VR中3D模型保护（模型加密）

	- ML-based 模型克隆检测算法

## VR Software Development and Test

### [x] [Virtual Reality (VR) Automated Testing in the Wild: A Case Study on Unity-Based VR Applications](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/_Virtual%20Reality%20(VR)%20Automated%20Testing%20in%20the%20Wild%20A%20Case%20Study%20on%20Unity-Based%20VR%20Applications.pdf)

- VR APP开发测试（测试代码比重、测试有效性）

  测试的目的：
  验证功能是否达到预期（如图形渲染达到预期效果）
  防止回归（新代码引入问题能够及时发现）
  自动化质量检查
  测试工作流：
  为单元、模块编写测试代码，定义预期输输入输出和断言条件（例如，渲染测试中，开发人员准备参考图像，定义一个可接受图形的像素差范围）
  使用测试框架（ NUnit、JUnit 或 Pytest ），自动化运行标记的测试方法，比较结果
  测试集成到“持续集成”（CI）：集成到CI / CD 管道，当代码提交到代码库，CI工具自动进行测试
  
  
  
	- 自动分析

		- Static Analysis（srcML解析C#代码，生成抽象语法树）

		- 测试代码与功能代码比例（手动考察被标为'Test'等的类和方法站总的比重）

		- 测试样例有效性（正确方法是采用计算Code Coverage，但是由于编译和版本兼容问题，采用Assertion Density metric规避）

		- 测试样例质量（考察Test Smell 测试异味检测，即代码中测试样例的的不规范\模棱两可，并做了人工评估）

	- Taxonomy：手动分析

		- 定性分析

		- 两位作者独立观察所有测试方法、源码，生成测试方法报告，第三个作者后加入验证

		- 370个方法分类，多轮会议投票达成共识消除不一致，使用Fleiss' Kappa系数

		- 卡片分类法

		- 第四位作者验证分类体系的规范性

### [x] [RE Methods for Virtual Reality Software Product Development A Mapping Study](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/_RE%20Methods%20for%20Virtual%20Reality%20Software%20Product%20Development%20A%20Mapping%20Study.pdf)

- 需求工程的映射文献研究 

	- 需求规范SRS、自动化RE方法、需求收集工具

## VR Software Use Experience and Interaction

### [x] [VRTest: An Extensible Framework for Automatic Testing](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/_VRTest%20An%20Extensible%20Framework%20for%20Automatic%20Testing.pdf)

- VRTest 场景自动化探索工具，自动控制相机移动、追踪物体交互事件（点击等），探索可交互物体

	- 场景监视器

		1. 计算拥有renderer组件，即可见物体的Bounding Boxes包围盒

		2. 位置

		3. 识别可交互物体

		4. 识别物体模板/预制体，从而减少对同一类型物体的重复计算

	- 物体仪表

		1. 状态变化报告器：获得物体的 EventTrigger组件，获得组件的Entry条目，对回调增加监听

		2. 将状态变化报告器添加到所有可交互物体，在对应event触发时向上报告

	- 控制器：通过不断询问场景监视器的信息，让物体仪表追逐状态，从而决定决策，执行物体的事件，并迭代进行

	- 配置

		- 相机移动、旋转限制参数配置

	- 探索算法

		- VRMonkey：随机

		- VRGreed：最短距离贪心

	- Evaluation实验可交互物体的探索程度

### [x] [VRGuide Efficient Testing of Virtual Reality Scenes via Dynamic Cut Coverage](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/VRGuide%20Efficient%20Testing%20of%20Virtual%20Reality%20Scenes%20via%20Dynamic%20Cut%20Coverage.pdf)

- VRGuide通过计算几何方法，对VRTest的相机寻路进行了优化

	- 静态计算几何

		- [Art Gallery Problem 美术馆问题](https://zhuanlan.zhihu.com/p/388919331 在一个)

		- Watchman Route Problem 观察者路由问题
	在多边形美术馆里，选择保安的最短路径，让他的沿途可以看到每一个角落。

		- The Cut Theory（Convex 凹点、割）

	- 动态计算

		- Dymamic Cut（面向边、面向物体、动态割）

		- 对VRTest的寻路算法提升：通过动态割寻路，计算距离最近的割的距离；相比VRTest的最短距离，得到提升。

	- Evaluation：通过随着时间对可交互物体的探索覆盖率指标，来对比VRTest的方法

	- 附加性的bug检测：通过地毯式触发交互式物体，执行代码，地毯式检查bug

		- AR/VRbug检测新方向：地毯式场景探索

	- 在事件检测中，对于需要固定交互顺序的交互式物体的检测效果一般；后续需要通过静态分析分析代码依赖实现。

		- 静态分析法对固定交互顺序进行检测

## Other

### [ ] [AdCube: WebVR Ad Fraud and Practical Confinement of Third-Party Ads](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/AdCube%20WebVR%20Ad%20Fraud%20and%20Practical%20Confinement%20of%20Third-Party%20Ads.pdf)

### [ ] [Demystifying Mobile Extended Reality in Web Browsers  How Far Can We Go](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Demystifying%20Mobile%20Extended%20Reality%20in%20Web%20Browsers%20%20How%20Far%20Can%20We%20Go.pdf)

### [ ] [Erebus Access Control for Augmented Reality Systems](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Erebus%20Access%20Control%20for%20Augmented%20Reality%20Systems.pdf)

### [ ] [Glib  towards automated test oracle for graphically-rich applications](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Glib%20%20towards%20automated%20test%20oracle%20for%20graphically-rich%20applications.pdf)

### [ ] [Less Cybersickness, Please Demystifying and Detecting Stereoscopic Visual Inconsistencies in Virtual Reality Apps](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Less%20Cybersickness,%20Please%20Demystifying%20and%20Detecting%20Stereoscopic%20Visual%20Inconsistencies%20in%20Virtual%20Reality%20Apps.pdf)

### [ ] [OVRseen Auditing Network Traffic and Privacy Policies in Oculus VR](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/OVRseen%20Auditing%20Network%20Traffic%20and%20Privacy%20Policies%20in%20Oculus%20VR.pdf)

### [ ] [Playing without paying Detecting vulnerable payment verification in native binaries of unity mobile games](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Playing%20without%20paying%20Detecting%20vulnerable%20payment%20verification%20in%20native%20binaries%20of%20unity%20mobile%20games.pdf)

### [ ] [PredART Towards Automatic Oracle Prediction of Object Placements in Augmented Reality Testing](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/PredART%20Towards%20Automatic%20Oracle%20Prediction%20of%20Object%20Placements%20in%20Augmented%20Reality%20Testing.pdf)

### [ ] [Secure Multi-User Content Sharing for Augmented Reality Applications](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Secure%20Multi-User%20Content%20Sharing%20for%20Augmented%20Reality%20Applications.pdf)

### [ ] [When the User Is Inside the User Interface An Empirical Study of UI Security Properties in Augmented Reality](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/When%20the%20User%20Is%20Inside%20the%20User%20Interface%20An%20Empirical%20Study%20of%20UI%20Security%20Properties%20in%20Augmented%20Reality.pdf)

### [ ] [Development of Real-Time QA/QC Tools for AEC in Unity](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Development%20of%20Real-Time%20QA_QC%20Tools%20for%20AEC%20in%20Unity.pdf)

### [ ] [Live Semantic 3D Perception for Immersive Augmented Reality](https://ieeexplore.ieee.org/document/8998140)

## Performance Optimization

### [x] [A Survey of Performance Optimization for Mobile Application](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Performance%20Optimization/A%20Survey%20of%20Performance%20Optimization%20for%20Mobile%20Application.pdf)

- 关于安卓APP性能优化的文献综述

  文章提到的概念：
  Non-functional Performance：非功能性性能
  Responsiveness：GUI交互响应时间（小与150ms）
  无cached数据的冷启动时间
  内存占用
  功耗
  软件性能工程 Software Performance Engineering
  
  对开发人员：
  要避免在优化性能的时候，引入功能性bug、影响代码可维护性
  
  
	- 响应时间优化

		- AR/VR 脚本语言对程序性能（响应时间）的影响研究

		- 编程语言

		- Offloading

		- Antipattern

		- Refactoring 重构代码

		- Prefetch（Cache Data）

		- 减少 I/O Request

	- 启动时间优化

		- Preloading（规律预测预加载）

		- 灵活LRU后台应用

	- 内存优化

		- 代码异味减少内存泄漏

		- 改进垃圾回收机制

		- 检测页面级的内存重复

		- 后台应用压缩到GPU缓冲区

		- 代码设计中，减少iterators, for–each loops, lambda expressions and the Stream API

	- 功耗优化

### [ ] [Research on Key Technologies for Deep Optimization  of Unity Based Scenarios](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Performance%20Optimization/Research%20on%20Key%20Technologies%20for%20Deep%20Optimization%20%20of%20Unity%20Based%20Scenarios.pdf)

### [ ] [Research on the 3D Game Scene Optimization of Mobile Phone Based on the Unity 3D Engine](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Performance%20Optimization/Research%20on%20the%203D%20Game%20Scene%20Optimization%20of%20Mobile%20Phone%20Based%20on%20the%20Unity%203D%20Engine.pdf)

### [ ] [Research on Unity Scene Optimization Based on Fast LoD Technique Performance Comparison on Android Mobile Platform](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Performance%20Optimization/Research%20on%20Unity%20Scene%20Optimization%20Based%20on%20Fast%20LoD%20Technique%20Performance%20Comparison%20on%20Android%20Mobile%20Platform.pdf)

### [ ] [Mobile Application Processors_Techniques for Software Power-Performance Optimization](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Performance%20Optimization/Mobile%20Application%20Processors_Techniques%20for%20Software%20Power-Performance%20Optimization.pdf)

### [ ] [Optimizing Immersion_Analyzing  Graphics and Performance  Considerations in Unity3D  VR Development](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Performance%20Optimization/Optimizing%20Immersion_Analyzing%20%20Graphics%20and%20Performance%20%20Considerations%20in%20Unity3D%20%20VR%20Development.pdf)

- 子主题 1

  VR性能优化旨在让VR体验尽可能流畅（更高的帧率，更低的时延，高分辨率立体渲染）；
  需要在美感和无缝性能表现上做平衡，且需彻底掌握Unity底层渲染功能；
  VR应用相比传统的移动应用，包含了复杂的实时动画、高的GPU需求，且除去代码设计外的场景和资源设计会对渲染性能有重大影响。
  Visual Teleportation 是弥补物理区域受限的一种策略；
  
  
  
	- 图形优化

		- Unity Dynamic Scaling 动态分辨率

		  https://www.bilibili.com/read/cv10380980/
		  
		  unity 动态分辨率 & 缩放渲染 & 设置分辨率 提升FPS - 知乎 (zhihu.com)
		  
		- Texture Compression（ASTC、ETC2）

		- 静态/动态遮挡剔除 (Static/Dynamic Occlusion Culling)：不展示为其他物体遮挡的物体；

		- 光照烘焙(Light Baking) 在编译阶段预计算(Pre-calculate) 光照效果；

		- AR/VR应用中的资源加载方式

	- CPU优化

		- 物体合批、GPU Instancing：可以通过让物体在一个Draw Call内绘制完成，从而减少Draw Calls；

		- 动态加载资源：AssetBundle\Addressable

		- Multithreadin & UI Optimization & Audio Enhancement

### [ ] [Performance Analysis and Optimization Techniques in Unity 3D](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Performance%20Optimization/Performance%20Analysis%20and%20Optimization%20Techniques%20in%20Unity%203D.pdf)

- 子主题 1

### [ ] [Performance optimization opportunities in the Android software stack](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Performance%20Optimization/Performance%20optimization%20opportunities%20in%20the%20Android%20software%20stack.pdf)

- 子主题 1

### [ ] [eFish’nSea: Unity Game Set for Learning Software Performance Issues Root Causes and Resolutions](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/___vr/Performance%20Optimization/eFish%E2%80%99nSea%20Unity%20Game%20Set%20for%20Learning%20Software%20Performance.pdf)

- 子主题 1

## 附件

### [CCF 推荐分级表](file:///F:/--CodeRepo/--CodeRepo/Paper%20Reading/%E4%B8%AD%E5%9B%BD%E8%AE%A1%E7%AE%97%E6%9C%BA%E5%AD%A6%E4%BC%9A%E6%8E%A8%E8%8D%90%E5%9B%BD%E9%99%85%E5%AD%A6%E6%9C%AF%E4%BC%9A%E8%AE%AE%E5%92%8C%E6%9C%9F%E5%88%8A%E7%9B%AE%E5%BD%95-2022%E6%9B%B4%E5%90%8D%E7%89%88.pdf)

### 检索平台

- [XOL](https://www.x-mol.com/)

- [ACM Digit library](https://dl.acm.org/search/advanced)

- [IEEE Xplore](https://ieeexplore.ieee.org/search/advanced)

- [AMiner](https://www.aminer.cn/)

