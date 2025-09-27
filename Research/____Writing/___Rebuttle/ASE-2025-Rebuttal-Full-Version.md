---
title: ASE 2025 Rebuttal Full Version
---

# 
# ASE 2025 Rebuttal Full Version
We'd like to express our gratitude for  providing your valuable feedback. We'll respond to you 
individually.
## Reviewer A
### RA-RQ1 & Comments-Soundness-ablation variants
> Why only interactions are removed in the ablation study but not other components? 


我们的消融实验聚焦于核心交互类型，因为它们是我们提出的 EAT 框架相较于现有方法的主要创新点。
其他模块（如导航和信息提取）是agent运行的基础前置组件，无法直接移除进行对比。不过，我们在补充实验中比较了 Greedy 算法和回溯剪枝算法，在不超过 100 个交互物体的场景下，回溯剪枝算法能提升 15%~20% 的运行效率，但会带来额外的计算开销，导致帧率下降。

Our ablation study focuses on core interactions, since these constitute the primary innovations of our EAT framework compared to existing approaches. 

Other modules, such as navigation and information extraction, serve as essential preconditions for the agent’s operation and cannot be directly omitted for comparison. However, in supplementary experiments, we compared the Greedy algorithm with the Backtracking-and-Pruning algorithm. Within scenarios involving up to 100 interactable objects, Backtracking-and-Pruning yields a 15%–20% runtime efficiency improvement but incurs additional computational overhead, resulting in frame rate degradation.

### RA-RQ2 & Comments-Soundness-detected bugs
> What happened to the detected real-world bugs?


我们通过分析 Unity 控制台日志、运行时代码报错和脚本触发链，人工确认了这些 Bug 均为真实异常，并非误报。由于评估项目已归档，我们已向原作者发送多封邮件，报告这些 Bug，目前仍在等待其回复。

We manually validated the reported bugs by examining Unity console logs, runtime exceptions, and the associated script-trigger chains to ensure they represent genuine faults rather than false positives. Because the evaluated project is archived, we have sent multiple emails to the original authors to report these bugs and are currently awaiting their response.




## Reviewer B
### RB-RQ1 
> Based on the Approach and Threats to Validity sections, it appears that the analysis and modeling process is primarily manual. Could you please clarify the specific steps involved in this manual analysis?

我们在 Section II.A (项目收集与分析) 和 II.B (模型抽象) 中的人工分析流程如下：


具体而言，我们要求开发者先完整体验项目的核心游戏流程，同时在运行时对场景状态、脚本调用、Console输出等信息进行逐帧或定期的动态观察与启发式分析，以识别动态的关键交互目标与任务逻辑链。通过人工对脚本的启发式分析，包括与 GameObject 的绑定关系、运行时事件顺序等，理解核心的交互逻辑，将覆盖的交互类型、交互物体进行启发式分类，从而进行Model Abstraction。

这种结合游戏体验与动态分析的方式虽然人工，但对确保高质量的任务建模是必要的。我们在105个项目上进行了分析，也是贡献的一部分工程实践价值。


In Section II.A (Project Collection and Analysis) and B (Model Abstraction), our manual analysis process is as follows: Specifically, we require developers to experience the core game flow of the project first, and at runtime, we conduct frame-by-frame or periodic dynamic observation and heuristic analysis of scene states, script calls, console outputs, etc., to identify dynamic key interactions, goals, and task logic chains.

Through manual heuristic analysis of scripts, including binding methods with GameObjects, runtime event sequences, etc., the core dynamic interaction logic is understood, and the covered interaction types and interaction objects are heuristically categorized for Model Abstraction. 

This combination of gameplay experience and dynamic analysis, although artificial, is necessary to ensure high-quality task modeling of the task modeling. We have analyzed this on 105 projects, and are contributing a portion of the value of the engineering practice.

### RB-RQ2
> Do you believe your current modeling process can generalize to a wide range of VR applications beyond the use case(s) demonstrated in the paper? For instance, how would it handle diverse interaction paradigms such as gesture-based input, gaze-based selection, or multi-user collaboration environments?


是的，我们的建模流程具备良好的可扩展性。这是因为，在我们的架构中，Action 层通过抽象输入动作及其绑定的目标物体来实现交互模拟。我们提供的 EAT 框架与具体的输入方式无关，而具体项目则负责提供具体的动作和目标物体。

例如您提到的手势输入可以通过以下方式集成：
在 Entity 层：手势可抽象为特殊的“输入触发器”组件（例如继承 XR 手势识别器接口）。 在 Action 层：可定义 GestureAction 类封装不同手势动作（如 swipe、pinch、point）。 在 Task 层：手势触发的任务可作为状态转移事件纳入 PFSM 中，如“swipe 以翻页”、“pinch 以缩放”等。这些扩展无需更改现有框架核心逻辑，只需在配置文件或组件层补充相应映射规则即可。


Yes, our modeling process has good scalability. This is because in our framework, the Action layer simulates interactions by abstracting input actions and their bound target objects. The EAT framework we provide is independent of specific input methods, with specific projects responsible for providing specific actions and target objects.

For example, the gesture input you mentioned can be integrated as follows: In the Entity layer, gestures can be abstracted as special "input triggers" components (such as inheriting XR gesture recognizer interface). In the Action layer, a GestureAction class can be defined to encapsulate different gesture actions (such as swipe, pinch, point). In the Task layer, tasks triggered by gestures can be incorporated as state transition events into the PFSM, such as "swipe to turn the page", "pinch to zoom", and so on. 

These extensions do not require modifying the core logic of the existing framework, just supplementing corresponding mapping rules in the configuration file or component layer.





### RB-Comments

- Soundness
    1. Analysis methodology & Model Abstraction: 我们要求开发者先完整体验项目的核心游戏流程，随后在运行时对场景状态、脚本调用、Console输出等信息进行逐帧或定期的动态观察与启发式分析，以识别关键交互目标与任务逻辑链。通过人工对脚本的全部启发式分析，包括与 GameObject 的绑定关系、运行时事件顺序等，理解核心的交互逻辑，将覆盖的交互类型、交互物体进行启发式分类，从而进行Model Abstraction。例如当遇到新的一个没有在工具包预定义的交互行为，测试者可以按照自己的测试需求，决定是否将其归类为已经建模完成的类别，或者增加自己新的类型接口。
    2. PFSM & Model Extension: shooting application 只是我们为了方便解释的一个例子。在Section V.A部分可以看到我们的测试项目集覆盖了不只是Shoot类型游戏，还包括Simulation, Adventure, Puzzle等，实验结果表面了我们的方法在不同类型的项目上都可以表现出色。至于如何扩展，由于PFSM的节点都是来自于EAT框架的Action和Task节点，PFSM实际上只负责Task节点的转换，可以很好的兼容不同类型的游戏。例如，解密游戏的Task可以被建模为若干物体触发、物体移动、角色移动的Task，这些都是与游戏类型无关的任务，也不需要测试者进行额外的配置。
    3. Dataset diversity and separation between training/modeling: 在Section V.A部分可以看到我们的测试项目集覆盖了不只是Shoot类型游戏，还包括Simulation, Adventure, Puzzle等类型。实际上我们的建模分析了105个项目的目的是为了了解常见的VR交互方式、交互物体，在去除了9个评估的项目后，得到的经验依然成立，并不涉及模型训练以及数据泄露的问题，也不会影响我们的实验结果。
    4. Criteria of real-world bugs detection: 非常好的建议。我们通过人工分析Unity控制台日志、运行时代码报错以及对应的脚本触发逻辑链，启发式地确认了这些 Bug 的确属于异常行为，并非误报。

- Novelty
    1. Automated modeling: 感谢您非常好的建议。目前我们的工具在项目分析统计时，已经用到了一些简单的自动化工具进行辅助，而核心的经验提取、模型建模，是无法通过现有的静态分析、物体分析学习到好的经验，因此我们的一个贡献点也是人工实证分析建模。后续工作，我们考虑用LLM等方法对项目进行全面分析，并验证LLM能够多大程度替代我们的人工分析，所以当前工作的人工分析也会成为后续的baseline对比。


- Verifiability
    1. Reamdme: 感谢建议。我们当前Github仓库包含了非工具核心部分（如本地测试项目等），实际上发行版本是仓库子目录VRExplorer-683A/Assets/VRExplorer(也是我们将后续公开的仓库)，在该目录下我们已经提供了一份复现指南README.md。为了更清晰展示，我们已经在当前仓库额外加上了Readme复现指南。
    
- Soundness
    1. Analysis Methodology & Model Abstraction: We require developers to fully experience the core game flow of a project. Then, during runtime, they perform frame-by-frame or periodic observation of the scene state, script dynamic calls, and Console output. This enables heuristic analysis to identify key dynamic interactive targets and the logical chain of task execution. Through comprehensive heuristic analysis of scripts—including their bindings to GameObjects and the runtime event sequence—we extract and understand the core interactive logic. Interaction types and interactive objects are heuristically categorized for abstraction. For instance, when encountering a new interaction behavior not pre-defined in the toolkit, test engineer can choose—based on testing needs—whether to classify it under an existing modeled category or define a new interaction type interface in Action layer.
    2. PFSM & Model Extension: The shooting application is merely an illustrative example. As shown in Section V.A, our evaluation includes not only shooting games, but also simulation, adventure, and puzzle projects. The experimental results demonstrate the effectiveness of our method across various genres. Regarding extensibility: our modeling process has good scalability. This is because in our framework, the Action layer simulates interactions by abstracting input actions and their bound target objects. The EAT framework we provide is independent of specific input methods, with specific projects responsible for providing specific actions and target objects. Moreover, since PFSM nodes originate from EAT framework's Action and Task nodes, PFSM merely manages task-level transitions. This makes it highly compatible with diverse game types. For example, puzzle game tasks can be modeled as combinations of object triggers, object movements, or character movements—all of which are task-level abstractions independent of game type, requiring no extra configuration from the tester.
    3. Dataset Diversity and Separation between Training/Modeling: As shown in Section V.A, our evaluated projects span multiple genres, including simulation, adventure, and puzzle games—not just shooting. The goal of analyzing 105 projects was to understand common VR interaction patterns and objects. After excluding the 9 projects used for evaluation, the insights gained remain valid. Our method involves no model training or data leakage, thus not impacting the experimental results.
    4. Criteria for Real-World Bug Detection: Thank you for the excellent suggestion. We heuristically confirmed that the reported bugs are genuine abnormal behaviors (not false positives) by manually analyzing Unity Console logs, runtime error traces, and the corresponding script-trigger chains.

- Novelty
    Automated Modeling: Thank you for the insightful suggestion. Currently, our tool utilizes some lightweight automation during project analysis and statistics gathering. However, the core process of extracting insights and building the model relies heavily on manual empirical analysis, as existing static or object-level analysis tools are insufficient to capture such domain knowledge. Thus, this manual modeling is also one of our key contributions. In future work, we plan to explore using LLMs for comprehensive project analysis and evaluate the extent to which LLMs can replicate or replace our manual process. Therefore, our current work will serve as a baseline for future comparisons.

- Verifiability
    README: Thank you for the suggestion. Our current GitHub repository includes more than just the core tool (e.g., local test projects). The actual tool release resides in the subdirectory VRExplorer-683A/Assets/VRExplorer, which will be the version made public later. In that directory, we have already provided a detailed README.md for replication. To enhance clarity, we’ve now also added a high-level README reproduction guide in the root directory of the repository.





## Reviewer C
### RC-RQ
> Will the selected Core code be useful for many other types of VR applications? If not, will it introduce additional workloads for the developers?

在 Section VI（内部有效性威胁）中，我们采用多数人投票机制邀请 4 位资深 Unity 开发者确定核心代码边界。投票结果完全一致，原则是剔除第三方库等非测试目标代码。
因此，开发者只需创建一个 Assembly Definition 文件来标识测试目标代码，无需额外手动筛选工作。

In Section.VI (Threats to Internal Validity), we describe using a majority-voting mechanism with four experienced Unity developers to determine the Core Code boundary. 

The voting was unanimous: exclude third-party libraries and non-testable code. As a result, developers only need to create a single Assembly Definition file to delineate the test-target code, incurring no additional manual filtering effort.

### RC-Comments & Minor Issues

We appreciate the reviewer’s careful reading and constructive comments regarding presentation consistency and clarity. We will fix these issues completely in the camera-ready version and check for other similar issues!