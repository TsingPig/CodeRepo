ASE 2025 Paper #283 Reviews and Comments
===========================================================================
Paper #283 VRExplorer: A Model-based Approach for Automated Virtual Reality
Scene Testing


Review #283A
===========================================================================

Overall merit
-------------
3. Weak accept

Paper summary
-------------
The rapid expansion of Virtual Reality (VR) applications brings in unique challenges in testing due to complex 3D environments and diverse interactions. This paper proposes VRExplorer, a novel model-based testing tool, to address these challenges with its Entity, Action, and Task (EAT) framework, enabling effective interaction with virtual objects and systematic scene exploration. VRExplorer also incorporates a path-finding algorithm and a Probabilistic Finite State Machine (PFSM) to achieve higher coverage and efficiency. The evaluation validated the enhancement over existing approaches on code coverage and bug detection.

Strengths
---------
+ Testing of VR applications is an important software engineering problem in the emerging domain
+ Comparison with the SOTA approach on multiple datasets and achieve better results
+ Supporting multiple VR-specific testing operations such as grabbing

Weaknesses
----------
- The ablation study considers only removing interactions
- Identified bugs seem to have not been validated by the developers

Detailed comments for authors
-----------------------------
Significance: Testing of AR scenes is a very important problem in software engineering of virtual reality, an emerging area in computer science. Currently, the approaches in the area are still in preliminary stage and missing a lot of support to VR-specific interactions. The paper filled this important gap and the evaluation shows that the proposed work can achieve high coverage. The implementation is based on Unity, which is a dominating framework in the area, so it is expected that the technique and the tool can be potentialy adopted by VR developers. 

Novelty: The problem of VR testing and the missing support of VR-specific interactions such as grabbing are well known in the area. However, this is the first work on automatic VR testing to fill this gap. The model-based testing strategy and the path finding technique are also used before in automatic testing, but they are also the first time being adapted to the VR testing area. The dataset which incorporates an existing dataset and includes more projects is also a good contribution to the area. 

Soundness: The comparison of the proposed approach with SOTA is thorough and on multiple datasets. So I believe the improvement is real and significant. The ablation study could use some more configurations to help understand the effectiveness of its component better. In particular, the current ablation study removes only different interactions in the configuration of variants, but does not consider the variants other components such as navigation algorithms and the information extraction component. It is good to see that the approach is able to find additional real-world bugs. However, I do not find in the paper how these bugs were handled by the developers. Have they been confirmed or fixed by the developers?

Verifiability: The paper has made available the code and project dataset being used in the evaluation. So I believe the work is reproducible. 

Presentation: The paper is easy to read and the approach / evaluation sections are well structured.

Questions for authors' response
-------------------------------
1. Why only interactions are removed in the ablation study but not other components?
2. What happened to the detected real-world bugs?

Artifact check
--------------
3. Satisfactory

Artifact check comments (if any)
--------------------------------
The code and project repositories used in the evaluation are both available. I did not try to set up the tool but the code looks reasonable in a quick scan.



Review #283B
===========================================================================

Overall merit
-------------
2. Weak reject

Paper summary
-------------
The paper presents a model-based testing framework, VRExplorer, for testing virtual reality (VR) applications. The model underlying VRExplorer is constructed based on three key abstractions: VR Entity, Action, and Task. Building on these models, the approach employs a path-finding algorithm and a probabilistic state machine to thoroughly test Unity-based VR applications through in-depth scene exploration and comprehensive interaction with virtual objects. Experimental results show that VRExplorer outperforms the state-of-the-art approach, VRGuide, achieving average performance gains of 72.8% in executable lines of code (ELOC) coverage and 46.9% in method (function) coverage.

Strengths
---------
- Significance (significant for VR test scene coverage)
- Verifiability (Available replication package)        
- Presentation (easy to read)

Weaknesses
----------
-  Soundness (potential bias in evaluation dataset selection)
-  Novelty (modeling process seems more manual)

Detailed comments for authors
-----------------------------
Detailed Comments for the author
----------------------------
#### Significance
-	Developing testing tools for VR applications is both essential and timely. Given that testing VR applications requires sufficient scene coverage and interaction fidelity, tools like VRExplorer are undoubtedly beneficial for helping developers test their applications more effectively and efficiently.
#### Soundness
-	In Section III.A, the paper discusses Project Collection and Analysis. However, the analysis process appears to be only partially automated and largely static in nature. The approach relies heavily on static inspection of scene hierarchies and attached scripts, which raises concerns about its ability to handle dynamic behaviors. Specifically, dynamically instantiated objects or components added at runtime through methods like Instantiate() or AddComponent() may be missed. Moreover, the paper provides only a brief description of the analysis procedure, lacking clear details on how C# scripts and their associated GameObjects are analyzed. A more thorough explanation of the analysis methodology would strengthen the technical soundness of the approach.

-	In Section III.B, the paper discusses the Model Abstraction process. However, the abstraction procedure appears to be largely manual and covers only a limited set of VR object types. In practice, VR applications can include a wide range of objects, such as those involving physical manipulation, trigger-based interactions, and environmental or scene-level components. The paper does not clearly explain how these categories were defined or selected. If the abstraction process is manual, the authors should explicitly state who was involved and outline the steps followed during model construction. Greater transparency here would improve reproducibility and clarify the scalability of the approach.


-	The use of NavMesh for path exploration in VR environments is a compelling and well-justified choice, and I appreciate its integration into the testing framework. However, the construction process for the probabilistic finite state machine (PFSM) model is not clearly described. While the probabilistic modeling approach seems appropriate for the shooting application—where stochastic behaviors can naturally reflect different application states—the paper does not explain how this modeling would extend to other types of VR applications (e.g., puzzle-based experiences or educational simulations). If the PFSM is constructed in an application-specific manner, this raises concerns about the generalizability of the approach. More discussion is needed to clarify whether the model can be adapted across diverse VR domains or if it requires significant manual tailoring per application.

-	The paper states that a constructed dataset was used to perform a preliminary analysis, from which both the object and action abstractions were derived. It appears that this same dataset is also used for evaluating the proposed framework and conducting the comparative analysis with the state-of-the-art approach, VRGuide. This raises a significant risk of data leakage, where insights gained during model construction may inadvertently influence evaluation outcomes. Furthermore, the project selection criteria are insufficiently detailed. It is unclear how diverse the evaluation set is across different types of VR applications. Based on the descriptions, I suspect that the evaluation projects are predominantly shooting-based or have similar event transition steps, which closely align with the modeling assumptions and structure of the proposed approach. This alignment could unintentionally bias the results, potentially overstating the performance advantage over VRGuide. A clearer justification of dataset diversity and separation between training/modeling and evaluation phases is needed to ensure the validity and fairness of the empirical analysis.
-	As part of RQ3, the paper reports that the approach identified three real-world bugs. However, it appears that these bugs were not confirmed by the original developers of the applications. This raises a question about how the authors validated that these were indeed legitimate bugs rather than false positives or expected behaviors. Clarifying the criteria used to label these findings as bugs—whether through manual inspection, heuristics, or expert validation—would strengthen the credibility of this result and help assess the practical impact of the proposed approach.

#### Novelty
-	Overall, I really like  the idea of using a modeling approach and path exploration based on NavMesh—it’s a promising direction. However, I found that much of the current approach appears to be manual and potentially tailored to specific application types, such as shooting games or other similar event-driven VR experiences. To improve its generality and applicability, I believe the approach should incorporate more automated modeling capabilities, driven by static code and GameObject analysis. This would allow it to support a broader range of VR appli
cations and make the method more scalable and adaptable to diverse interaction contexts.

#### Presentation
-   Overall, the paper is easy to read and understand.

#### Verifiability
-    Artifacts are available. However, the readme file seems empty. The readme file should include steps to reproduce the artifact.

Questions for authors' response
-------------------------------
Q1. Based on the Approach and Threats to Validity sections, it appears that the analysis and modeling process is primarily manual. Could you please clarify the specific steps involved in this manual analysis? 

Q2. Do you believe your current modeling process can generalize to a wide range of VR applications beyond the use case(s) demonstrated in the paper? For instance, how would it handle diverse interaction paradigms such as gesture-based input, gaze-based selection, or multi-user collaboration environments?



Review #283C
===========================================================================

Overall merit
-------------
3. Weak accept

Paper summary
-------------
The study proposes VRExplorer, an approach to test the diverse interactions in a virtual environment. VRExplorere is based on EAT framework (Entity, Action, Task), a testing tool, to interact and explore the virtual application. The study outperformed the current state-of-the-art model on 9 VR projects, improving performance by 72.8% and 46.9% in executable lines of code coverage and method coverage, respectively.

Strengths
---------
- Useful Testing Tool.
- Enhanced Interaction Coverage.

Weaknesses
----------
- Confusing subsection heading.
- Missing guidelines to replicate the study.
- Minor Presentation issues.

Detailed comments for authors
-----------------------------
The paper presents an approach based on the EAT (Entity, Action, Task) framework, showcasing results that outperform the baseline model. The proposed method has practical value and can be utilized by developers and test engineers to evaluate their applications and improve overall software quality.

The paper presents a new approach compared to existing methods. While prior approaches rely on a single type of interaction (e.g., mouse click), the proposed method incorporates multiple interaction types such as move, grab, and trigger. This broader action coverage contributes to its improved performance over existing work.

The paper provides a replication package, which enhances the transparency of the work. However, there is no procedure on how we can replicate the study; the readme file only includes the title of the study.

There are some minor presentation issues in the manuscript. The issues include discrepancies in the subheading:
In Section Overview (D. VRExplorer Testing), the title of paragraph two is labeled as "The Entity Interface Layer", whereas the same component is referred to as "Implementation Interface" in Figure 1 and again in Section D (VRExplorer Testing), first paragraph. This inconsistency in naming breaks the coherence of the presentation and may confuse the reader. It would be helpful to standardize the terminology throughout the paper for clarity.

In the Experimental Result section B (Ablation Study), Ablation Experiment Configuration, second paragraph: in the third sentence
“For example, we obtain VRExplorer w/o T by removing the Triggerable
module from the Entity layer’s interface Triggerable, the Action layer’s class TriggerAction, all tasks involve triggering in the Task layer, and the corresponding Mono C# scripts
XRTriggerable.cs. 
“ 
If “w/o” stands for “without,” as clarified in the following sentence, it would improve readability to introduce that abbreviation earlier. Since "w/o" is used without prior explanation, adding “without (w/o)” when it first appears would help readers unfamiliar with the abbreviation.

#### Minor issues

The paper is easy to read and follow, though there are a few minor areas that could be improved.
The layout of Figure 1 (Overview) can be improved by aligning it with the order of the sections (e.g., A, B, C, D). For instance, in the second column, it would enhance clarity if Section C were placed before Section D.

In Table VII, it is stated that underlining indicates the highest performance. However, the table also uses a gray background to highlight certain entries (e.g., VRExplorer w/o T,  VRExplorer w/o Tf ), which visually draws more attention than the underlining. To improve clarity, it would be better to use bold text in addition to underlining to highlight the best-performing results.

Questions for authors' response
-------------------------------
- Will the selected Core code be useful for many other types of VR applications? If not, will it introduce additional workloads for the developers?