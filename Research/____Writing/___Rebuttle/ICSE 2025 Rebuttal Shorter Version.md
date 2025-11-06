---
title: ICSE 2025 Rebuttal Shorter Version

---

# ICSE 2025 Rebuttal Shorter Version
We'd like to express our gratitude for  providing your valuable feedback. We'll respond to you individually.

## Reviewer A
### RA-Q1:
Given Unity's dominance with >60% market share in XR software development, we believe XRFix can facilitate XR software development.
Our framework can be extended to other XR engines by 1) constructing bug datasets, 2) identifying bugs through tailored static analysis tools, and 3) utilizing LLMs to repa bug repair with diverse prompts.

### RA-Q2:
For the fix rate, we follow Reference [49] in the paper by querying each LLM five times, considering a bug fixed if any of the five samples pass the tests. We also compare the percentage of plausible fixes across all five attempts to minimize randomness. 

For CodeBLEU scores, we take the Wilcoxon signed-rank test at a significance level of α = 0.05 to assess statistical significance. Our findings indicate a statistically significant difference between Prompt c and Prompt d (P-value=0.015), demonstrating that Prompt d outperforms Prompt c in producing reliable responses.

### RA-Comments:
- Novelty
(i) In the XR domain, there's a significant gap on accurate bug detection tools for *performance bugs*. Unlike XRFix, UnityLint primarily addressess *bad practices*. To bridge this gap, we customized both UnityLint and CodeQL to effectively detect these bugs.
(ii) Existing APR methods fix bugs in program codes, but they fall short for XR applications, including diverse virtual objects and 3D scenes. Moreover, the scarcity of datasets such as automated tests complicates the adaptation of APR methods to the XR environment.

- Rigor
1. *Bug Detection Metrics*: As discussed in RQ2, we use *precision* to evaluate bug detection tools.
To verify how two authors agree on the detection result, we employ Cohen’s Kappa coefficient. The result is 0.83 (>0.8), indicating strong consistency.
2. *Misuse of terminology*: In Unity development, *bad practices* are the root cause of performance bugs. 
3. *Bug Complexity*: Referenced from Defects4j V2.0.0, we classified bug complexity based on the location of data flows' source and sink provided by CodeQL. We'll add this in our revision.
4. *Prompt Styles*: The style used in this paper has examined to be the most effective with alternative trials.

## Reviewer B
### RB-Comments
- Rigor
1. *Confidence Interval*: Thanks for your advice, we'll add the confidence intervals of each evaluation metric in our revision.
2. *LLM Settings*: The temperature setting aligns with conventional APR configurations, like ChatRepair. We set the max tokens to be 4000. For other parameters, we use the default values.
3. *Ablation*: As illustrated in Table 4, we've integrated different types of information to guide LLMs. Prompt a includes task descriptions. Prompts b and c expand on this by including bug localization information and specific fix suggestions. For prompt d, we altered the style of the comments. The ablation study on these prompt templates demonstrates the effectiveness of integrating different information.
4. *Patch Merging*: To include all code changes from LLMs, we first utilize Tree-sitter to extract Abstract Syntax Tree (AST) information from the LLM-generated code. This allows us to curate syntax patterns for querying all functions and variables. If these are not relevant to the buggy functions, we replace the original content with the generated content. We then replace the buggy functions with the fixed ones to form a potentially LLM-fixed repository. In the revised version, we will provide more detail using descriptive figures.
5. *Manual Evaluation*: We selected SORL and AOBL because they can demonstrate the performance of bug repair both quantitatively and qualitatively. As outlined in Section 4.6.3, addressing the SORL bug can decrease FPS, whereas fixing the AOBL bug can enhance users' visual experience.


## Reviewer C
### RC-Q1:
Here are our suggestions for real‐world applications:
(1) Efficiency: General LLMs perform better than Code LLMs, with GPT‐4o outperforms GPT‐3.5‐Turbo.
(2) Bug type specificity: Avoid using GPT‐3.5‐Turbo for complex class‐level bugs.
(3) Code LLMs: Deepseek‐Coder delivers more reliable results than CodeLlama and StarChat‐β.
(4) Prompt templates: Prompt c yields more plausible fixes, while responses guided by Prompt d are more reliable. Practitioners should consider these factors during selection.

### RC-Q2:
As far as we know, we're the *first work* to incorporate LLMs for APR in *the XR domain*. We intend to use zero-shot prompting to evaluate our constructed dataset, which however, is largely lacking in XR. Moreover, we also want to evaluate the effectiveness of different LLMs under zero-shot setting, with potential extensions to few-shot, fine-tuned, and other scenarios based on our foundation. 

### RC-Q3:
We consider a bug detection to be successful if the detected issue matches our bug definition. As shown in Table 5, our customized CodeQL queries successfully identify 60 out of 104 (57.7%) bugs without any false positives. Overall, our static analysis tools, which include both existing and customized queries, achieve a high precision rate of 90.4%.

### RC-Q4:
Researchers can extend our XRFix to other bugs of XR apps developed by other engines. Following our ways, they can employ static analysis tools to construct a comprehensive bug repair dataset. Also, our prompt templates can be enhanced to effectively guide LLMs in bug‐fixing. As future work, we'll extend our framework to other XR engines and incorporate advanced techniques such as RAG and agents. 

### RC-Comments
- Novelty 
1. *Agent-based APR*: Thanks for your advice. In our revision, we'll incorporate more recent innovations in APR. We recognize that agent-based approaches face challenges in the XR domain due to a lack of dynamic contextual information, as 79% of VR projects lack automated tests according to Reference [55] in the paper. To address this, we utilize static analysis to generate bug instructions for LLMs with zero-shot prompting, thereby establishing a solid foundation within XR contexts.
2. *RQ Analysis*: For RQ1, we can also conclude in Figure 5 that IDU and RWT are the most universal bugs among the XR projects. In our revision, we'll deliver in-depth discussions on the underlying causes of these bugs.