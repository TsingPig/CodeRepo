import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------
# 数据整理
# -------------------------------
data = [
    (2025, "Gewu", "Embodied Agent", "Development", "RL-based playground"),
    (2025, "AUTOVR", "XR", "Testing", "Automated UI exploration"),
    (2025, "XRintTest", "XR", "Testing", "Automated testing"),
    (2025, "DAF", "XR", "Reliability", "Automated testing"),
    (2025, "REALITYCHECK", "XR", "Reliability", "Enhanced W3C PROV"),
    (2025, "XR Attacks", "XR", "Reliability", "Demonstrates side-channel attacks in VR"),
    (2025, "AR UI Security", "XR", "Reliability", "Empirical study of UI security properties"),
    (2024, "Shared-State Attacks", "XR", "Reliability", "Attacks on multi-user AR shared state"),
    (2024, "MARG", "GUI", "Testing", "Multi-Agent RL"),
    (2024, "FaSE4Games", "Game", "Development", "LLM-based test generation"),
    (2024, "ChatDev", "General", "Development", "Chat-powered development"),
    (2024, "Janus", "GUI", "Testing", "Vision-transformer"),
    (2024, "DQT", "GUI", "Testing", "Deep Q-Network"),
    (2024, "VR-SP Detector", "XR", "Reliability", "Empirical study"),
    (2024, "GLIB", "Game", "Testing", "Automated GUI testing"),
    (2023, "VRGuide", "XR", "Testing", "Computational geometry"),
    (2024, "Generative AI VR", "XR", "Testing", "Generative AI"),
    (2024, "Kea", "GUI", "Testing", "Property-based testing"),
    (2024, "VOPA", "XR", "Testing", "Oracle prediction"),
    (2024, "AutoConsis", "GUI", "Testing", "Multimodal model"),
    (2024, "HarmonyOS T.", "GUI", "Testing", "Model-based testing"),
    (2023, "Erebus", "XR", "Reliability", "Access control"),
    (2023, "VOYAGER", "Embodied Agent", "Development", "LLM-driven learning"),
    (2023, "WDTEST", "Game", "Testing", "Widget detection"),
    (2023, "3DSCAN", "Game", "Reliability", "3D model detection"),
    (2023, "Visual Bugs HTML5", "Game", "Testing", "Visual bug detection"),
    (2023, "VR Automated T.", "XR", "Testing", "Empirical study"),
    (2023, "RLbT", "Game", "Testing", "Reinforcement learning"),
    (2023, "PredART", "XR", "Testing", "Prediction model"),
    (2022, "AI2-THOR", "Embodied Agent", "Development", "3D simulation"),
    (2022, "VRTest", "XR", "Testing", "Automated testing"),
    (2022, "RL Load T.", "Game", "Testing", "Reinforcement learning"),
    (2022, "Stoat", "GUI", "Testing", "Model-based testing"),
    (2022, "Fastbot2", "GUI", "Testing", "Model-based testing"),
    (2021, "AdCube", "XR", "Reliability", "WebVR sandboxing"),
    (2021, "EMoGen", "Game", "Development", "Evolutionary model"),
    (2020, "ix4XR & Aplib", "XR", "Testing", "Agent-based testing"),
    (2020, "DRL Game T.", "Game", "Testing", "Deep RL"),
    (2020, "RoboTHOR", "Embodied Agent", "Development", "RoboTHOR"),
    (2019, "Habitat", "Embodied Agent", "Development", "3D simulation"),
    (2019, "OpenAI Five", "Game", "Development", "Self-play RL"),
    (2019, "Robot Vision T.", "Embodied Agent", "Testing", "Robot vision algorithm test platform"),
    (2019, "Wuji", "Game", "Development", "Evolutionary algorithms"),
    (2019, "VR Usability", "XR", "Testing", "Usability evaluation"),
    (2017, "Stoat", "GUI", "Testing", "Model-based testing"),
    (2015, "Platform Game T.", "Game", "Testing", "Model-based testing"),
    (2021, "Habitat 2.0", "Embodied Agent", "Development", "ReplicaCAD dataset, physics-enabled simulator, Home Assistant Benchmark (HAB), RL vs SPA comparison"),
    (2021, "iGibson", "Embodied Agent", "Development", "Interactive large-scale realistic scenes, virtual sensors, domain randomization, motion planning, imitation learning interface"),
    (2024, "LEGENT", "Embodied Agent", "Development", "Open embodied-agent platform, LLM/LMM integration, data generation pipeline")
]


df = pd.DataFrame(data, columns=['Year', 'Method', 'Domain', 'ProblemType', 'Methodology'])

# -------------------------------
# 样式设置
# -------------------------------
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

domain_colors = {
    'GUI': '#1f77b4',
    'Game': '#ff7f0e',
    'XR': '#2ca02c',
    'Embodied Agent': '#d62728',
    'General': '#9467bd'
}

problem_markers = {
    'Development': 'o',
    'Testing': 's',
    'Reliability': '^'
}

# 领域排序权重
domain_order = {'XR': 0, 'Game': 1, 'GUI': 2, 'Embodied Agent': 3, 'General': 4}

# -------------------------------
# 绘制时间轴
# -------------------------------
fig, ax = plt.subplots(figsize=(15, 8))
y_positions = {}
current_y = 0

for year in range(2015, 2026):
    year_data = df[df['Year'] == year].copy()
    if not year_data.empty:
        # 按领域排序
        year_data['DomainRank'] = year_data['Domain'].map(domain_order)
        year_data = year_data.sort_values(by='DomainRank')
        
        y_positions[year] = []
        for i, (idx, row) in enumerate(year_data.iterrows()):
            y_pos = current_y + i * 10
            y_positions[year].append(y_pos)
            
            color = domain_colors.get(row['Domain'], '#7f7f7f')
            marker = problem_markers.get(row['ProblemType'], 'x')
            
            ax.scatter(year, y_pos, c=color, marker=marker, s=150, alpha=0.8, 
                       edgecolors='white', linewidth=1)
            
            ax.text(year + 0.1, y_pos, f"{row['Method']}", 
                    fontsize=10.5, va='center', ha='left',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        current_y = max(y_positions[year]) + 2

# -------------------------------
# 坐标轴和标题
# -------------------------------
ax.set_xlim(2014.5, 2025.5)
ax.set_ylim(-1, current_y)
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Research Methods', fontsize=12, fontweight='bold')
#ax.set_title('Timeline of Research Methods in GUI, Games, XR, and Embodied Agents (2015-2025)', 
 #            fontsize=14, fontweight='bold', pad=20)

# 年份参考线和标签
for year, y_pos_list in y_positions.items():
    if y_pos_list:
        ax.axvline(x=year, ymin=0, ymax=(max(y_pos_list) + 1)/current_y, 
                   color='gray', alpha=0.3, linestyle='--')


# -------------------------------
# 图例
# -------------------------------
from matplotlib.lines import Line2D

domain_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
           markersize=10, label=domain, markeredgewidth=1)
    for domain, color in domain_colors.items()
]

problem_legend_elements = [
    Line2D([0], [0], marker=marker, color='w', markerfacecolor='black', 
           markersize=10, label=problem, markeredgewidth=1)
    for problem, marker in problem_markers.items()
]

legend1 = ax.legend(handles=domain_legend_elements, title="Domain", 
                    loc='upper left', bbox_to_anchor=(0, 1))
legend2 = ax.legend(handles=problem_legend_elements, title="Problem Type", 
                    loc='upper left', bbox_to_anchor=(0.15, 1))
ax.add_artist(legend1)

# -------------------------------
# 网格和y轴
# -------------------------------
ax.grid(True, alpha=0.2)
ax.set_axisbelow(True)
ax.set_yticks([])

plt.tight_layout()
plt.show()

# -------------------------------
# 统计摘要
# -------------------------------
print("=== 统计摘要 ===")
print(f"总研究方法数量: {len(df)}")
print("\n按领域分布:")
print(df['Domain'].value_counts())
print("\n按问题类型分布:")
print(df['ProblemType'].value_counts())
print("\n按年份分布:")
print(df['Year'].value_counts().sort_index())
