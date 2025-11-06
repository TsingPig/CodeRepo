 [An_Empirical_Study_on_Oculus_Virtual_Reality_Applications_Security_and_Privacy_Perspectives.pdf](An_Empirical_Study_on_Oculus_Virtual_Reality_Applications_Security_and_Privacy_Perspectives.pdf) 

ICSE

VR应用的安全性漏洞、隐私政策缺陷问题研究测试软件 **VR-SP detector**。（PII Data Leaks、 task-hijacking、StrandHogg、IL2CPP、GDPR、 AndroidManifest.xml、IAP、TaintAnalysis）

<img src="C:\Users\TsingPig\AppData\Roaming\Typora\typora-user-images\image-20240831135954674.png" alt="image-20240831135954674" style="zoom:33%;" />

##  

 [An auxiliary development framework for lightweight RPG games based on Unity3D.pdf](An auxiliary development framework for lightweight RPG games based on Unity3D.pdf) 

轻量级辅助RPG游戏开发框架**TRPGFramework**（**ECS框架、性能优化、MVC、LitJSON、BTFlow行为树、UnityEngineProfiler**）

JCR 4区

<img src="C:\Users\TsingPig\AppData\Roaming\Typora\typora-user-images\image-20240830153601729.png" alt="image-20240830153601729" style="zoom:33%;" />



 [Affect-Driven_VR_Environment_for_Increasing_Muscle_Activity_in_Assisted_Gait_Rehabilitation.pdf](Affect-Driven_VR_Environment_for_Increasing_Muscle_Activity_in_Assisted_Gait_Rehabilitation.pdf) 

通过提高肌肉活动的基于情感驱动的辅助康复VR环境。（**Affective Computing**、RAGT、UserStudy）

<img src="C:\Users\TsingPig\AppData\Roaming\Typora\typora-user-images\image-20240831155249550.png" alt="image-20240831155249550" style="zoom:33%;" />



 [An Overview of Blockchain Technology Architecture, Consensus, and Future Trends.pdf](An Overview of Blockchain Technology Architecture, Consensus, and Future Trends.pdf) 

区块链综述：架构、共识和未来趋势（**Bitcoin**、Byzantine Generals Problem、**PoW**、**mining**、**PoS**、**PBFT**、**DPoS**、Ripple、Tendermint）

<img src="C:\Users\TsingPig\AppData\Roaming\Typora\typora-user-images\image-20240902103824525.png" alt="image-20240902103824525" style="zoom:35%;" />

 [Overview_On_Hardware_Characteristics_Of_Virtual_Reality_Systems.pdf](___References\Overview_On_Hardware_Characteristics_Of_Virtual_Reality_Systems.pdf) 

VR设备综述

<img src="C:\Users\TsingPig\AppData\Roaming\Typora\typora-user-images\image-20240904162742219.png" alt="image-20240904162742219" style="zoom:33%;" />

$E=mc^2$

$\sin 30 = \frac{1}{2}$

Markdown + Latex

123456

Typora





# Python 安装指南

Python 是一种流行的编程语言，安装过程非常简单。以下是不同操作系统下的 Python 安装方法：

## Windows 系统安装 Python

1. **下载 Python**

    - 访问 Python 官方网站
    - 点击 "Download Python 3.x.x"（最新稳定版）
    - 下载 Windows 安装程序（64位或32位，根据你的系统选择）

2. **运行安装程序**

    - 双击下载的 `.exe` 文件
    - 勾选 **"Add Python 3.x to PATH"**（重要！这将允许你在命令行中使用 Python）
    - 点击 "Install Now"（默认安装）或 "Customize installation"（自定义安装）
    - 等待安装完成

3. **验证安装**

    - 打开命令提示符（Win+R，输入 `cmd`）
    - 输入 `python --version` 或 `python -V`
    - 如果显示 Python 版本号（如 `Python 3.9.7`），说明安装成功

    

```c++
#include <iostream>
#include <vector>

// 迭代实现二分查找
int binarySearch(const std::vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2; // 防止溢出
        
        if (arr[mid] == target) {
            return mid; // 找到目标，返回索引
        } else if (arr[mid] < target) {
            left = mid + 1; // 在右半部分继续查找
        } else {
            right = mid - 1; // 在左半部分继续查找
        }
    }
    
    return -1; // 未找到目标
}

// 递归实现二分查找
int binarySearchRecursive(const std::vector<int>& arr, int target, int left, int right) {
    if (left > right) {
        return -1; // 基本情况：未找到
    }
    
    int mid = left + (right - left) / 2;
    
    if (arr[mid] == target) {
        return mid;
    } else if (arr[mid] < target) {
        return binarySearchRecursive(arr, target, mid + 1, right);
    } else {
        return binarySearchRecursive(arr, target, left, mid - 1);
    }
}

int main() {
    std::vector<int> arr = {1, 3, 5, 7, 9, 11, 13, 15};
    int target = 7;
    
    // 使用迭代版本
    int result = binarySearch(arr, target);
    if (result != -1) {
        std::cout << "迭代版本: 元素 " << target << " 在索引 " << result << " 处找到" << std::endl;
    } else {
        std::cout << "迭代版本: 元素 " << target << " 未找到" << std::endl;
    }
    
    // 使用递归版本
    result = binarySearchRecursive(arr, target, 0, arr.size() - 1);
    if (result != -1) {
        std::cout << "递归版本: 元素 " << target << " 在索引 " << result << " 处找到" << std::endl;
    } else {
        std::cout << "递归版本: 元素 " << target << " 未找到" << std::endl;
    }
    
    return 0;
}
```

