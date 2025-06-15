# 🍽️ 味觉密码学 | Gustatory Cryptography

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Spark-3.0+-orange.svg)](https://spark.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 通过先进的大数据分析算法，破译隐藏在海量食谱数据中的味觉偏好密码，为每个用户解锁专属的美食世界。

## 🎯 项目概述

味觉密码学项目是一个基于大数据技术的食谱推荐系统，利用机器学习和数据挖掘技术分析用户的饮食偏好，提供个性化的食谱推荐。

### 主要目标
- 🔍 发现食材之间的关联规律
- 👥 识别用户群体的饮食偏好模式
- 🎯 构建精准的个性化推荐系统
- 📊 提供可视化的数据分析结果

## ✨ 功能特性

- [x] **关联规则挖掘** - 发现食材组合的频繁模式
- [x] **聚类分析** - 用户偏好群体识别
- [x] **食谱推荐系统** - 基于协同过滤的个性化推荐
- [x] **特征工程** - 多维度特征提取与优化
- [ ] **实时推荐** - 流式数据处理（开发中）
- [ ] **Web 界面** - 用户友好的交互界面（规划中）

## 🛠️ 技术栈

| 技术 | 用途 | 版本 |
|------|------|------|
| **PySpark** | 分布式数据处理 | 3.0+ |
| **Jupyter Notebook** | 交互式开发环境 | Latest |
| **MLlib** | 机器学习算法库 | Spark 内置 |
| **Spark SQL** | 结构化数据查询 | Spark 内置 |
| **Matplotlib** | 数据可视化 | 3.5+ |
| **Seaborn** | 统计图形可视化 | 0.11+ |

## 🚀 快速开始

### 环境准备

1. **使用 Docker（推荐）**
   ```bash
   # 拉取镜像
   docker pull quay.io/jupyter/pyspark-notebook:latest
   
   # 启动容器
   docker run -d -p 8888:8888 --name notebook quay.io/jupyter/pyspark-notebook:latest
   ```

2. **本地安装**
   ```bash
   # 创建虚拟环境
   conda create -n gustatory python=3.8
   conda activate gustatory
   
   # 安装依赖
   pip install pyspark jupyter matplotlib seaborn pandas numpy
   ```

### 启动服务

访问 Jupyter Notebook：
```
http://localhost:8888
```

## 📊 数据集

### 主要数据源

| 数据集 | 描述 | 规模 | 来源 |
|--------|------|------|------|
| **Food.com Recipes** | 食谱和评论数据 | 500K+ 食谱 | [Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews/) |

### 数据结构
- **食谱信息**：ID、名称、食材、步骤、营养信息
- **用户评论**：评分、评论内容、用户ID、时间戳
- **交互数据**：用户-食谱互动记录

## 🤝 贡献指南

我们欢迎任何形式的贡献！

### 如何贡献
1. Fork 这个仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## Contributors |  贡献者

- [CLZZ](https://github.com/Zephyruston) 