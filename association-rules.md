### **一、关联规则挖掘算法设计**

#### **1. 算法目标**

从 Food.com 食谱数据中挖掘食材之间的关联关系，发现频繁出现的食材组合（频繁项集），并生成有价值的关联规则（如“面粉 → 鸡蛋”），用于食谱推荐、食材搭配分析等场景。

#### **2. 技术选型**

- **算法**：FP-Growth（频繁模式增长算法），相比 Apriori 算法效率更高，适合处理大规模数据集。
- **框架**：Apache Spark（分布式计算框架），用于高效处理海量数据，适配 8 核 8GB Linux 系统。
- **数据预处理**：使用 UDF（用户自定义函数）清洗食材名称，去除冗余信息（如量词、烹饪术语），确保数据质量。

#### **3. 数据预处理流程**

##### **输入数据**

- `recipes.parquet`：包含食谱 ID、食材列表、类别等信息。
- `reviews.parquet`：包含用户评论数据（代码中未深度使用，主要用于后续扩展）。

##### **处理步骤**

1. **过滤无效数据**：
   - 移除食材列表为空或仅有 1 种食材的食谱（要求至少 2 种食材）。
2. **清洗食材名称**：
   - **去量词**：删除“1 杯”“2 汤匙”等数量描述。
   - **去修饰词**：移除“新鲜”“切碎”“可选”等无关词汇。
   - **标准化**：统一小写、去除标点符号和多余空格（如将“Chocolate, chopped”转为“chocolate”）。
3. **去重与唯一性校验**：
   - 确保每个食谱的食材列表无重复项，使用`array_distinct`函数进一步验证。

**代码映射**：

```python
def clean_and_dedupe_ingredients(ingredients_array):
    # 清洗逻辑：去数字、量词、修饰词、标点等
    # 最终返回唯一食材列表
```

#### **4. FP-Growth 算法参数设计**

| 参数            | 说明                                                                | 代码映射                |
| --------------- | ------------------------------------------------------------------- | ----------------------- |
| `itemsCol`      | 输入列名（预处理后的食材列表）                                      | `"validated_items"`     |
| `minSupport`    | 最小支持度（频繁项集的最小出现比例）                                | `0.005`（默认，可调整） |
| `minConfidence` | 最小置信度（关联规则的最小可信度）                                  | `0.3`（默认，可调整）   |
| `retry机制`     | 若首次训练失败，自动提升`minSupport`至 0.005 重试，避免数据稀疏问题 | `try-except`代码块      |

**核心逻辑**：

```python
fpGrowth = FPGrowth(
    itemsCol="items",
    minSupport=effective_min_support,  # 自动处理极小值
    minConfidence=min_confidence
)
model = fpGrowth.fit(validated_transactions)
```

#### **5. 结果分析与可视化**

##### **（1）频繁项集分析**

- **目标**：识别高频出现的食材组合（如“面粉+鸡蛋+糖”）。
- **输出**：按频率排序的频繁项集列表，包含项集元素和支持度计数。

**代码映射**：

```python
self.frequent_itemsets = model.freqItemsets  # 存储频繁项集
```

##### **（2）关联规则分析**

- **指标**：
  - **置信度（Confidence）**：规则的可信度（如“面粉 → 鸡蛋”的置信度=同时包含两者的食谱数/包含面粉的食谱数）。
  - **提升度（Lift）**：规则的实际有效性（Lift>1 表示正相关）。
- **输出**：按置信度排序的规则列表，可视化展示“置信度-提升度”散点图和 top 规则柱状图。

**代码映射**：

```python
rules_df = pd.DataFrame(rules_data)  # 转换为DataFrame便于分析
# 绘制置信度 vs 提升度散点图
ax1.scatter(rules_df['Confidence'], rules_df['Lift'], s=rules_df['Support']*10000)
```

##### **（3）特定食材关联查询**

- **功能**：查询与目标食材（如“鸡肉”“大蒜”）相关的关联规则，支持自定义置信度阈值。
- **代码映射**：
  ````python
  def find_ingredient_combinations(self, target_ingredient, min_confidence=0.5):
      # 过滤包含目标食材的规则
      target_rules = self.association_rules.filter(
          array_contains(col("antecedent"), target_ingredient) |
          array_contains(col("consequent"), target_ingredient)
      )
      ```
  ````

#### **6. 系统优化策略**

1. **Spark 配置调优**：
   - `spark.driver.memory=4g`：为驱动节点分配更多内存，处理数据聚合与调度。
   - `spark.executor.cores=2`：每个执行器使用 2 核，适配 8 核 CPU 的并行计算。
   - `spark.sql.adaptive.enabled=true`：启用自适应查询优化，自动调整分区数。
2. **数据缓存**：
   - 对预处理后的食谱数据（`processed_recipes`）和挖掘结果（`frequent_itemsets`/`association_rules`）执行`cache()`，避免重复计算。
3. **错误处理**：
   - 捕获 FP-Growth 训练异常，自动提升`minSupport`重试，避免因数据稀疏导致任务失败。
