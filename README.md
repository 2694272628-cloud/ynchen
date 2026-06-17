# PM 植物图像抠图可视化结果说明

本仓库内容用于课程作业中的植物图像抠图模型评测。主要上传内容是各模型在 PM 植物测试集上的 alpha 预测可视化结果，同时保留用于统一计算指标和生成可视化图的代码。测试集共 150 张植物图像，所有模型使用同一批图像和同一份 GT alpha 进行评估。

需要说明的是，本次比较不是完全无提示自动抠图，而是让不同模型在各自需要的 prompt 条件下完成植物 alpha 预测。为了保证不同模型之间可控对比，prompt 主要由 GT alpha 自动派生得到，因此结果更适合解释为“给定前景提示后的抠图细化能力比较”。

## 使用的模型

本次可视化结果来自以下模型或设置：

| 模型/设置 | 使用的前景提示 | 说明 |
| --- | --- | --- |
| SDMatte | mask prompt | 将由 GT alpha 派生的粗 mask 作为视觉提示，引导模型生成 alpha。 |
| MatAnyone | binary mask prompt | 将植物前景区域转为二值 mask，作为目标对象提示后预测 alpha。 |
| ZIM-base | point prompt | 从 GT alpha 中采样前景点和背景点，使用点提示生成 matte。 |
| ZIM-large | point prompt | 与 ZIM-base 相同，但使用更大的模型配置。 |
| ViTMatte PM_k05 | trimap prompt | 使用较窄或较弱设置的 trimap 作为输入提示。 |
| ViTMatte PM_k15 | trimap prompt | 使用中等设置的 trimap 作为输入提示。 |
| ViTMatte PM_k29 | trimap prompt | 使用较宽或较强设置的 trimap 作为输入提示。 |

各模型在 150 张测试图上的整体指标如下，数值越小表示预测 alpha 与 GT alpha 越接近：

| 模型/设置 | MAE | MSE | SAD |
| --- | ---: | ---: | ---: |
| SDMatte | 0.025615 | 0.014323 | 26.858757 |
| MatAnyone | 0.029986 | 0.018393 | 31.442455 |
| ZIM-large | 0.056556 | 0.039270 | 59.302902 |
| ZIM-base | 0.072558 | 0.053652 | 76.082823 |
| ViTMatte PM_k15 | 0.141187 | 0.101833 | 148.045521 |
| ViTMatte PM_k29 | 0.149228 | 0.109872 | 156.476486 |
| ViTMatte PM_k05 | 0.154136 | 0.119559 | 161.623054 |

## Prompt 的生成方式

本次实验中的 prompt 不是人工逐张标注得到，而是从 GT alpha 自动生成，用于模拟模型推理时给定的前景先验。

MatAnyone 使用 binary mask prompt。具体做法是将 GT alpha 按阈值二值化，得到“哪里是植物目标”的前景 mask。模型接收 RGB 图像和该 mask 后，输出预测 alpha。这个 prompt 较强，能够明确指定植物区域，但模型仍需要处理边界、孔洞和细枝叶细节。

SDMatte 使用 mask prompt。其 prompt 由 GT alpha 阈值化后再经过腐蚀、膨胀等形态学扰动生成，目的是构造一张不完全精确的粗前景 mask。模型利用该视觉提示估计更细致的 alpha matte。由于 SDMatte 推理时出现 OOM，实际使用的 mask-prompt 为 896 尺寸；可视化中将该 prompt 居中放置，并用白色边框补齐到 1024 尺寸，以便和 GT alpha、Pred alpha 保持同一显示尺度。

ViTMatte 使用 trimap prompt。trimap 是三值图：确定背景、未知区域、确定前景。它通常由 GT alpha 的前景和背景区域经过腐蚀/膨胀得到，中间边界带设为 unknown。PM_k05、PM_k15、PM_k29 对应不同 trimap 设置，可以理解为不同未知区域宽度或提示强度下的结果。

ZIM 使用 point prompt。其做法是从 GT alpha 的前景区域中选择若干正点，从背景区域中选择若干负点，再把这些点作为交互式提示输入模型。相比完整 mask 或 trimap，点提示更弱，因此更考验模型根据少量交互点定位植物主体的能力。

## 可视化小图含义

本次提交上传的可视化结果统一为带 prompt 的六图结果，顺序为：

```text
Image | Prompt | GT alpha
Pred alpha | Abs error | Pred cutout
```

其中各小图含义如下：

| 小图名称 | 含义 |
| --- | --- |
| Image | 原始 RGB 植物图像。 |
| Prompt | 模型实际接收的前景提示。MatAnyone 和 SDMatte 主要表现为 mask，ViTMatte 表现为 trimap，ZIM 表现为正负点提示可视化。其中 SDMatte 的白色边框只是 896 prompt 补齐到 1024 尺寸后的显示区域，不表示额外的前景或背景判断。 |
| GT alpha | 数据集提供的人工 alpha 标注，白色表示前景，黑色表示背景，灰色表示半透明或边界过渡。 |
| Pred alpha | 模型预测的 alpha。越接近 GT alpha，说明模型抠图越准确。 |
| Abs error | 预测 alpha 与 GT alpha 的绝对误差热力图，亮色区域表示误差更大。 |
| Pred cutout | 使用 Pred alpha 将植物从原图中抠出，并放在棋盘格背景上显示。 |

观察 Prompt 与 Pred alpha 的关系，可以判断模型是在较强提示下仍然产生边界误差，还是由于提示本身较弱导致目标定位不稳定。

## 评价指标及公式

所有指标都在归一化到 `[0, 1]` 的 alpha 图上计算。设预测 alpha 为 \(P_i\)，GT alpha 为 \(G_i\)，图像共有 \(N\) 个像素。

MAE 表示平均绝对误差：

```text
MAE = (1 / N) * sum(|P_i - G_i|)
```

MSE 表示均方误差：

```text
MSE = (1 / N) * sum((P_i - G_i)^2)
```

SAD 表示绝对误差和，按照图像抠图评测常用写法除以 1000：

```text
SAD = sum(|P_i - G_i|) / 1000
```

MAE 更反映平均像素误差，MSE 对较大的局部错误更敏感，SAD 则能反映一张图整体误差面积的大小。在植物抠图任务中，细叶、枝条、孔洞和低对比边界往往会同时影响这三个指标。

## 代码说明

`eval_pm_alpha.py` 用于读取模型已经输出的 alpha 结果和对应 prompt，统一计算 MAE、MSE、SAD，并生成两行三列的结果可视化图。这样不同模型的预测结果可以在同一评价标准和同一可视化格式下比较。

`make_pm_split.py` 用于生成或更新 PM 测试集 split。它会扫描图像和 alpha 标注文件，按文件名匹配成对样本，并写出统一测试列表，保证各模型使用同一批测试图像。

## 扩展数据集目录说明

仓库中的 `扩展数据集/` 目录包含补充整理的数据样本、resize 后的验证样本，以及四个模型配置在验证样本上的 prompt、预测 alpha、可视化结果和指标文件。

顶层目录结构如下：

```text
扩展数据集/
├── image/
├── alpha/
├── resize_image/
├── resize_alpha/
├── splits/
├── MatAnyone/
├── SDMatte/
├── ViTMatte_k15/
└── ZIM_large/
```

其中，`image/` 和 `alpha/` 保存 10 张整理后的扩展数据集样本：

```text
image/1.jpg  ... image/10.jpg
alpha/1.png  ... alpha/10.png
```

`resize_image/` 和 `resize_alpha/` 保存从扩展数据集中选出的 5 张 plant-only 验证图片及其 resize 后的 alpha 标注：

```text
resize_image/2.jpg   resize_alpha/2.png
resize_image/4.jpg   resize_alpha/4.png
resize_image/5.jpg   resize_alpha/5.png
resize_image/8.jpg   resize_alpha/8.png
resize_image/10.jpg  resize_alpha/10.png
```

`splits/test.txt` 保存这 5 张验证图片对应的测试列表。

四个模型配置的验证推理结果分别保存在以下目录：

```text
MatAnyone/
SDMatte/
ViTMatte_k15/
ZIM_large/
```

各模型目录中的文件含义如下：

| 路径 | 内容 |
| --- | --- |
| `alpha/` | 模型在 5 张 resize 验证图片上推理得到的预测 alpha。 |
| `prompt/` | 模型验证推理时使用的 prompt 可视化文件。 |
| `vis/` | 由评估脚本生成的六宫格可视化结果，包含 Image、Prompt、GT alpha、Pred alpha、Abs error 和 Pred cutout。 |
| `metrics_summary.json` | 5 张验证图片上的平均 MAE、MSE、SAD。 |
| `metrics_per_image.csv` | 每张验证图片的 MAE、MSE、SAD 明细。 |
| `difference_sorted.csv` | 按 SAD 从高到低排序后的逐图指标表。 |

四个模型配置在 5 张 resize 验证图片上的指标汇总如下：

| 模型/设置 | 图像数 | MAE | MSE | SAD |
| --- | ---: | ---: | ---: | ---: |
| ViTMatte_k15 | 5 | 0.017528 | 0.002787 | 18.379132 |
| ZIM_large | 5 | 0.020920 | 0.005341 | 21.936423 |
| SDMatte | 5 | 0.025153 | 0.008292 | 26.374311 |
| MatAnyone | 5 | 0.042342 | 0.024787 | 44.398975 |
