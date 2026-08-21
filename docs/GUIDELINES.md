# 语料格式与替换

## 为什么是数据不是代码

语料是 `osteosarc_agent/guidelines/library/` 下的 JSON 文件（装了 PyYAML 时也接受
`.yaml`）。这样一家获得了指南授权的机构可以直接替换内容，不必改任何 Python：

```bash
python -m osteosarc_agent assess --case demo --corpus /srv/licensed-corpus/
python -m osteosarc_agent guidelines --corpus /srv/licensed-corpus/ --stats
```

装载失败即整体失败——一份跑不起来的语料比一份悄悄少了一半的语料好。

## ⚠️ 内置语料的性质

**仓库内不含任何授权指南原文。** 每条 `statement_zh` 都是公开推荐要点的编辑性转述，
标记为 `provenance: editorial_paraphrase`、`verbatim: false`。控制台每一屏、
CLI 每份报告都会显示这一声明。

替换为授权原文时，把 `provenance` 改为 `licensed_verbatim`、`verbatim` 改为 `true`，
并在 `citation` 里写清版本与检索时间。`corpus.stats()["verbatim_records"]` 会随之变化，
可用于核对本机构语料的授权覆盖率。

## 文件结构

一个语料文件可以包含三类顶层数组，都可选：

```json
{
  "questions":       [ … 临床问题目录 … ],
  "sources":         [ … 指南文件 … ],
  "recommendations": [ … 推荐条目 … ]
}
```

### questions —— 临床问题

```json
{
  "question_id": "q.nutrition.protein_target",
  "label_zh": "每日蛋白质摄入目标",
  "exclusive": true,
  "note": "肌少症营养建议与非透析 CKD 的蛋白限制方向相反。"
}
```

`exclusive` 决定冲突检测怎么读这个问题下的多个答案：

* `true` —— 各项正向建议是**互斥备选**，不同答案即分歧；
* `false`（默认）—— 各项建议**互补**，可以同时成立。

没有这个标注，冲突检测要么对每个多来源话题狼来了，要么漏掉真正的分歧。
**推荐引用的每个 question 必须在此登记**，否则拒绝装载。

### sources —— 指南文件

```json
{
  "source_id": "CN.OP.2022",
  "title_zh": "原发性骨质疏松症诊疗指南（2022）",
  "issuer": "中华医学会骨质疏松和骨矿盐疾病分会",
  "year": 2022,
  "region": "CN",
  "tradition": "western"
}
```

`region` ∈ `CN / US / EU / INTL / APAC`；
`tradition` ∈ `western / tcm / geriatrics / sarcopenia / nutrition / rehab / pharmacy`。
两者都参与冲突呈现：读者需要知道分歧发生在**中美之间**还是**内分泌与老年医学之间**。

### recommendations —— 推荐条目

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `rec_id` | ✓ | 全局唯一，台账按它寻址 |
| `source_id` | ✓ | 必须已定义 |
| `topic` | ✓ | `diagnosis/screening/bone_protection/exercise/nutrition/falls/medication_safety/monitoring/tcm` |
| `question` | ✓ | 必须已登记 |
| `action` | ✓ | 规范化动作键，冲突检测按它分组 |
| `direction` | ✓ | `recommend / require / consider / avoid / against` |
| `strength` | ✓ | `strong / conditional / consensus / expert / good_practice` |
| `statement_zh` | ✓ | 面向医师的推荐正文 |
| `applies_when` | | 适用谓词，缺省为无条件适用 |
| `excluded_when` | | 排除谓词，命中即不适用（**优先于 applies_when**） |
| `subsumes` | | 本动作已满足的其他动作，用于避免把嵌套目标误判为分歧 |
| `evidence_level`、`rationale`、`citation`、`tags` | | 展示与检索用 |

## 谓词语言

```json
{"all": [                                   // 全部满足（取最小值）
  {"fact": "age", "op": ">=", "value": 65},
  {"any": [                                 // 任一满足（取最大值）
    {"fact": "prior_fragility_fracture", "op": "is_true"},
    {"fact": "tscore_min", "op": "<=", "value": -2.5}
  ]},
  {"not": {"fact": "on_antiosteoporosis_therapy", "op": "is_true"}}
]}
```

运算符：`>= > <= < == != is_true is_false in not_in contains any_in known unknown`。

**三值逻辑**：事实缺失 → `unknown`，并向上传播（`all` 取最小、`any` 取最大、
`not` 交换真假、保留未知）。只有 `known` / `unknown` 两个运算符在事实缺失时仍然确定。

一个条目的判定结果因此有四种：

| 状态 | 条件 |
| --- | --- |
| `applies` | 适用为真且排除为假 |
| `excluded` | 排除为真（无论适用如何） |
| `not_applicable` | 适用为假且排除不为真 |
| `insufficient_data` | 任一侧为未知 |

## 装载期校验（全部失败关闭）

* 谓词引用的每个 `fact` 必须在 `guidelines/facts.py` 中声明；
* 每条推荐的 `question` 必须在问题目录中登记；
* `source_id` 必须已定义；
* `rec_id` 全局唯一；
* `direction / strength / region / tradition / provenance` 必须在词表内；
* 谓词结构合法（运算符已知、比较类运算符带 `value`）。

## 新增一个事实

如果授权语料需要一个平台还没有的事实（例如骨小梁分数 TBS），
先在 `guidelines/facts.py` 的 `FACT_SPECS` 里声明：

```python
_f("trabecular_bone_score", "骨小梁分数", "bone", "", source="intake"),
```

声明后它自动获得：中文标签（用于谓词的可读化与推演树）、单位（用于画像面板）、
分组（用于覆盖度统计）。未声明就直接在语料里使用会被装载器拒绝。
