import json
import re
from collections.abc import Iterable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..models.analysis import PaperAnalysis
from ..models.document import Document
from ..models.page import DocumentPage
from .llm import complete

ANALYSIS_KEYS = {
    "summary_zh",
    "innovations",
    "methodology",
    "experiments",
    "limitations",
    "chart_insights",
}

TERM_MAP = {
    "spectral clustering": "谱聚类",
    "deep spectral": "深度谱聚类",
    "bootstrap": "bootstrap 自助采样",
    "ensemble": "集成学习",
    "optimal transport": "最优传输",
    "bounded plane": "有界平面分段",
    "plane segment": "平面分段",
    "local affinity": "局部亲和关系",
    "nonlinear manifold": "非线性流形",
    "manifold": "流形结构",
    "unsupervised": "无监督学习",
    "clustering": "聚类",
}

METHOD_HINTS = (
    "method",
    "approach",
    "algorithm",
    "framework",
    "model",
    "architecture",
    "pipeline",
    "we propose",
    "we present",
)

EXPERIMENT_HINTS = (
    "experiment",
    "evaluation",
    "result",
    "benchmark",
    "dataset",
    "baseline",
    "nmi",
    "ari",
    "accuracy",
    "ablation",
)

LIMITATION_HINTS = (
    "limitation",
    "future work",
    "slower",
    "cost",
    "expensive",
    "sensitive",
    "depends on",
    "large datasets",
)

FIGURE_HINTS = ("figure", "fig.", "table", "chart", "plot")


def fallback_analysis(title: str, context: str) -> dict[str, str]:
    """Create useful local paper notes when the remote LLM is unavailable."""
    text = _clean(context)
    terms = _detect_terms(f"{title} {text}")
    method_sentences = _matching_sentences(text, METHOD_HINTS, limit=3)
    experiment_sentences = _matching_sentences(text, EXPERIMENT_HINTS, limit=3)
    limitation_sentences = _matching_sentences(text, LIMITATION_HINTS, limit=2)
    figure_sentences = _matching_sentences(text, FIGURE_HINTS, limit=3)

    topic = _topic_phrase(terms)
    method_focus = _method_focus(terms, method_sentences)
    experiment_focus = _experiment_focus(experiment_sentences)

    return {
        "summary_zh": (
            f"本文《{title}》主要研究{topic}问题。"
            f"从已解析内容看，论文的核心思路是{method_focus}。"
            f"{experiment_focus}"
        ),
        "innovations": _build_innovations(terms, method_sentences),
        "methodology": _build_methodology(terms, method_sentences),
        "experiments": _build_experiments(experiment_sentences),
        "limitations": _build_limitations(limitation_sentences),
        "chart_insights": _build_chart_insights(figure_sentences),
    }


def analyze_document(db: Session, document: Document) -> PaperAnalysis:
    pages = list(
        db.scalars(
            select(DocumentPage)
            .where(DocumentPage.document_id == document.id)
            .order_by(DocumentPage.page_number)
        )
    )
    context = "\n\n".join(f"Page {page.page_number}: {page.text}" for page in pages)
    data = fallback_analysis(document.title, context)

    llm_data = _llm_analysis(document.title, context)
    if llm_data:
        data.update(llm_data)

    db.execute(delete(PaperAnalysis).where(PaperAnalysis.document_id == document.id))
    analysis = PaperAnalysis(document_id=document.id, **data)
    db.add(analysis)
    document.status = "analyzed"
    db.add(document)
    db.commit()
    db.refresh(analysis)
    return analysis


def _llm_analysis(title: str, context: str) -> dict[str, str] | None:
    prompt = f"""
请你作为中文论文阅读助手，基于下面的论文/PPT解析文本生成严格 JSON。
不要复述大段原文，不要输出 Markdown，不要把英文正文整体塞进字段。

JSON 必须包含这些字符串字段：
summary_zh, innovations, methodology, experiments, limitations, chart_insights

写作要求：
1. summary_zh 用 3-5 句中文说明研究问题、核心方法和主要结论。
2. innovations 用中文项目符号总结创新点。
3. methodology 解释方法流程和关键模块。
4. experiments 总结数据集/指标/对比/消融/结论，缺失时说明“解析文本中未明确给出”。
5. limitations 总结局限与可追问点。
6. chart_insights 解读图表或说明当前解析文本中未检测到明确图表标题。

标题：{title}
解析文本：
{context[:10000]}
"""
    llm_text = complete(prompt)
    if not llm_text:
        return None
    return _parse_llm_json(llm_text)


def _parse_llm_json(text: str) -> dict[str, str] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if match:
        stripped = match.group(0)

    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(raw, dict):
        return None

    parsed: dict[str, str] = {}
    for key in ANALYSIS_KEYS:
        value = raw.get(key)
        if isinstance(value, list):
            value = "\n".join(f"- {item}" for item in value)
        if isinstance(value, str) and value.strip():
            parsed[key] = value.strip()

    return parsed if parsed else None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _sentences(text: str) -> list[str]:
    compact = _clean(text)
    if not compact:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", compact)
    return [part.strip() for part in parts if len(part.strip()) > 12]


def _matching_sentences(text: str, hints: Iterable[str], limit: int) -> list[str]:
    hint_set = tuple(hint.lower() for hint in hints)
    matches = [
        sentence
        for sentence in _sentences(text)
        if any(hint in sentence.lower() for hint in hint_set)
    ]
    return matches[:limit]


def _detect_terms(text: str) -> list[str]:
    lowered = text.lower()
    terms = [zh for en, zh in TERM_MAP.items() if en in lowered]
    return list(dict.fromkeys(terms))


def _topic_phrase(terms: list[str]) -> str:
    if not terms:
        return "论文所描述的研究任务"
    selected = "、".join(terms[:4])
    return f"围绕{selected}展开的研究"


def _method_focus(terms: list[str], method_sentences: list[str]) -> str:
    if "有界平面分段" in terms or "平面分段" in terms:
        return "通过构造局部几何分段来刻画复杂流形结构"
    if "最优传输" in terms and "bootstrap 自助采样" in terms:
        return "结合 bootstrap 集成与最优传输来稳定聚类结果"
    if "谱聚类" in terms:
        return "围绕谱聚类框架改进相似度建模和聚类稳定性"
    if method_sentences:
        return "围绕方法章节中提到的模型、流程或算法进行改进"
    return "需要结合更多正文页面进一步确认方法细节"


def _experiment_focus(experiment_sentences: list[str]) -> str:
    joined = " ".join(experiment_sentences)
    metrics = []
    for metric in ("NMI", "ARI", "accuracy", "F1", "AUC"):
        if metric.lower() in joined.lower():
            metrics.append(metric)

    if experiment_sentences and metrics:
        return f"实验部分提到使用 {', '.join(metrics)} 等指标评估，与基线方法比较后展示性能改进。"
    if experiment_sentences:
        return "实验部分包含数据集、基线或结果描述，可继续追问具体数值和对比设置。"
    return "当前解析文本中实验细节较少，建议查看 Results、Experiments 或 Evaluation 页面。"


def _build_innovations(terms: list[str], method_sentences: list[str]) -> str:
    points: list[str] = []
    if "bootstrap 自助采样" in terms and "谱聚类" in terms:
        points.append("将 bootstrap 自助采样/集成思想引入谱聚类，以提升聚类结果稳定性。")
    if "最优传输" in terms:
        points.append("利用最优传输对不同采样或分区结果进行对齐与融合。")
    if "有界平面分段" in terms or "平面分段" in terms:
        points.append("用局部几何分段近似非线性流形结构，增强对复杂簇形状的表达能力。")
    if not points and method_sentences:
        points.append("围绕作者提出的方法流程进行改进，创新点集中在任务建模、算法模块或训练/推理流程。")
    if not points:
        points.append(
            "解析文本中未明确出现创新点标题，建议补充 Introduction 或 Contribution 页面后重新分析。"
        )
    return "\n".join(f"- {point}" for point in points)


def _build_methodology(terms: list[str], method_sentences: list[str]) -> str:
    parts: list[str] = []
    if "局部亲和关系" in terms:
        parts.append("先估计样本之间的局部亲和关系，作为后续图结构或谱分解的基础。")
    if "有界平面分段" in terms or "平面分段" in terms:
        parts.append("再用平面分段描述局部流形几何，降低非球形簇结构带来的建模难度。")
    if "bootstrap 自助采样" in terms:
        parts.append("随后通过 bootstrap 生成多个扰动视角，形成集成聚类结果。")
    if "最优传输" in terms:
        parts.append("最后使用最优传输融合不同分区，减少单次聚类的不稳定性。")
    if not parts and method_sentences:
        parts.append("方法章节表明论文围绕模型/算法流程做了设计，但当前解析文本不足以还原完整步骤。")
    if not parts:
        parts.append("当前解析文本没有明显的方法章节，请补充更多页面或在问答区追问具体方法。")
    return "方法分析：" + "\n".join(f"\n- {part}" for part in parts)


def _build_experiments(experiment_sentences: list[str]) -> str:
    joined = " ".join(experiment_sentences)
    baselines = []
    for name in ("k-means", "spectral clustering", "deep clustering"):
        if name in joined.lower():
            baselines.append(name)

    metrics = []
    for metric in ("NMI", "ARI", "accuracy", "F1", "AUC"):
        if metric.lower() in joined.lower():
            metrics.append(metric)

    if not experiment_sentences:
        return (
            "实验结论：解析文本中未明确给出实验设置。"
            "建议补充 Results、Experiments、Evaluation 或表格页面后重新分析。"
        )

    pieces = ["实验结论：文本中检测到实验/评估相关内容。"]
    if metrics:
        pieces.append(f"评价指标包括 {', '.join(metrics)}。")
    if baselines:
        pieces.append(f"对比基线包括 {', '.join(baselines)}。")
    pieces.append("从描述看，作者声称方法在相关数据集或基准上取得改进；具体数值仍需结合表格逐项核对。")
    return "".join(pieces)


def _build_limitations(limitation_sentences: list[str]) -> str:
    joined = " ".join(limitation_sentences).lower()
    points = []
    if "slower" in joined or "large datasets" in joined or "cost" in joined:
        points.append("计算成本或运行速度可能是主要局限，尤其是在大规模数据集上。")
    if "depends on" in joined or "sensitive" in joined:
        points.append("方法可能对关键超参数、采样次数或数据分布较敏感。")
    if not points and limitation_sentences:
        points.append("论文提到了局限或未来工作，但需要结合对应段落进一步核对具体原因。")
    if not points:
        points.append("解析文本中未检测到明确局限性描述，可重点追问泛化性、复杂度、消融实验和失败案例。")
    return "\n".join(f"- {point}" for point in points)


def _build_chart_insights(figure_sentences: list[str]) -> str:
    if not figure_sentences:
        return (
            "图表解读：当前解析文本中未检测到明确 Figure/Table 标题。"
            "系统已保存页面截图，可后续接入视觉模型或根据具体页码继续追问图表含义。"
        )
    return "图表解读：" + "\n".join(
        f"\n- 检测到图表线索：{_shorten(sentence, 180)}" for sentence in figure_sentences
    )


def _shorten(text: str, limit: int) -> str:
    compact = _clean(text)
    return compact if len(compact) <= limit else compact[: limit - 1] + "..."
