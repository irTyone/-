

import json
import argparse


TOPIC_LABELS = {
    "0": "中日关系与涉台政治议题",
    "1": "消费市场与产业结构调整",
    "2": "资本市场与投资活动",
    "3": "国际交流与人物访谈报道",
    "4": "企业治理与股权结构",
    "5": "全球发展趋势与研究观点",
    "6": "网络平台舆情与日常资讯",
    "7": "运行管理与经济要素信息",
    "8": "城市发展与区域竞争",
    "9": "跨境犯罪与安全治理",

    "10": "城市安全事件与应急响应",
    "11": "人工智能与前沿科技发展",
    "12": "宏观政策与社会影响",
    "13": "数字经济与资本运作",
    "14": "交通运输与市场波动",
    "15": "体育赛事与文化传播",
    "16": "政府治理与公共效率",
    "17": "法治建设与社会规范",
    "18": "海外社会事件与治安问题",
    "19": "科技企业动态与产业布局",

    "20": "商业生态与媒体观察",
    "21": "美国政治与国际政策动向",
    "22": "国际交流与人员往来",
    "23": "影视产业与文化消费市场",
    "24": "台海局势与周边外交",
    "25": "制造业与科技产品发布",
    "26": "政务信息发布与平台运作",
    "27": "组织管理与社会情绪",
    "28": "竞技体育与体育荣誉",
    "29": "产业监管与发展规划",

    "30": "社会现场事件与公共表达",
    "31": "企业经营与财务表现",
    "32": "安全生产与事故管理",
    "33": "干部管理与廉政监督",
    "34": "体育产业与区域项目发展",
    "35": "生态环境与空间治理",
    "36": "科技创新与产业升级",
    "37": "股票市场与指数波动",
    "38": "金融体系与风险管理",
    "39": "网络舆论与社会反馈",

    "40": "国家战略与高质量发展",
    "41": "公共健康与社会影响评估",
    "42": "宏观经济结构与调整趋势",
    "43": "外交事务与官方表态",
    "44": "国际科技与教育合作",
    "45": "国际经贸与产业博弈",
    "46": "产业投融资与技术应用",
    "47": "地缘政治冲突与安全局势",
    "48": "风险预警与市场规范",
    "49": "医疗体系与健康服务发展"
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_labeled_topics(topic_words_dict):
    """
    topic_words_dict: Dict[str, List[str]]
    """
    if not isinstance(topic_words_dict, dict):
        raise ValueError("输入 JSON 必须是 dict（key=topic_id, value=关键词列表）")

    output = {}

    for tid, words in topic_words_dict.items():
        label = TOPIC_LABELS.get(str(tid), "未定义主题")

        output[str(tid)] = {
            "label": label,
            "keywords": words
        }

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据主题词文件生成统一主题标签 JSON"
    )
    parser.add_argument(
        "--input",
        default="/home/liuyuan/Class_data/model/topics.json",
        help="主题词 JSON 文件"
    )
    parser.add_argument(
        "--output",
        default="/home/liuyuan/Class_data/model/topics_labeled.json",
        help="输出 JSON 文件名"
    )

    args = parser.parse_args()

    topic_words = load_json(args.input)
    labeled_topics = build_labeled_topics(topic_words)

    save_json(labeled_topics, args.output)

    print(f"[OK] 已处理 {len(labeled_topics)} 个主题")
    print(f"[FILE] 输出文件：{args.output}")
