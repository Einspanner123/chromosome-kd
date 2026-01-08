class ErrorAnalyzer:
    """
    简易错误分析器，基于 TIDE 逻辑分析检测错误类型。
    """

    def __init__(self, iou_thr=0.5):
        self.iou_thr = iou_thr

    def analyze(self, results, gts):
        """
        results: list of dict(bboxes, scores, labels)
        gts: list of dict(bboxes, labels)
        """
        summary = {
            "Cls": 0,  # 分类错误: IoU > thr 但类别错
            "Loc": 0,  # 定位错误: 类别对但 0.1 < IoU < thr
            "Both": 0,  # 分类且定位错误
            "Dupe": 0,  # 重复检测
            "Bkg": 0,  # 背景误报: IoU < 0.1
            "Miss": 0,  # 漏检
        }

        # 详细逻辑实现...
        # 这里可以使用 mmdet.evaluation.functional 中的工具函数
        # 为了演示，我们先输出一个结构化的占位
        print("Error Analysis Summary (Mock):")
        for k, v in summary.items():
            print(f"- {k}: {v}")
        return summary


if __name__ == "__main__":
    print("Error Analyzer script initialized. Use this to analyze saved pkl results.")
