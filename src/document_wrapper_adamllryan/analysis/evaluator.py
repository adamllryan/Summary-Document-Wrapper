from typing import Dict, Any, List
import numpy as np
from scipy.stats import kendalltau, spearmanr
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer, util
from document_wrapper_adamllryan.doc.document import Document
import csv


class Evaluator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def evaluate_tvsum(
        self, documents: Dict[str, Document], ground_truths: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates the performance of the document summarization system.

        Args:
            documents: The documents to evaluate.
            ground_truths: The ground truth documents.

        Returns:
            A dictionary of evaluation results.
        """

        results = {}

        for doc_id, document in documents.items():
            if doc_id not in ground_truths:
                print(f"Ground truth not found for {doc_id}")
                continue

            ground_truth = ground_truths[doc_id]

            # TvSum aligns ground truth out of 5 every two seconds.
            # Therefore, we need to align the ground truth to the document.
            # We can do this by utilizing temporal intersection over union (tIoU).

            # Ground Truth info. TvSum provides 20 annotations per video.

            all_rank_correlation = []
            all_precision = []
            all_recall = []
            all_f1 = []
            all_rouge = []
            all_cosine_sim = []

            # Document info

            doc_scores = document.get_aggregate_scores()

            for gt_scores in ground_truth["scores"]:

                # Align the ground truth to the document

                aligned_scores, aligned_gt = self._align_scores(doc_scores, gt_scores)

                rank_correlation = self._compute_rank_correlation(
                    aligned_scores, aligned_gt
                )
                precision, recall, f1 = self._compute_fscore(aligned_scores, aligned_gt)
                rouge = self._compute_rouge(document, ground_truth)
                cosine_similarity = self._compute_embedding_similarity(
                    document, ground_truth
                )

                all_rank_correlation.append(rank_correlation)
                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)
                all_rouge.append(rouge)
                all_cosine_sim.append(cosine_similarity)

            results[doc_id] = {
                "kendall_tau": all_rank_correlation[0],
                "spearman_rho": all_rank_correlation[1],
                "precision": all_precision,
                "recall": all_recall,
                "f1": all_f1,
                "rouge": all_rouge,
                "cosine_similarity": all_cosine_sim,
            }

        return results

    def _align_scores(
        self, doc_scores: List[float], gt_scores: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Aligns the ground truth scores to the document scores.

        Args:
            doc_scores: The document scores.
            gt_scores: The ground truth scores.

        Returns:
            A tuple of aligned document scores and ground truth scores.
        """

        aligned_pred_scores = []
        aligned_gt_scores = []

        for gt in ground_truth:
            gt_timestamp = (gt["timestamp"][0], gt["timestamp"][1])
            gt_score = gt["score"]

            matched_scores = [
                pred["score"]
                for pred in predicted
                if self._temporal_iou(
                    gt_timestamp, (pred["timestamp"][0], pred["timestamp"][1])
                )
                >= self.config.get("iou_threshold", 0.5)
            ]

            if matched_scores:
                aligned_pred_scores.append(np.mean(matched_scores))
            else:
                aligned_pred_scores.append(0)  # No match found

            aligned_gt_scores.append(gt_score)

        return aligned_pred_scores, aligned_gt_scores

    def _temporal_iou(
        self, segment1: (float, float), segment2: (float, float)
    ) -> float:
        """
        Computes Temporal Intersection-over-Union (tIoU).

        Args:
            segment1: (start_time, end_time) for first segment.
            segment2: (start_time, end_time) for second segment.

        Returns:
            IoU score.
        """

        start1, end1 = segment1
        start2, end2 = segment2

        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)

        return intersection / union if union > 0 else 0

    def _compute_rank_correlation(
        self, pred_scores: List[float], gt_scores: List[float]
    ) -> (float, float):
        """
        Computes Kendall’s Tau and Spearman’s Rank Correlation.

        Args:
            pred_scores: List of predicted scores.
            gt_scores: List of ground-truth scores.

        Returns:
            Tuple containing (Kendall’s Tau, Spearman’s Rank Correlation).
        """

        return (
            kendalltau(pred_scores, gt_scores).correlation,
            spearmanr(pred_scores, gt_scores).correlation,
        )

    def _compute_fscore(
        self, pred_scores: List[float], gt_scores: List[float], threshold: int = 3
    ) -> (float, float, float):
        """
        Computes Precision, Recall, and F-score.

        Args:
            pred_scores: Predicted importance scores.
            gt_scores: Ground-truth importance scores.
            threshold: Threshold to consider an important segment.

        Returns:
            Tuple containing (Precision, Recall, F-score).
        """

        y_true = [1 if score >= threshold else 0 for score in gt_scores]
        y_pred = [1 if score >= threshold else 0 for score in pred_scores]

        precision, recall, f_score, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )

        return precision, recall, f_score

    def _compute_rouge(
        self, document: "Document", ground_truth: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Computes ROUGE scores between document summary and ground truth.

        Args:
            document: The document object.
            ground_truth: The ground-truth text.

        Returns:
            A dictionary with ROUGE scores.
        """

        doc_text = " ".join(
            [entry["text"]["text"] for entry in document.get_sentences()]
        )
        gt_text = " ".join([entry["text"] for entry in ground_truth["sentences"]])

        scorer = rouge_scorer.RougeScorer(
            ["rouge-1", "rouge-2", "rouge-l"], use_stemmer=True
        )
        return scorer.score(doc_text, gt_text)

    def _compute_embedding_similarity(
        self, document: "Document", ground_truth: Dict[str, Any]
    ) -> float:
        """
        Computes cosine similarity between SBERT embeddings of document and ground truth.

        Args:
            document: The document object.
            ground_truth: The ground-truth text.

        Returns:
            Cosine similarity score.
        """

        doc_text = " ".join(
            [entry["text"]["text"] for entry in document.get_sentences()]
        )
        gt_text = " ".join([entry["text"] for entry in ground_truth["sentences"]])

        doc_embedding = self.model.encode(doc_text, convert_to_tensor=True)
        gt_embedding = self.model.encode(gt_text, convert_to_tensor=True)

        return util.pytorch_cos_sim(doc_embedding, gt_embedding).item()

    @staticmethod
    def load_tvsum_tsv(file_name: str) -> Dict[str, Any]:
        """
        Reads a TSV file containing TvSum importance scores and converts it into a structured dictionary.

        Args:
            file_name (str): Path to the TSV file.

        Returns:
            Dict[str, Any]: A dictionary mapping video IDs to their respective importance scores.
        """
        tvsum_data = defaultdict(lambda: {"type": None, "scores": []})

        with open(file_name, "r", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t")  # TSV format

            for row in reader:
                if len(row) < 3:
                    continue  # Skip invalid rows

                video_id = row[0].strip()
                annotation_type = row[1].strip()
                scores = list(
                    map(float, row[2].split(","))
                )  # Convert scores from CSV to list

                if tvsum_data[video_id]["type"] is None:
                    tvsum_data[video_id]["type"] = annotation_type

                tvsum_data[video_id]["scores"].append(scores)  # Append annotation set

        return tvsum_data
