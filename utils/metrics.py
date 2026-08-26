"""
Evaluation metrics for EEG-to-Text generation
"""

from typing import List, Dict
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
try:
    from bert_score import score as bert_score_func
except ImportError:
    bert_score_func = None


def _bleu_weights(n: int):
    if n == 1:
        return (1.0, 0.0, 0.0, 0.0)
    if n == 2:
        return (0.5, 0.5, 0.0, 0.0)
    if n == 3:
        return (1 / 3, 1 / 3, 1 / 3, 0.0)
    return (0.25, 0.25, 0.25, 0.25)


def compute_bleu(reference: List[str], candidate: List[str], n: int = 4):
    """Sentence-level BLEU-n (diagnostic only)."""
    smoothing = SmoothingFunction().method1
    try:
        return sentence_bleu(
            [reference],
            candidate,
            weights=_bleu_weights(n)[:n],
            smoothing_function=smoothing
        )
    except Exception:
        return 0.0


def compute_corpus_bleu(references: List[List[str]], candidates: List[List[str]], n: int = 4) -> float:
    """Corpus BLEU-n, the score reported in experiment tables."""
    if len(references) == 0:
        return 0.0
    smoothing = SmoothingFunction().method1
    try:
        return float(corpus_bleu(
            [[ref] for ref in references],
            candidates,
            weights=_bleu_weights(n),
            smoothing_function=smoothing
        ))
    except Exception:
        return 0.0


def compute_rouge(reference: str, candidate: str, rouge_type: str = 'rouge1'):
    """
    Compute ROUGE score
    
    Args:
        reference: Reference text
        candidate: Candidate text
        rouge_type: Type of ROUGE ('rouge1', 'rouge2', 'rougeL')
        
    Returns:
        rouge_scores: Dictionary with 'precision', 'recall', 'fmeasure'
    """
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores[rouge_type]


def compute_wer(reference: List[str], candidate: List[str]) -> float:
    """
    Word error rate: (substitutions + deletions + insertions) / reference length.
    """
    if len(reference) == 0:
        return 0.0 if len(candidate) == 0 else 1.0

    n, m = len(reference), len(candidate)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == candidate[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost
            )
    return float(dp[n, m]) / float(n)


def compute_bertscore(references: List[str], candidates: List[str]):
    """
    Compute BERTScore
    
    Args:
        references: List of reference texts
        candidates: List of candidate texts
        
    Returns:
        precision: BERTScore precision
        recall: BERTScore recall
        f1: BERTScore F1
    """
    if bert_score_func is None:
        raise ImportError("bert_score not installed. Install with: pip install bert-score")
    
    P, R, F1 = bert_score_func(
        candidates,
        references,
        lang='en',
        verbose=False
    )
    
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }


def evaluate_predictions(
    references: List[List[str]],
    candidates: List[List[str]],
    compute_bert: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of predictions
    
    Args:
        references: List of reference token sequences
        candidates: List of candidate token sequences
        compute_bert: Whether to compute BERTScore (slower)
        
    Returns:
        metrics: Dictionary of metric scores
    """
    metrics = {}
    
    # Convert token lists to strings for ROUGE
    ref_strings = [' '.join(ref) for ref in references]
    cand_strings = [' '.join(cand) for cand in candidates]
    
    wer_scores = []
    rouge_scores = {
        'rouge1': {'precision': [], 'recall': [], 'fmeasure': []},
        'rouge2': {'precision': [], 'recall': [], 'fmeasure': []},
        'rougeL': {'precision': [], 'recall': [], 'fmeasure': []}
    }

    for ref, cand in zip(references, candidates):
        wer_scores.append(compute_wer(ref, cand))
        ref_str = ' '.join(ref)
        cand_str = ' '.join(cand)
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            rouge = compute_rouge(ref_str, cand_str, rouge_type)
            rouge_scores[rouge_type]['precision'].append(rouge.precision)
            rouge_scores[rouge_type]['recall'].append(rouge.recall)
            rouge_scores[rouge_type]['fmeasure'].append(rouge.fmeasure)

    for n in range(1, 5):
        metrics[f'bleu_{n}'] = compute_corpus_bleu(references, candidates, n) * 100
    metrics['wer'] = float(np.mean(wer_scores) * 100) if wer_scores else 0.0

    # Average ROUGE scores
    for rouge_type, scores_dict in rouge_scores.items():
        for metric_name, scores in scores_dict.items():
            key = f'{rouge_type}_{metric_name[0].upper()}' if metric_name == 'fmeasure' else f'{rouge_type}_{metric_name}'
            metrics[key] = np.mean(scores) * 100
    
    # BERTScore
    if compute_bert and bert_score_func is not None:
        try:
            bert_scores = compute_bertscore(ref_strings, cand_strings)
            metrics['bertscore_precision'] = bert_scores['precision'] * 100
            metrics['bertscore_recall'] = bert_scores['recall'] * 100
            metrics['bertscore_f1'] = bert_scores['f1'] * 100
        except:
            pass
    
    return metrics


def detokenize(tokens: List[str], tokenizer) -> str:
    """
    Convert tokens back to text string
    
    Args:
        tokens: List of tokens
        tokenizer: Tokenizer object
        
    Returns:
        text: Detokenized text string
    """
    if hasattr(tokenizer, 'decode'):
        return tokenizer.decode(tokens)
    else:
        return ' '.join(tokens)

