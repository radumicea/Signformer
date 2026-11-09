# coding: utf-8
"""
The translation scores: sacreBLEU's BLEU and chrF with their default settings (cased; BLEU with
the 13a tokenizer and exponential smoothing, chrF with character 6-grams and beta 2).
"""
import sacrebleu


def scores(references, hypotheses) -> dict:
    """
    Corpus BLEU-4 and chrF of the hypotheses against the references (one each).

    :param references: list of references (strings)
    :param hypotheses: list of hypotheses (strings)
    :return: bleu (BLEU-4), bleu_precisions (of 1- to 4-grams), chrf, len_ratio (hypothesis /
        reference length in 13a tokens), bp (brevity penalty), bleu_signature, chrf_signature
    """
    bleu_metric, chrf_metric = sacrebleu.metrics.BLEU(), sacrebleu.metrics.CHRF()
    bleu = bleu_metric.corpus_score(hypotheses, [references])
    chrf = chrf_metric.corpus_score(hypotheses, [references])
    return {
        "bleu": bleu.score,
        "bleu_precisions": list(bleu.precisions),
        "chrf": chrf.score,
        "len_ratio": bleu.sys_len / bleu.ref_len if bleu.ref_len else 0.0,
        "bp": bleu.bp,
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf_signature": str(chrf_metric.get_signature()),
    }
