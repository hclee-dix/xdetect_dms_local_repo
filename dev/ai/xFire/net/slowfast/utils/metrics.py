#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch
import dev.ai.xFire.net.slowfast.utils.logging as logging
logger = logging.get_logger(__name__)
from sklearn.metrics import confusion_matrix, accuracy_score


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    # import dev.ai.xFire.net.slowfast.utils.logging as logging
    # logger = logging.get_logger(__name__)
    # logger.info(f"PREDS: {preds.size()}, LABELS: {labels.size()}")

    assert preds.size(0) == labels.size(0), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(preds, max(ks), dim=1, largest=True, sorted=True)

    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()

    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)

    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)

    list_label = rep_max_k_labels[0, :].tolist()
    list_pred = top_max_k_inds[0, :].tolist()

    logger.info(f"LIST LABEL: {list_label}")
    logger.info(f"LIST PRED: {list_pred}")

    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]

    # logger.info(f"top_max_k_inds: {top_max_k_inds}, shape: {top_max_k_inds.size()}")
    # logger.info(f"rep_max_k_labels: {rep_max_k_labels}, shape: {rep_max_k_labels.size()}")
    # logger.info(f"top_max_k_correct: {top_max_k_correct}, shape: {top_max_k_correct.size()}")
    #
    # logger.info(f"flatten top_max_k_inds: {torch.flatten(top_max_k_inds)}, shape: {torch.flatten(top_max_k_inds).size()}")
    # logger.info(f"flatten rep_max_k_labels: {torch.flatten(rep_max_k_labels)}, shape: {torch.flatten(rep_max_k_labels).size()}")
    confusion_mat = confusion_matrix(list_label, list_pred)
    logger.info(f"CONFUSION MATRIX: {confusion_mat}, ks: {ks}")

    # logger.info(f"topks_correct: {topks_correct}")
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]

