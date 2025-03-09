#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import csv
import os

class PointHistoryClassifier(object):
    def __init__(
        self,
        csv_path='model/point_history_classifier/point_history.csv',
        k=5,
        score_th=0.5,
        invalid_value=0,
    ):
        """
        Initializes a KNN classifier for point history using training data
        loaded from a CSV file. A voting confidence check is done against score_th.
        If the confidence is below score_th, invalid_value is returned.
        """
        self.csv_path = csv_path
        self.k = k
        self.score_th = score_th
        self.invalid_value = invalid_value
        self.training_features = None
        self.training_labels = None
        self.load_training_data()

    def load_training_data(self):
        features = []
        labels = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, 'r', newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        # The CSV is assumed to have label as the first item,
                        # followed by flattened point history features.
                        labels.append(int(row[0]))
                        features.append(list(map(float, row[1:])))
        if len(features) > 0:
            self.training_features = np.array(features, dtype=np.float32)
            self.training_labels = np.array(labels, dtype=np.int64)
        else:
            self.training_features = np.empty((0, ))
            self.training_labels = np.empty((0, ))

    def __call__(
        self,
        point_history,
    ):
        """
        Given a pre-processed point history list (a one-dimensional list),
        this method predicts the class using a simple KNN algorithm.
        It then checks the confidence as the fraction of votes for the predicted label.
        If the confidence is below score_th, invalid_value is returned.
        """
        sample = np.array(point_history, dtype=np.float32)
        if self.training_features.size == 0:
            return self.invalid_value

        k = min(self.k, self.training_features.shape[0])
        distances = np.linalg.norm(self.training_features - sample, axis=1)
        sorted_indices = np.argsort(distances)[:k]
        nearest_labels = self.training_labels[sorted_indices]
        # Majority vote calculation
        counts = np.bincount(nearest_labels)
        predicted_label = np.argmax(counts)
        vote_count = counts[predicted_label]
        confidence = vote_count / k
        if confidence < self.score_th:
            return self.invalid_value
        else:
            return predicted_label
