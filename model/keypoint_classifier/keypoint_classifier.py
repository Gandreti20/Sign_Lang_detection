#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import csv
import os


class KeyPointClassifier(object):
    def __init__(
        self,
        csv_path='model/keypoint_classifier/keypoint.csv',
        k=5,
    ):
        """
        Instead of loading a tflite model, we now initialize a KNN
        classifier that loads training features and labels from a CSV file.
        """
        self.csv_path = csv_path
        self.k = k
        self.training_features = None
        self.training_labels = None
        self.load_training_data()

    def load_training_data(self):
        features = []
        labels = []
        expected_feature_length = None
        if os.path.exists(self.csv_path):
            with open(self.csv_path, 'r', newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        # The CSV is assumed to have a label as the first element,
                        # followed by the flattened landmark features.
                        try:
                            row_features = list(map(float, row[1:]))
                        except ValueError:
                            # Skip rows with conversion errors
                            continue

                        # Establish the expected feature length based on the first valid row
                        if expected_feature_length is None:
                            expected_feature_length = len(row_features)
                        # Skip rows that do not have the expected number of features
                        if len(row_features) != expected_feature_length:
                            continue

                        labels.append(int(row[0]))
                        features.append(row_features)
        if features:
            self.training_features = np.array(features, dtype=np.float32)
            self.training_labels = np.array(labels, dtype=np.int64)
        else:
            # If no valid training data exists, use empty arrays.
            self.training_features = np.empty((0,))
            self.training_labels = np.empty((0,))

    def __call__(
        self,
        landmark_list,
    ):
        """
        Given a pre-processed landmark list (a one-dimensional list),
        compute distances to each training sample and return the majority
        vote among the k nearest neighbors. If no training data is available,
        return -1.
        """
        sample = np.array(landmark_list, dtype=np.float32)
        # Check if we have training data
        if self.training_features.size == 0:
            return -1

        # Ensure k does not exceed the number of available samples
        k = min(self.k, self.training_features.shape[0])
        # Compute Euclidean distances from the input sample to all training samples
        distances = np.linalg.norm(self.training_features - sample, axis=1)
        # Get indices of the k nearest neighbors
        sorted_indices = np.argsort(distances)[:k]
        nearest_labels = self.training_labels[sorted_indices]
        # Majority vote using bincount (assumes labels are nonnegative integers)
        counts = np.bincount(nearest_labels)
        predicted_label = np.argmax(counts)

        return predicted_label
