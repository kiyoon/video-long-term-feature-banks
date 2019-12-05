# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""
Build confusion matrix
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cPickle as pkl
import csv
import logging
import numpy as np
import os
import sys


FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

NUM_TEST_SEG = 5281


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


NUM_NOUN_CLASS = 352
NUM_VERB_CLASS = 125
NOUN_CLASSES = range(NUM_NOUN_CLASS)
VERB_CLASSES = range(NUM_VERB_CLASS)

TRAINING_DATA = ['P' + x.zfill(2) for x in map(str, range(1, 26))]  # P01 to P25

def compute_top_k_verbs_or_nouns(scores, labels, K):
    """Compute top-k accuracy for verbs or nouns."""
    assert NUM_TEST_SEG == scores.shape[0]
    assert NUM_TEST_SEG == labels.shape[0]

    correct_count = 0
    for i in range(NUM_TEST_SEG):
        if int(labels[i]) in scores[i].argsort()[-K:]:
            correct_count += 1

    accuracy = 100.0 * float(correct_count) / NUM_TEST_SEG
    logger.info('Top-%d: %.04f%%' % (K, accuracy))


def softmax(x):
    """Row-wise softmax given a 2D matrix."""
    assert len(x.shape) == 2
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def count_train_class_frequency(args):
    num_class = NUM_NOUN_CLASS if args.verb_or_noun == 'noun' else NUM_VERB_CLASS
    classname = args.verb_or_noun + '_class'

    with open(os.path.join(args.annotation_root, 'EPIC_train_action_labels.pkl'), 'rb') as f:
        df = pkl.load(f)

    training_mask = np.isin(df['participant_id'], TRAINING_DATA)
    class_frequency = np.bincount(df[classname][training_mask], minlength=num_class)
    return class_frequency



def save_confusion_matrix(args):
    """save confusion matrix"""

    with open(args.pred_file, 'rb') as f:
        verbnoun_pred, verbnoun_labels = pkl.load(f)

    verbnoun_pred = softmax(verbnoun_pred)

    assert verbnoun_pred.shape[0] == NUM_TEST_SEG
    assert verbnoun_labels.shape[0] == NUM_TEST_SEG


    if args.verb_or_noun == 'noun':
        class_indices = NOUN_CLASSES
        class_labels = pd.read_csv(os.path.join(args.annotation_root, 'EPIC_noun_classes.csv'), quotechar='"', skipinitialspace=True)['class_key']
    else:
        class_indices = VERB_CLASSES
        class_labels = pd.read_csv(os.path.join(args.annotation_root, 'EPIC_verb_classes.csv'), quotechar='"', skipinitialspace=True)['class_key']
    
    pred_labels = np.argmax(verbnoun_pred, axis=1)
    cm = confusion_matrix(verbnoun_labels, pred_labels, labels = class_indices)
    class_frequency_in_train = count_train_class_frequency(args)

    if args.sort_method == 'val_class_frequency':
        # sort by class frequency in validation data
        num_samples_per_target = cm.sum(axis=1)
        sort_labels = num_samples_per_target.argsort()[::-1]

    elif args.sort_method == 'train_class_frequency':
        # sort by class frequency in training data
        sort_labels = class_frequency_in_train.argsort()[::-1]

    else:
        # sort by accuracy per class
        cm = normalize(cm, axis=1, norm='l1')     # row (true labels) will sum to 1.
        sort_labels = cm.diagonal().argsort()[::-1]

    cm_sorted = confusion_matrix(verbnoun_labels, pred_labels, labels = sort_labels)
    num_samples_per_target = cm_sorted.sum(axis=1)  # class frequency in val
    num_correct_pred_per_target = cm_sorted.diagonal()
    class_frequency_in_train = class_frequency_in_train[sort_labels]

    if args.normalise:
        cm_sorted = normalize(cm_sorted, axis=1, norm='l1')     # row (true labels) will sum to 1.

    # Generate visualisation and summary files

    out_dir = 'sort_%s' % args.sort_method
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df_cm = pd.DataFrame(cm_sorted, class_labels[sort_labels],
                              class_labels[sort_labels])
    fig = plt.figure(figsize = (350,250))
    ax = fig.add_subplot(111)
    # x label on top
    ax.xaxis.tick_top()

    sn.set(font_scale=10)#for label size
    sn_plot = sn.heatmap(df_cm, annot=False, annot_kws={"size": 12}, cmap="YlGnBu")# font size
    plt.xlabel('Predicted', fontsize=300)
    plt.ylabel('Target', fontsize=300)

    # This sets the yticks "upright" with 0, as opposed to sideways with 90.
    plt.yticks(fontsize=50, rotation=0) 
    plt.xticks(fontsize=50, rotation=90) 

    fig.set_tight_layout(True)

    plt.savefig('%s/confusion_%s.pdf' % (out_dir, args.verb_or_noun))

    with open('%s/per_class_%s_accuracy.csv' % (out_dir, args.verb_or_noun), mode='w') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=str(','), quotechar=str('"'), quoting=csv.QUOTE_MINIMAL)

	csvwriter.writerow(['class_key', 'accuracy (%)', 'num_correct_pred', 'num_samples_in_val', 'num_samples_in_train'])

        for class_label, num_correct_pred, num_samples_in_val, num_samples_in_train in zip(class_labels[sort_labels], num_correct_pred_per_target, num_samples_per_target, class_frequency_in_train):
            csvwriter.writerow([class_label, float(num_correct_pred) / num_samples_in_val * 100 if num_samples_in_val != 0 else 'NaN', num_correct_pred, num_samples_in_val, num_samples_in_train])

    
    for K in [1, 5]:
        logger.info(args.verb_or_noun + ':')
        compute_top_k_verbs_or_nouns(verbnoun_pred, verbnoun_labels, K)


def main():
    parser = argparse.ArgumentParser(
        description='EPIC-Kitchens verb/noun evaluation and generating confusing matrix')
    parser.add_argument(
        '--pred_file', type=str, required=True, help='Verb/noun prediction results.')
    parser.add_argument(
        '--verb_or_noun', type=str, required=True, choices=['verb', 'noun'], help='Is it verb or noun?')
    parser.add_argument(
        '--no-normalise', action='store_false', dest='normalise', help='Do not normalise the confusion matrix')
    parser.add_argument(
        '--sort_method', type=str, default='train_class_frequency', choices=['train_class_frequency', 'val_class_frequency', 'val_per_class_accuracy'], help='Sorting method')
    parser.add_argument(
        '--annotation_root', type=str, default='data/epic/annotations',
        help='Path to EPIC-Kitchens annotation folder.')

    args = parser.parse_args()
    save_confusion_matrix(args)


if __name__ == '__main__':
    main()
