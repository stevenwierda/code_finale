import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

#classes = ['BG', 'PET', 'PEF', 'BPET', 'BPEF', '6', '11']
#classes = ['BG', 'square', 'triangle']
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Precision(self):
        true_pos = np.diag(self.confusion_matrix)
        false_pos = np.sum(self.confusion_matrix, axis=0) - true_pos

        precision = (true_pos / (true_pos + false_pos))
        precision = np.nanmean(precision)
        return precision

    def Recall(self):
        true_pos = np.diag(self.confusion_matrix)
        false_neg = np.sum(self.confusion_matrix, axis=1) - true_pos

        recall = (true_pos / (true_pos + false_neg))
        recall = np.nanmean(recall)
        return recall


    def F1_Score(self):

        F1score = ((2 * (self.Precision() * self.Recall())) / (self.Precision()+ self.Recall()))
        F1score = np.nanmean(F1score)
        return F1score

    def IoU(self):
        true_pos = np.diag(self.confusion_matrix)
        false_pos = np.sum(self.confusion_matrix, axis=0) - true_pos
        false_neg = np.sum(self.confusion_matrix, axis=1) - true_pos

        IoU = (true_pos) / (true_pos + false_pos + false_neg)
        IoU = np.nanmean(IoU)
        return IoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        print(confusion_matrix)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def plot_confusion_matrix(self, cm, classes, normalize=True,
                              title=None,
                              cmap=plt.cm.Blues):

        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        ax.margins(15)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, " " + format(cm[i, j], fmt) + " ",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig('test.png')
        plt.close()
        return ax

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


