"""Hand-checkable tests for labels, metric definitions and excluded pixels."""
import unittest
import numpy as np
from metrics import decode_mask, confusion_matrix, metrics


class MetricTests(unittest.TestCase):
    def test_unknown_labels_are_not_silent_background(self):
        rgb=np.array([[[0,0,0],[100,100,100],[255,255,255],[99,99,99]]],dtype=np.uint8)
        np.testing.assert_array_equal(decode_mask(rgb),[[0,1,2,-100]])
        np.testing.assert_array_equal(decode_mask(rgb,'legacy-background'),[[0,1,2,0]])

    def test_known_confusion_and_foreground(self):
        # Four background correct, one vertebra missed, one disc mislabeled bone.
        cm=confusion_matrix(np.array([0,0,0,0,0,1,2]),np.array([0,0,0,0,1,2,2]))
        np.testing.assert_array_equal(cm,[[4,0,0],[1,0,0],[0,1,1]])
        m=metrics(cm)
        self.assertAlmostEqual(m['pixel_accuracy'],5/7)
        self.assertAlmostEqual(m['dice_macro_foreground'],1/3)
        self.assertAlmostEqual(m['miou_foreground'],1/4)
        self.assertAlmostEqual(m['legacy_mean_recall_background_vertebra'],.5)

    def test_ignore_and_absent_class(self):
        cm=confusion_matrix(np.array([0,2,1]),np.array([0,-100,1]))
        m=metrics(cm)
        self.assertEqual(m['valid_pixels'],2)
        self.assertEqual(m['dice_macro_all'],1)
        self.assertIsNone(m['dice_per_class'][2])

    def test_invalid_predictions_rejected(self):
        with self.assertRaises(ValueError):confusion_matrix(np.array([4]),np.array([1]))


if __name__=='__main__':unittest.main()
