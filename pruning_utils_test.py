
import unittest
from pruning_utils import load_data, fit_model, backup_caches, create_pruning_config, calculate_accuracy


class TestPruningUtils(unittest.TestCase):
    def test_accuracy_changes_after_pruning_and_restored_properly(self):
        X_train, X_test, y_train, y_test = load_data('digits')
        
        # Use a small subset for speed
        X_train = X_train[:10]
        y_train = y_train[:10]
        X_test = X_test[:30]
        y_test = y_test[:30]

        classifier = fit_model(X_train, y_train, n_estimators=2)
        cache_backup = backup_caches(classifier)
        accuracy_1 = calculate_accuracy(classifier, X_test, y_test)
        pruned_accuracy = calculate_accuracy(
            classifier,
            X_test,
            y_test,
            create_pruning_config(classifier, len(X_train) - 1, same_across_layers=True),
            cache_backup
        )
        accuracy_2 = calculate_accuracy(classifier, X_test, y_test)
        
        self.assertNotEqual(pruned_accuracy, accuracy_1, "Pruning should change the accuracy")
        self.assertEqual(accuracy_1, accuracy_2, "Restored accuracy should match original accuracy")


if __name__ == '__main__':
    unittest.main()
