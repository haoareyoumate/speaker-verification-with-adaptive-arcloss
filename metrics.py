import tensorflow as tf
from sklearn.metrics import roc_curve
import numpy as np

class AverageAngle(tf.keras.metrics.Metric):

    def __init__(self, **kwargs):
        super(AverageAngle, self).__init__(**kwargs)
        self.theta_same_class = self.add_weight(initializer="zeros")
        self.theta_diff_class = self.add_weight(initializer="zeros")
        self.num_utter_same = self.add_weight(initializer="zeros")
        self.num_utter_diff = self.add_weight(initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        theta = tf.math.acos(y_pred)

        theta_same_class = tf.reduce_sum(y_true * theta)
        theta_diff_class = tf.reduce_sum((1 - y_true) * theta)
        num_utter_same = tf.reduce_sum(y_true)
        num_utter_diff = tf.reduce_sum(1-y_true)

        self.theta_same_class.assign_add(theta_same_class)
        self.theta_diff_class.assign_add(theta_diff_class)
        self.num_utter_same.assign_add(num_utter_same)
        self.num_utter_diff.assign_add(num_utter_diff)

    def result(self):
        theta_same_class = self.theta_same_class / (self.num_utter_same + tf.keras.backend.epsilon())
        theta_diff_class = self.theta_diff_class / (self.num_utter_diff + tf.keras.backend.epsilon())
        check_bug = self.num_utter_diff/self.num_utter_same
        # Returning both precision and recall as a dictionary
        return {"angle_same_class": theta_same_class, "angle_diff_class": theta_diff_class}

    def reset_states(self):
        self.theta_same_class.assign(0.0)
        self.theta_diff_class.assign(0.0)
        self.num_utter_diff.assign(0.0)
        self.num_utter_same.assign(0.0)


class EqualErrorRate:

    def __init__(self, dataset, test_size=1680):
        self.dataset = dataset
        self.test_size = test_size
        self.batch_size = self.dataset.element_spec.shape[0]

        mask = tf.linalg.band_part(tf.ones((self.test_size, self.test_size), dtype=tf.bool), 0, -1)
        self.mask = tf.linalg.set_diag(mask, tf.zeros(self.test_size, dtype=tf.bool))




    def pair_vector(self, model):
        all_vectors = tf.TensorArray(tf.float32, size=self.test_size)
        all_labels = tf.TensorArray(tf.int32, size=self.test_size)
        i = 0
        for data, label in self.dataset:
            embed_vectors = model(data)
            embed_vectors = tf.math.l2_normalize(embed_vectors, axis=-1)
            all_vectors = all_vectors.scatter(tf.range(i*self.batch_size, i*self.batch_size + self.batch_size), embed_vectors)
            all_labels = all_labels.scatter(tf.range(i*self.batch_size, i*self.batch_size + self.batch_size), tf.math.argmax(label, axis=-1, output_type=tf.dtypes.int32))
            i += 1
        all_vectors = all_vectors.stack()
        all_labels = all_labels.stack()
        
        cosine_simlarity_matrix = tf.linalg.matmul(all_vectors, all_vectors, transpose_b=True)
        label_for_eer_metric = tf.cast((all_labels == tf.reshape(all_labels,[self.test_size,1])), dtype=tf.float32 )
        y_score = tf.boolean_mask(cosine_simlarity_matrix, self.mask)
        y_true = tf.boolean_mask(label_for_eer_metric, self.mask)
        return y_true.numpy(), y_score.numpy()


    def calculate_eer(self, model):
        y_true, y_score = self.pair_vector(model)

        # Calculate ROC curve and thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        # Calculate differences between FPR and 1 - TPR
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)

        # Find the threshold index where the difference is minimized (closest to equal error rate)
        eer_index = np.argmin(abs_diffs)
        eer = (fpr[eer_index] + fnr[eer_index]) / 2

        return eer