import math
import sys


class RocchioClassifier:
    def __init__(self, train_set):
        self.training_set = train_set
        self.class_centroids = {}
        self.training()

    def training(self):
        class_size = {}
        for doc_name, document_vector in self.training_set.items():
            doc_class = document_vector[-1]
            if doc_class not in self.class_centroids.keys():
                self.class_centroids[doc_class] = document_vector[0:-1]
                class_size[doc_class] = 1
            else:
                self.class_centroids[doc_class] = [self.class_centroids[doc_class][i] + document_vector[i]
                                                   for i in range(len(document_vector) - 1)]
                class_size[doc_class] += 1
        for c in self.class_centroids.keys():
            for i in range(len(self.class_centroids[c])):
                self.class_centroids[c][i] /= float(class_size[c])

    @staticmethod
    def euclidean_dist(vec1, vec2):
        if len(vec1) != len(vec2):
            print('Error. Vectors of different size')
            print(vec1)
            print(vec2)
            exit(0)

        return sum([(vec1[i] - vec2[i])**2 for i in range(len(vec1))])**0.5

    def old_predict(self, vector):
        winner_class = -1
        lowest_distance = sys.float_info.max
        for class_name, class_vector in self.class_centroids.items():
            distance = self.euclidean_dist(vector, class_vector)
            if distance < lowest_distance:
                winner_class = class_name
                lowest_distance = distance

        return winner_class

    def predict(self, vector, cosine):
        if cosine:
            winner_class = -1
            longest_distance = 0
            for class_name, class_vector in self.class_centroids.items():
                distance = self.cosine_similarity(vector, class_vector)
                if distance > longest_distance:
                    winner_class = class_name
                    lowest_distance = distance
            return winner_class
        else:
            winner_class = -1
            lowest_distance = sys.float_info.max
            for class_name, class_vector in self.class_centroids.items():
                distance = self.euclidean_dist(vector, class_vector)
                if distance < lowest_distance:
                    winner_class = class_name
                    lowest_distance = distance

            return winner_class

    @staticmethod
    def cosine_similarity(vector1, vector2):
        if len(vector1) != len(vector2):
            print('Error. Vectors of different size')
            print(vector1)
            print(vector2)
            exit(0)

        vec_multiply = 0
        div1 = 0
        div2 = 0
        vec_multiply = sum([vector1[i] * vector2[i] for i in range(len(vector1))])
        div1 = sum([(vector1[i])**2 for i in range(len(vector1))]) ** 0.5
        div2 = sum([(vector2[i]) ** 2 for i in range(len(vector2))]) ** 0.5
        return vec_multiply/ (div1 * div2)

