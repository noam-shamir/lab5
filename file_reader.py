import math


class FileReader:

    def __init__(self, input_file):
        self.file = input_file
        self.words = {}
        self.df = {}
        self.stop_words = []
        self.create_stop_words_list()
        self.create_words_bank()
        self.inv_words = {v: k for k, v in self.words.items()}
        self.doc_amount

    def build_set(self, vector_type, file_to_vector):
        if vector_type == 'boolean':
            return self.build_set_boolean(file_to_vector)
        if vector_type == 'tf':
            return self.build_set_tf(file_to_vector)
        if vector_type == 'tfidf':
            return self.build_set_tfidf(file_to_vector)


    def create_stop_words_list(self):
        with open('./stop_words.txt') as stop_words_file:
            for stop_word in stop_words_file:
                self.stop_words.append(stop_word.rstrip())

    def pre_process_word(self, word_original):
        word = word_original.lower()
        word = word.rstrip().rstrip('!').rstrip('.').rstrip(',').rstrip('?').rstrip(':')
        if word in self.stop_words:
            return ''
        return word

    def create_words_bank(self):
        index = 0
        count_of_lines = 0
        with open(self.file, 'r') as reader: # open the file "file"
            for line in reader: # for each line in file
                count_of_lines += 1
                seen_in_this_line = []
                for word in line.split("\t")[0].split(): # for each word in the line
                    word = self.pre_process_word(word)
                    if word == '':
                        continue
                    if word not in self.df:
                        self.df[word] = 1 # document frequency
                        seen_in_this_line.append(word)
                    if word not in seen_in_this_line:
                        self.df[word] += 1
                        seen_in_this_line.append(word)
                    if word not in self.words.keys(): # if the word doesnt already exists in the words dictionary
                        self.words[word] = index # add it
                        index += 1
            self.doc_amount = count_of_lines

    def build_set_boolean(self, file_to_vector):
        doc_set = {}
        reg_representation = {}
        index = 0
        with open(file_to_vector, 'r') as reader:
            for line in reader:
                vec = len(self.words)*[0,]
                for word in line.split("\t")[0].split():
                    word = self.pre_process_word(word)
                    if word == '':
                        continue
                    vec[self.words[word]] = 1
                doc_class = line.split("\t")[1].rstrip()
                vec.append(doc_class)
                doc_set['doc'+str(index)] = vec
                reg_representation['doc' + str(index)] = line.split("\t")[0]
                index += 1
        return doc_set, reg_representation

    def build_set_tf(self, file_to_vector):
        doc_set = {}
        reg_representation = {}
        index = 0
        with open(file_to_vector, 'r') as reader:
            for line in reader:
                vec = len(self.words) * [0, ]
                for word in line.split("\t")[0].split():
                    word = self.pre_process_word(word)
                    if word == '':
                        continue
                    vec[self.words[word]] += 1
                doc_class = line.split("\t")[1].rstrip()
                vec.append(doc_class)
                doc_set['doc' + str(index)] = vec
                reg_representation['doc' + str(index)] = line.split("\t")[0]
                index += 1
        for row_to_change in range(len(doc_set)):
            for index_to_change in range(len(doc_set['doc' + str(row_to_change)])-1):
                if doc_set['doc' + str(row_to_change)][index_to_change] != 0:
                    doc_set['doc' + str(row_to_change)][index_to_change] = 1 + math.log10(doc_set['doc' + str(row_to_change)][index_to_change])


        # doc_set = {}
        # index = 0
        # for line in reg_represantation.values():
        #     vec = len(self.words) * [0, ]
        #     for word in line.split(" "):
        #         vec[self.words[word]] += 1
        #     doc_class = line.split("\t")[1].rstrip()
        #     vec.append(doc_class)
        #     doc_set['doc' + str(index)] = vec
        #     index += 1
        #
        # for row_to_change in range(len(doc_set)):
        #     for index_to_change in range(len(doc_set[row_to_change])):
        #         if doc_set['doc' + str(row_to_change)][index_to_change] != 0:
        #             doc_set['doc' + str(row_to_change)][index_to_change] = \
        #                 1 +math.log10(doc_set['doc' + str(row_to_change)][index_to_change])
        return doc_set



    def build_set_tfidf(self, file_to_vector):
        doc_set = {}
        reg_representation = {}
        index = 0
        with open(file_to_vector, 'r') as reader:
            for line in reader:
                vec = len(self.words) * [0, ]
                for word in line.split("\t")[0].split():
                    word = self.pre_process_word(word)
                    if word == '':
                        continue
                    vec[self.words[word]] += 1
                doc_class = line.split("\t")[1].rstrip()
                vec.append(doc_class)
                doc_set['doc' + str(index)] = vec
                reg_representation['doc' + str(index)] = line.split("\t")[0]
                index += 1

        upside_dic = {}
        for key, value in self.words.items():
            upside_dic[value] = key

        for row_to_change in range(len(doc_set)):
            for index_to_change in range(len(doc_set['doc' + str(row_to_change)]) - 1):
                if doc_set['doc' + str(row_to_change)][index_to_change] != 0:
                    doc_set['doc' + str(row_to_change)][index_to_change] =\
                        (doc_set['doc' + str(row_to_change)][index_to_change]) *\
                        math.log10(self.doc_amount/self.df[upside_dic[index_to_change]])
        return doc_set
