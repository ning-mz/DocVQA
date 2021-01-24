# XJTLU - PremiLab
# Author: Maizhen Ning
# This file for loading data form given dataset


from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import os
import json
import collections
import string
import cv2
from .data_loaders_utils import Document, Question, InputFeatures
from transformers import BertTokenizer



class DocVQADataLoader(Dataset):
    def __init__(self, data_dir: str = None,
                qa_dir: str = None,
                max_sequence_length: int = 512,
                max_question_length: int = 128,
                window_stride: int = 128,
                training_flag = True):

        self.max_sequence_length = max_sequence_length
        self.max_question_length = max_question_length
        self.training_flag = training_flag
        self.window_stride = window_stride

        self.documents_dict, self.q_count = self.load_data(data_dir, qa_dir)
        # self.tokenizer = BertTokenizer.from_pretrained("/Data_HDD/phd20_maizhen_ning/Projects/pytorch-template/utils/bert-base-uncased-vocab.txt", do_lower_case=True) # layoutlm used autotokenizer
        self.tokenizer = None
        self.features = None
        self.num_questions = None
        self.words_dict = None
        # get features for BERT


    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.generate_dataset()

    def generate_dataset(self):
        self.features, self.num_questions, self.words_dict = self.convert_to_features(self.documents_dict, self.tokenizer)
        print('Questions: original total: {}, actual useable: {}'.format(self.q_count, self.num_questions))



    def load_data(self, folder_dir, qa_dir):
        q_count = 0 # number of questions loaded

        # if it is trainin, load QA base file, for test, it will not include the answer
        with open(qa_dir, 'r', encoding='utf-8') as f:
            quesiton_set = json.load(f)['data'] # load a list of QA, each one is a dict

        documents_dict = {} # a dict for store documents by using docId
        for question in quesiton_set: # all of questions (and answers), each is a dict
            if documents_dict.__contains__(question['docId']):
                documents_dict[question['docId']].add_question(Question(question, self.training_flag))
                q_count = q_count + 1
            else:
                # create a new document object and add into the list
                # image_tmp = cv2.imread(folder_dir+'/'+question['image'])
                with open(folder_dir+'/ocr_results/'+question['image'][10:-4] + '.json') as f:
                    tmp = json.load(f)['recognitionResults'][0] # sentence level
                    w = tmp['width']
                    h = tmp['height']
                    text_tmp = tmp['lines']
                documents_dict[question['docId']] = Document(docId=question['docId'], image=None,
                                                             text=text_tmp, width=w, height=h,
                                                             question_list=[Question(question, self.training_flag)])
                q_count = q_count + 1

        return documents_dict, q_count



    # this function will return a list of feature objects for LayoutLM
    def convert_to_features(self, documents_dict, tokenizer):
        feature_list = []
        matched_num = 0
        unique_id = 0
        words_dict = {}

        for key in documents_dict: # for each document object in the input list
            document = documents_dict[key]
            words_list = []
            concatenated_sequences = []
            indexes_concatenated = []
            boxes_list = []
            current_index = 0

            for i in document.text: # this is a list of many dict
                for j in i['words']: # this is a list of many dict
                    words_list.append(j['text'])
                    concatenated_sequences.extend(list(j['text'].lower()))
                    indexes_concatenated.extend([current_index] * len(list(j['text'])))
                    boxes_list.append(j['boundingBox'])
                    current_index+=1

            # get only characters from OCR and matched indexes of box, with lower case
            purged_concatenated_sequences, purged_indexes_concatenated = self.remove_punctuation(concatenated_sequences, indexes_concatenated)

            # for each question, we create a set of features for current document
            for ques in document.question_list:
           
                start_position = -1
                end_position = -1
                # check whether it exist ground truth
                if self.training_flag:
                    answer, _ = self.remove_punctuation(ques.answer_text.lower(), [-1] * len(ques.answer_text))
                    matched = False


                    # this part should be improved in future!!!!!!!!!!
                    if len(answer) > 1:
                        for i in range(max(0, len(purged_concatenated_sequences) - len(answer) + 1)):
                            if purged_concatenated_sequences[i : i + len(answer)] == answer:
                                matched = True
                                start_position = purged_indexes_concatenated[i]
                                end_position = purged_indexes_concatenated[i + len(answer) - 1]



                    if matched: # get ground truch in OCR, add features
                        # actually above part is similar to the 'examples' in LayoutLM code
                        matched_num += 1
                        feature_list.extend(self.generate_feature(words_list, boxes_list, ques.question_text, start_position, end_position, ques.questionId, unique_id, document.width, document.height))

                    else:
                        # now just drop examples not matched
                        pass
                else: # preparing for test data
                
                    feature_list.extend(self.generate_feature(words_list, boxes_list, ques.question_text, None, None, ques.questionId, unique_id, document.width, document.height))
                    words_dict[ques.questionId] = words_list
                    matched_num += 1

        return feature_list, matched_num, words_dict


    # for each quesiton in one document generate a set of features
    def generate_feature(self, transcripts, boxes, question, start_position, end_position, question_id, unique_id, width, height):


        # transcripts is a list of all words in a document
        # this if for get result
        transcripts_location = [i for i in range(len(transcripts))] # a increasing list start from 0
        new_transcripts_location = [] 


        if self.training_flag:
            assert start_position <= end_position
    

        result = []
        tokenized_words_list = []
        tokenized_boxes_list = []
        tokenized_transcripts_location_list = []
        zero_padding_box = [0, 0, 0, 0]
        sep_token_box = [1000, 1000, 1000, 1000]
        

        # process quesiton to tokens
        question_tokens = self.tokenizer.tokenize(question)
        if len(question_tokens) > self.max_question_length: 
            question_tokens = question_tokens[:self.max_question_length]

        max_tokens_of_words = self.max_sequence_length - len(question_tokens) - 3


        new_start_position = 0
        new_end_position = 0


        if self.training_flag:             
            count = 0 # for calculating new index of 2 positions
            # process words from documents to tokenization
            for index, words in enumerate(transcripts):
                if index == start_position:
                    new_start_position = start_position + count
                
                if index == end_position:
                    new_end_position = end_position + count

                sub_tokens = self.tokenizer.tokenize(words)
                count = count + len(sub_tokens) - 1                 
                for sub_token in sub_tokens:
                    tokenized_words_list.append(sub_token)
                    tokenized_boxes_list.append(boxes[index])
                    tokenized_transcripts_location_list.append(transcripts_location[index])

        else:
            for index, words in enumerate(transcripts):
                sub_tokens = self.tokenizer.tokenize(words)               
                for sub_token in sub_tokens:
                    tokenized_words_list.append(sub_token)
                    tokenized_boxes_list.append(boxes[index])
                    tokenized_transcripts_location_list.append(transcripts_location[index])     
               



        # silde window !!! should be changed in future
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(tokenized_words_list):
            length = len(tokenized_words_list) - start_offset
            if length > max_tokens_of_words:
                length = max_tokens_of_words
            doc_spans.append(_DocSpan(start = start_offset, length = length))
            if start_offset + length >= len(tokenized_words_list):
                break
            start_offset = start_offset + min(length, self.window_stride)


        # generate training data, each time of loop will give out a feature object
        for index, docspan in enumerate(doc_spans):
            tokens = []
            boxes_tokens = []
            location_tokens = [] # -1 for inrelavent information
            token_type = []
            token_type_convert = []
            input_mask = []
            

            # add question part of sequence
            tokens.append('[CLS]')
            boxes_tokens.append(zero_padding_box)
            location_tokens.append(-1)
            token_type.append(0)
            token_type_convert.append(0)

            # add question text
            tokens.extend(question_tokens)
            boxes_tokens.extend([zero_padding_box] * len(question_tokens))
            location_tokens.extend([-1] * len(question_tokens))
            token_type.extend([0] * len(question_tokens))
            token_type_convert.extend([1] * len(question_tokens))
            tokens.append('[SEP]')
            boxes_tokens.append(sep_token_box)
            location_tokens.append(-1)
            token_type.append(0)
            token_type_convert.append(1)

            # add context
            for i in range(docspan.length):
                tmp_index = docspan.start + i
                tokens.append(tokenized_words_list[tmp_index])
                boxes_tokens.append(self.process_box(tokenized_boxes_list[tmp_index], width, height))
                location_tokens.append(tokenized_transcripts_location_list[tmp_index])
                token_type.append(1)
                token_type_convert.append(0)

            tokens.append('[SEP]')
            boxes_tokens.append(sep_token_box)
            location_tokens.append(-1)
            token_type.append(1)
            token_type_convert.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(tokens)

            # do padding
            while len(tokens) < self.max_sequence_length:
                tokens.append(0)
                boxes_tokens.append(zero_padding_box)
                location_tokens.append(-1)
                token_type.append(0)
                input_ids.append(0)
                input_mask.append(0)
                token_type_convert.append(1)

            # check length
            assert len(tokens) == self.max_sequence_length
            assert len(boxes_tokens) == self.max_sequence_length
            assert len(location_tokens) == self.max_sequence_length
            assert len(token_type) == self.max_sequence_length
            assert len(input_ids) == self.max_sequence_length
            assert len(token_type_convert) == self.max_sequence_length
            assert len(input_mask) == self.max_sequence_length


            assert new_start_position <= new_end_position 
            if self.training_flag:
                doc_start = docspan.start
                doc_end = docspan.start + docspan.length - 1
                out_of_span = False
                if not (new_start_position >= doc_start and new_end_position <= doc_end):
                    start_p = 0
                    end_p = 0
                else:
                    doc_offset = len(question_tokens) + 2
                    start_p = new_start_position - doc_start + doc_offset
                    end_p = new_end_position - doc_start + doc_offset
                    
                result.append(InputFeatures(input_ids, tokens, input_mask, token_type, token_type_convert, start_p, end_p, boxes_tokens, question_id, unique_id, transcripts, location_tokens))
                
            else:
                result.append(InputFeatures(input_ids, tokens, input_mask, token_type, token_type_convert, -1, -1, boxes_tokens, question_id, unique_id, transcripts, location_tokens))
                

            unique_id += 1
        return result


    # just get dataset
    def get_data_set(self):
        all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        all_token_type_convert = torch.tensor([f.token_type_convert for f in self.features], dtype=torch.long)
        all_boxes = torch.tensor([f.boxes for f in self.features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in self.features], dtype=torch.long) 
        all_end_positions = torch.tensor([f.end_position for f in self.features], dtype=torch.long)
        all_question_id = torch.tensor([f.question_id for f in self.features], dtype=torch.long)
        all_unique_id = torch.tensor([f.unique_id for f in self.features], dtype=torch.long)
        # all_transcripts = torch.tensor([f.transcripts for f in self.features], dtype=torch.long)
        all_locations = torch.tensor([f.location_of_words for f in self.features], dtype=torch.long)
        # when use this as batch, should follow the defeination of this order
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_token_type_convert, all_boxes, all_start_positions, all_end_positions, all_question_id, all_unique_id, all_locations)
        # index for easier usi         0            1                   2                    3               4              5                   6                   7             8                 9        

        return dataset




    def get_words_list(self):
        return self.words_dict


    # get data loader
    def get_data_loader(self):
        data_loader = DataLoader(self.get_data_set, batch_size=8)
        return data_loader

    def remove_punctuation(self, transcripts, indexes):

        new_transcripts, new_indexes = [], []
        for index, char in enumerate(transcripts):
            if char not in string.punctuation and char not in string.whitespace:
                new_transcripts.append(char)
                new_indexes.append(indexes[index])

        return new_transcripts, new_indexes



    # return tokenizer for envaluation
    def get_tokenizer(self):
        return self.tokenizer


    def process_box(self, box, w, h):
        x_all = box[::2]
        y_all = box[1::2]

        x_left = min(x_all)
        x_right = max(x_all)
        y_upper = min(y_all)
        y_lower = max(y_all)

        return [int(1000 * (x_left / w)), int(1000 * (y_upper / h)), int(1000 * (x_right / w)), int(1000 * (y_lower / h))]

        '''
        if box[0]<=box[4]:
            if box[1]<=box[5]:              
                return [int(1000 * (box[0] / w)), int(1000 * (box[1] / h)), int(1000 * (box[4] / w)), int(1000 * (box[5] / h))]
            else: 
                print(box)
        elif box[2]<=box[6]:
            if box[3]<=box[7]:
                return [int(1000 * (box[2] / w)), int(1000 * (box[3] / h)), int(1000 * (box[6] / w)), int(1000 * (box[7] / h))]
            else:
                print(box)
        elif box[6]<=box[2]:
            if box[7]<=box[3]:
                return [int(1000 * (box[6] / w)), int(1000 * (box[7] / h)), int(1000 * (box[2] / w)), int(1000 * (box[3] / h))]
            else:
                print(box)
        else:
            print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
            print(box)
        '''