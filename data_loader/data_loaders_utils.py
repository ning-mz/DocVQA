import cv2



class Question(object):
    def __init__(self, original_question: dict, training_flag: bool = False):
        self.questionId = original_question['questionId']
        self.question_text = original_question['question']
        if training_flag:
            self.answer_text = original_question['answers'][0] # now just used the first result as correct result
        self.training_flag = training_flag
        self.image_file = original_question['image']


class Document(object):
    def __init__(self,
                docId: int = None,
                image = None,
                text: list = None,
                width: int = None,
                height: int = None,
                question_list: list = None
                ):
        self.docId = docId
        self.image = image
        self.text = text
        self.width = width
        self.height = height
        self.question_list = question_list


    
    def add_question(self, new_question: Question):
        # self.question_list.append(Question(new_quesiton, training_flag))
        self.question_list.append(new_question)





# Copy from LayoutLM, but do some changes
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        tokens,
        input_mask,
        segment_ids,
        token_type_convert,
        start_position,
        end_position,
        boxes,
        question_id,
        unique_id,
        transcripts,
        location_of_words
    ):
        # assert (
        #     0 <= all(boxes) <= 1000
        # ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
        #     boxes
        # )
        self.input_ids = input_ids
        self.tokens = tokens
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.token_type_convert = token_type_convert
        self.start_position = start_position
        self.end_position = end_position
        self.boxes = boxes
        self.question_id = question_id
        self.unique_id = unique_id
        self.transcripts = transcripts
        self.location_of_words = location_of_words
    

    