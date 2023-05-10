import pickle
import tensorflow_text as text
import tensorflow as tf
import numpy as np
import sys
from absl import flags
sys.argv=['preserve_unused_tokens=False']
flags.FLAGS(sys.argv)

class BERTclassifier:
    MAX_LEN = 100
    
    def __init__(self, base_dir): #these values are set as default values. These can be changed
        self.base_dir = base_dir
        self.models =[]
        self.scores ={}
        self.tokenizer = None
        self.load_models()
        self.load_tokenizer()

    def load_models(self):
        for i, model in enumerate([
            f"{self.base_dir}/bert_model/saved_model/bert", 
            f"{self.base_dir}/bert_model/saved_model/bert"
        ]):
            model = tf.keras.models.load_model(f"{model}_{i}")
            self.models.append(model)
    
    def load_tokenizer(self):
        with open("tokenizer", "rb") as f:
            self.tokenizer = pickle.load(f)

    def encode(self, texts):
    
        input_ids =[] #tokens converted into numerical form
        attention_masks =[] #1- acual word , 0- padding
        token_type_ids =[] # segment id 
        
        for text in texts: 
            
            text = self.tokenizer.tokenize(text)
            text = text[:BERTclassifier.MAX_LEN - 2]
            input_sequence = ['[CLS]'] + text + ['[SEP]']
            pad_len = BERTclassifier.MAX_LEN - len(input_sequence)
            
            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * BERTclassifier.MAX_LEN
            

            input_ids.append(tokens)
            attention_masks.append(pad_masks)
            token_type_ids.append(segment_ids)
        
        return np.array(input_ids), np.array(attention_masks), np.array(token_type_ids)
    
    
    def predict(self, text):

        input_ids_test, attention_mask_test, token_type_ids_test = self.encode([text])
        
        y_pred = np.zeros((input_ids_test.shape[0], 1))

        for model in self.models:
            y_pred += model.predict([input_ids_test, attention_mask_test, token_type_ids_test]) / len(self.models)

        return y_pred
    

# if __name__=="__main__":
#     classify = BERTclassifier()
#     print(classify.predict("this is an earthquake"))