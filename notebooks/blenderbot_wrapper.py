import torch
import mlflow.pyfunc

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import BlenderbotTokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotConfig
from transformers import BlenderbotTokenizerFast 

class BlenderbotWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to use Blenderbot Model
    """

    def load_context(self, context = None):
        """
          This method is called when loading an MLflow model with pyfunc.load_model(),
          as soon as the Python Model is constructed.
          Args:
              context: MLflow context where the model artifact is stored.
        """

        self.tokenizer = BlenderbotTokenizerFast.from_pretrained(context.artifacts["hf_tokenizer_path"])
        self.model = BlenderbotForConditionalGeneration.from_pretrained(context.artifacts["hf_model_path"])

    def predict(self, model_input, context = None):
        """This is an abstract function. We customized it into a method to fetch the Hugging Face model.
        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to score the model.
        Returns:
            (response, history): Tuple containing the response plus the conversation history
        """

        question = model_input["question"]
        history = model_input["history"]
        history.append(question)
    
        context = ' '.join([str(elem)for elem in history[len(history)-3:]])
        input_ids = self.tokenizer(
          [(context)],
          return_tensors="pt",
          max_length=512,
          truncation=True
        )
        next_reply_ids = self.model.generate(
          **input_ids,
          max_length = 512,
          pad_token_id = self.tokenizer.eos_token_id
        )
        response = self.tokenizer.batch_decode(
          next_reply_ids,
          skip_special_tokens = True
        )[0]
        history.append(response)
        # convert to tuples of list
        response = [(history[i], history[i+1]) for i in range(0, len(history)-1, 2)]
        response_payload = {
          "answer": response[-1],
          "history": history
        }
        return response_payload


def _load_pyfunc(data_path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return BlenderbotWrapper(data_path)