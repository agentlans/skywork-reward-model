import gc
from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SkyworkRewardModel:
    """A wrapper class for evaluating model responses using Skywork Reward Models.

    This class handles the initialization, tokenization, formatting, and scoring
    of prompt-response pairs using a sequence classification model. It also provides
    robust memory management utilities to explicitly free VRAM/RAM when processing
    is complete.

    Attributes:
        model (AutoModelForSequenceClassification): The underlying transformer 
            sequence classification model.
        tokenizer (AutoTokenizer): The tokenizer paired with the reward model.
        device (torch.device): The primary device where the model is allocated.

    Example:
        >>> with SkyworkRewardModel("Skywork/Skywork-Reward-Llama-3.1-8B") as rm:
        ...     scores = rm.evaluate("What is 2+2?", ["It is 4.", "It is 5."])
        >>> print(scores)
    """

    def __init__(self, model_dir: str):
        """Initializes the SkyworkRewardModel with a local or Hugging Face hub path.

        Args:
            model_dir (str): The path to the model directory or the Hugging Face 
                repository identifier.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            device_map="auto",
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device = self.model.device

    def _format_for_tokenizer(self, prompt: str, response: str) -> dict:
        """Formats a single prompt-response pair using the model's chat template.

        Args:
            prompt (str): The user query or prompt.
            response (str): The assistant response to evaluate.

        Returns:
            dict: A dictionary of tokenized inputs (tensors) mapped to the 
                target device.
        """
        conv = [
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": response}
        ]
        
        # Apply the chat template without tokenizing immediately to handle token stripping
        conv_formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
        
        # Clean up the BOS token if present to prevent double-tokenization issues
        if self.tokenizer.bos_token and conv_formatted.startswith(self.tokenizer.bos_token):
            conv_formatted = conv_formatted[len(self.tokenizer.bos_token):]
            
        return self.tokenizer(conv_formatted, return_tensors="pt").to(self.device)

    def evaluate(self, prompt: str, responses: List[str]) -> List[float]:
        """Evaluates multiple responses against a single prompt and returns scalar scores.

        Args:
            prompt (str): The user query or prompt.
            responses (List[str]): A list of candidate responses to score.

        Returns:
            List[float]: A list of scalar reward scores corresponding to each response.
        """
        tokenized_inputs = [self._format_for_tokenizer(prompt, res) for res in responses]
        scores = []
        
        with torch.no_grad():
            for x in tokenized_inputs:
                outputs = self.model(**x)
                # Extract the scalar score from the first batch item's logit
                score = outputs.logits.flatten()[0].item()
                scores.append(score)
                
        return scores

    def close(self) -> None:
        if hasattr(self, 'model') and self.model is not None:
            # Check if torch still exists before moving to CPU
            if 'torch' in globals() and torch is not None:
                try:
                    self.model.cpu()
                except Exception:
                    pass
            del self.model
            self.model = None
    
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
    
        # Safely call gc if it hasn't been torn down
        if 'gc' in globals() and gc is not None:
            gc.collect()
    
        # Safely check torch.cuda
        if 'torch' in globals() and torch is not None:
            if hasattr(torch, 'cuda') and torch.cuda is not None:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def __enter__(self):
        """Enables context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Enables context manager exit, ensuring resources are freed automatically."""
        self.close()
        
    def __del__(self):
        """Fallback finalizer to catch cases where the user forgets to call close() 
        or use a context manager.
        """
        self.close()
