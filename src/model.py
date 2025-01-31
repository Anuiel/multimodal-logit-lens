import typing as tp
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


class VLM(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name)
        self.model.eval()

    def inference(self, image: Image.Image, conversation: str, max_new_tokens: int = 32) -> dict[str, tp.Any]:
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, output_hidden_states=True, max_new_tokens=max_new_tokens, return_dict_in_generate=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids['sequences'])]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        hidden_states = output_ids['hidden_states'][0]
        hidden_states_features = self._process_hidden_features(hidden_states)
        return {
            "output_text": output_text[0],
            "hidden_states_features": hidden_states_features
        }

    def _process_hidden_features(self, hidden_states: list[torch.Tensor]) -> list[torch.Tensor]:
        hidden_states_features: dict[str, tp.Any] = {}
        for i, hidden_state in enumerate(hidden_states):
            with torch.no_grad():
                hidden_state = hidden_state[0, -1, :]
                if i != 28:
                    hidden_state = self.model.model.norm(hidden_state)
                logits = self.model.lm_head(hidden_state).to(dtype=torch.float).detach().cpu()
            _, topk_indices = logits.topk(k=5)
            topk_tokens = [self.processor.decode([x]) for x in topk_indices]
            topk_probs = [round(float(x), 3) for x in torch.softmax(logits, -1)[topk_indices.to(dtype=int)].cpu()]
            hidden_states_features[f"layer_{i}"] = {
                "topk_tokens": topk_tokens,
                "topk_probs": topk_probs
            }
        return hidden_states_features

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
