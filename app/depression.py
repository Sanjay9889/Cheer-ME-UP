from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class Depression:

    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        scores = torch.softmax(logits, dim=1).detach().numpy()[0]
        return scores


if __name__ == '__main__':
    text = "I feel huge pain my girlfriend left me"
    depression = Depression()
    scores = depression.predict(text)
    print(f"Depression score: {scores[0]}, Suicidal score: {scores[1]}")
