from transformers import pipeline
from engine import engine

if __name__ == '__main__':
    trainer = engine()
    # load model from huggingface.co/models using our repository id
    classifier = pipeline("sentiment-analysis", model=trainer.repo_id, tokenizer=trainer.repo_id, device=0)

    sample = "I have been waiting longer than expected for my bank card, could you provide information on when it will arrive?"

    pred = classifier(sample)
    print(pred)
    # [{'label': 'card_arrival', 'score': 0.9903606176376343}]