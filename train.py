from huggingface_hub import login
from engine import engine

if __name__ == '__main__':
    login(
      token="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", # ADD YOUR TOKEN HERE
      add_to_git_credential=True
    )

    # Start training
    trainer = engine()
    trainer.trainer.train()

    # Save processor and create model card
    trainer.tokenizer.save_pretrained(trainer.repo_id)
    trainer.create_model_card()
    trainer.push_to_hub()