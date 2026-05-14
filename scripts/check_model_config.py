from transformers import AutoConfig, AutoTokenizer

MODEL_NAME = "smilegate-ai/kor_unsmile"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    print("model_name:", MODEL_NAME)
    print("tokenizer.model_max_length:", tokenizer.model_max_length)
    print("config.max_position_embeddings:", getattr(config, "max_position_embeddings", None))


if __name__ == "__main__":
    main()


'''
model_name: smilegate-ai/kor_unsmile
tokenizer.model_max_length: 300
config.max_position_embeddings: 300
'''