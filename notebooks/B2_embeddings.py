import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

df = pd.read_csv("../data/corpus.csv")

model_name = "dicta-il/dictabert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

embeddings = []

for i, row in df.iterrows():
    text = str(row["text_clean"])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    embedding = summed / counts

    embeddings.append(embedding.cpu().numpy()[0])

    print(f"Processed {i + 1}/{len(df)}: {row['title']}")

embeddings = np.array(embeddings)

Path("../data").mkdir(exist_ok=True)
np.save("../data/embeddings.npy", embeddings)

print("Saved embeddings to ../data/embeddings.npy")
print("Embeddings shape:", embeddings.shape)