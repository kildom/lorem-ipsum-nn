
import json
import torch
from model import LatinSharedNet, LETTERS_PER_CONTEXT, ALPHABET_LENGTH, LETTER_TO_INDEX, INDEX_TO_LETTER, DEFAULT_TEMP, LETTER_EMBEDDING_SIZE, LETTER_EMBEDDING_INTER_SIZE, GROUP_EMBEDDING_INTER_SIZE, LETTERS_PER_GROUP



model = LatinSharedNet()
model.load_state_dict(torch.load("latin_model.pt"))
model.eval()

def get_max_letter_embedding():
    max_values = [0] * LETTER_EMBEDDING_SIZE
    for i in range(ALPHABET_LENGTH):
        emb = model.get_letter_embedding(i)
        emb = emb.detach().cpu().numpy().tolist()
        for j in range(LETTER_EMBEDDING_SIZE):
            max_values[j] = max(max_values[j], emb[j])
    return max_values

with torch.no_grad():

    # === Adjust weight and bias of linear layers, so letter embeddings are in range [0, 254]

    # Calculate needed adjustment
    max_values = get_max_letter_embedding()
    print("Max values:  ", max_values)
    letter_feature_adjustments = [(1 / x * 254 if x > 0 else 1) for x in max_values]
    print("Adjustments: ", letter_feature_adjustments)
    # Adjust the weight and bias of the letter embedding output layer
    for i in range(LETTER_EMBEDDING_SIZE):
        model.letter_embedding[2].bias[i] *= letter_feature_adjustments[i]
        for j in range(LETTER_EMBEDDING_INTER_SIZE):
            model.letter_embedding[2].weight[i][j] *= letter_feature_adjustments[i]
    # Adjust the weight and bias of the group embedding input layer
    for i in range(GROUP_EMBEDDING_INTER_SIZE):
        for j in range(LETTERS_PER_GROUP * LETTER_EMBEDDING_SIZE):
            letter_feature_index = j % LETTER_EMBEDDING_SIZE
            model.group_embedding[0].weight[i][j] /= letter_feature_adjustments[letter_feature_index]
    # Check the adjustments
    max_values = get_max_letter_embedding()
    print("Adjusted max values:  ", max_values)
    assert min(max_values) > 253.5
    assert max(max_values) < 254.5
    # Print final embeddings for each letter
    letter_to_embedding = [[]] * ALPHABET_LENGTH
    for i in range(ALPHABET_LENGTH):
        emb = model.get_letter_embedding(i)
        emb = emb.detach().cpu().numpy().tolist()
        print(f"{i:2d} {INDEX_TO_LETTER[i]}: {[round(x) for x in emb]}")
        letter_to_embedding[i] = [round(x) for x in emb]
    with open("data/letter-embeddings.json", 'w') as file:
        json.dump(letter_to_embedding, file, indent=2)
