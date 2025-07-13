
import random
import json
import re
from data_loader import load_text, random_probs


def prob_normalize(probabilities, total=65536):
    sum_prob = sum(probabilities)
    probabilities = [x / sum_prob * total for x in probabilities]
    int_prob = [int(round(x)) for x in probabilities]
    while sum(int_prob) > total:
        diff = [probabilities[i] - int_prob[i] for i in range(len(probabilities))]
        min_diff = min(diff)
        index = diff.index(min_diff)
        assert int_prob[index] > 0
        int_prob[index] -= 1
    while sum(int_prob) < total:
        diff = [probabilities[i] - int_prob[i] for i in range(len(probabilities))]
        max_diff = max(diff)
        index = diff.index(max_diff)
        int_prob[index] += 1
    cumulative_sum = [sum(int_prob[:i + 1]) for i in range(len(int_prob))]
    return cumulative_sum


text = load_text()
text = re.sub(r'\s*\w+\s*', 'a', text, 0)

index = 0

MAX_WORDS_IN_SENTENCE = 48
MAX_WORDS_TO_COMMA = 24

prob_dot = [0] * (MAX_WORDS_IN_SENTENCE + 1)
max_words = 0

while index < len(text):
    next_dot = text.find('.', index)
    if next_dot <= 0:
        break
    words = text[index:next_dot].count('a')
    if words > MAX_WORDS_IN_SENTENCE:
        index = next_dot + 1
        continue
    max_words = max(max_words, words)
    prob_dot[words] += 1
    index = next_dot + 1

prob_dot[0] = 0
prob_dot[1] = 0

# print(prob_dot)
# print(prob_normalize(prob_dot))

index = 0

prob_comma = [[0] * (MAX_WORDS_TO_COMMA + 1) for _ in range(MAX_WORDS_IN_SENTENCE + 1)]

while index < len(text):
    next_dot = text.find('.', index)
    next_comma = text.find(',', index)
    if next_dot <= 0 or next_comma <= 0:
        break
    words_to_the_end_of_sentence = text[index:next_dot].count('a')
    words_to_comma = text[index:next_comma].count('a')
    if words_to_the_end_of_sentence > MAX_WORDS_IN_SENTENCE or words_to_comma > MAX_WORDS_TO_COMMA:
        index = next_comma + 1
        continue
    if words_to_comma > words_to_the_end_of_sentence:
        words_to_comma = words_to_the_end_of_sentence
    prob_comma[words_to_the_end_of_sentence][words_to_comma] += 1
    index = next_comma + 1
    #print(f"{index}/{len(text)}", end='\r')

prob_comma[0][0] = 1

prob_comma[1][0] = 0
prob_comma[1][1] = 1

prob_comma[2][0] = 0
prob_comma[2][1] = 0
prob_comma[2][2] = 1

probs_data = {
    "dot": prob_normalize(prob_dot),
    "comma": [prob_normalize(prob_comma[i][:i + 1]) for i in range(MAX_WORDS_IN_SENTENCE + 1)]
}

# for i in range(3, MAX_WORDS_IN_SENTENCE + 1):
#     print(i, prob_normalize(prob_comma[i][:i + 1]))

with open('data/punctuation.json', 'w') as f:
    json.dump(probs_data, f)

def test():
    print("Testing random_probs function")
    probs = prob_normalize([0, 1, 2, 1, 4, 3, 0, 1], 12)
    print(random_probs(probs, 0))
    print(random_probs(probs, 1))
    print(random_probs(probs, 2))
    print(random_probs(probs, 3))
    print(random_probs(probs, 4))
    print(random_probs(probs, 5))
    print(random_probs(probs, 6))
    print(random_probs(probs, 7))
    print(random_probs(probs, 8))
    print(random_probs(probs, 9))
    print(random_probs(probs, 10))
    print(random_probs(probs, 11))


if __name__ == "__main__":
    test()
