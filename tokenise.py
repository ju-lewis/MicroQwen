
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="./Qwen2.5-0.5B/tokenizer.json")


string = input()
token_list = fast_tokenizer.tokenize(string)

with open("tokens.txt", "w") as fp:
    fp.write("\n".join(token_list))

print("Finished token generation.")
