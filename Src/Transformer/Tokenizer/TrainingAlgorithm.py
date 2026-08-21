from tqdm import tqdm


class Tokenizer():
    def __init__(self, text):
        self.text = text

    def word_counter(self, text):
        word_dict = {}
        for word in text:
            tupled_word = tuple(word)
            if tupled_word not in word_dict.keys():
                word_dict[tupled_word] = 0
            word_dict[tupled_word] += 1
        sorted_word_dict = sorted(word_dict.items(), key=lambda item: item[1], reverse = True)
        return sorted_word_dict

    def pair_counter(self, word_text):
        pair_dict = {}
        for word in word_text:
            i = 0
            while i < len(word[0]) - 1:
                if (word[0][i], word[0][i + 1]) not in pair_dict:
                    pair_dict[word[0][i], word[0][i + 1]] = 0
                pair_dict[word[0][i], word[0][i + 1]] += 1 * word[1]
                i += 1
        sorted_pair_dict = sorted(pair_dict.items(), key=lambda item: item[1], reverse=True)
        return sorted_pair_dict

    def tokenize(self, target_pair, text, token):
        index = 0
        token_list = text
        for word in token_list:
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == target_pair:
                    word[i: i + 2] = [token]
                else:
                    i += 1
        token += 1
        index += 1
        return token_list

    def train(self, merges, start_token):
        current_token = start_token
        merge_dict = {}
        for i in tqdm(range(merges), desc= "Training..."):
            word_dict = self.word_counter(self.text)
            pair_dict = self.pair_counter(word_dict)

            if not pair_dict:
                break

            top_pair = pair_dict[0][0]

            merge_dict[top_pair] = current_token

            frequency = pair_dict[0][1]
            if frequency < 5:
                break

            self.text = self.tokenize(top_pair, self.text, current_token)
            current_token += 1

        return self.text, merge_dict

