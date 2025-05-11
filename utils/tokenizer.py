import os
import json
import re
import spacy
from collections import defaultdict, Counter
import string

class BPETokenizer:
    def __init__(self, path=".datas/merges_bpe.json", num_merges=1000, dataset=None, use_stopwords=True):
        self.path = path
        self.num_merges = num_merges
        self.merges = None
        self.nlp = spacy.load("fr_core_news_sm")
        self.use_stopwords = use_stopwords
        
        # Define French stopwords (common words that don't carry much meaning)
        self.stopwords = set(['le', 'la', 'les', 'un', 'une', 'des', 'et', 'est', 'il', 'elle', 
                             'ils', 'elles', 'ce', 'cette', 'ces', 'se', 'sa', 'son', 'ses',
                             'on', 'nous', 'vous', 'qui', 'que', 'quoi', 'dont', 'où', 'je',
                             'tu', 'de', 'a', 'pour', 'en', 'dans', 'par', 'sur', 'au', 'aux'])

        if dataset is not None:
            self.learn_merges(dataset)
        elif os.path.exists(self.path):
            self._load_merges()

    def preprocess_sentence(self, sentence):
        # Convert to lowercase
        text = sentence.lower()
        
        # Remove punctuation
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract lemmas, filter alpha and optionally filter stopwords
        if self.use_stopwords:
            tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in self.stopwords]
        else:
            tokens = [token.lemma_ for token in doc if token.is_alpha]
        
        # Join and normalize whitespace
        text = " ".join(tokens)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _get_vocab(self, corpus):
        vocab = defaultdict(int)
        for sentence in corpus:
            words = sentence.lower().split()
            for word in words:
                chars = list(word) + ['</w>']
                vocab[tuple(chars)] += 1
        return vocab

    def _get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word)-1):
                pairs[(word[i], word[i+1])] += freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = pair
        replacement = ''.join(bigram)
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and word[i] == bigram[0] and word[i+1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def learn_merges(self, dataset):
        print(f"Learning BPE merges (num_merges={self.num_merges})...")
        print("Preprocessing dataset...")
        preprocessed = [self.preprocess_sentence(s) for s in dataset]
        
        print("Building vocabulary...")
        vocab = self._get_vocab(preprocessed)
        merges = []

        print("Performing BPE merges...")
        for i in range(self.num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            merges.append(best)
            
            # Print progress every 100 merges
            if (i+1) % 100 == 0:
                print(f"  Progress: {i+1}/{self.num_merges} merges")

        self.merges = merges
        self._save_merges()
        print(f"{len(merges)} merges learned and saved.")
        
        # Also print vocabulary statistics
        word_pieces = [token for word in vocab.keys() for token in word]
        most_common = Counter(word_pieces).most_common(10)
        print(f"Most common tokens: {most_common}")

    def _apply_bpe_to_word(self, word):
        word = list(word) + ["</w>"]
        for merge in self.merges:
            i = 0
            while i < len(word) - 1:
                if word[i] == merge[0] and word[i + 1] == merge[1]:
                    word[i:i + 2] = ["".join(merge)]
                else:
                    i += 1
        return word

    def _load_merges(self):
        with open(self.path, "r", encoding="utf-8") as f:
            merges = json.load(f)
        self.merges = [tuple(pair) for pair in merges]
        print(f"Loaded {len(self.merges)} BPE merges from {self.path}")

    def _save_merges(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.merges, f, ensure_ascii=False, indent=2)

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        text = self.preprocess_sentence(text)
        tokens = []
        for word in text.split():
            tokens.extend(self._apply_bpe_to_word(word))
        return tokens

    def __call__(self, text):
        return self.tokenize(text) 
