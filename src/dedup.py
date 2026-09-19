"""MinHash предлагает кандидатов, точный Жаккар подтверждает совпадения.

Обратный индекс шинглов дополняет вероятностный поиск: для небольшого
учебного набора нельзя пропускать пары только из-за случайности LSH.
Это точная проверка лексического сходства, а не всех смысловых парафразов.
"""
from collections import defaultdict
from typing import Sequence
from datasketch import MinHash, MinHashLSH
from src.textnorm import normalize_text, shingles


def build_minhash(text, shingle_words, num_perm):
    mh = MinHash(num_perm=num_perm, seed=42)
    mh.update_batch([s.encode('utf-8') for s in sorted(shingles(normalize_text(text), shingle_words))])
    return mh


def exact_duplicates(keys: Sequence[str]) -> list[int]:
    seen, dupes = set(), []
    for i, key in enumerate(keys):
        if key in seen:
            dupes.append(i)
        else:
            seen.add(key)
    return dupes


class SimilarityIndex:
    def __init__(self, shingle_words, num_perm, threshold):
        if not 0 < threshold < 1 or shingle_words < 1:
            raise ValueError('Неверные параметры сходства')
        self.size, self.perm, self.threshold = shingle_words, num_perm, threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.sets, self.postings = {}, defaultdict(set)

    def query(self, text):
        ss = shingles(normalize_text(text), self.size)
        mh = build_minhash(text, self.size, self.perm)
        candidates = set(self.lsh.query(mh))
        # При положительном Жаккаре обязательно есть общий шингл.
        for token in ss:
            candidates.update(self.postings[token])
        matches = []
        for key in sorted(candidates, key=int):
            other = self.sets[key]
            if min(len(ss), len(other)) < self.threshold * max(len(ss), len(other)):
                continue
            union = ss | other
            if union and len(ss & other) / len(union) >= self.threshold:
                matches.append(int(key))
        return matches

    def insert(self, key, text):
        key = str(key)
        ss = shingles(normalize_text(text), self.size)
        self.sets[key] = ss
        self.lsh.insert(key, build_minhash(text, self.size, self.perm))
        for token in ss:
            self.postings[token].add(key)


def near_duplicates(texts, shingle_words, num_perm, threshold):
    index = SimilarityIndex(shingle_words, num_perm, threshold)
    dupes = []
    for i, text in enumerate(texts):
        if index.query(text):
            dupes.append(i)
        else:
            index.insert(i, text)
    return dupes


def cross_near_duplicates(left, right, shingle_words, num_perm, threshold):
    index = SimilarityIndex(shingle_words, num_perm, threshold)
    for i, text in enumerate(left):
        index.insert(i, text)
    return [(i, j) for j, text in enumerate(right) for i in index.query(text)]
