from collections import Counter
import logging

logger = logging.getLogger(__name__)

class ChunkTrainer:
    def __init__(self, pre_tokens: Counter[bytes]):
        self.tokens: Counter[tuple[bytes, ...]] = Counter()
        self.last_merge: tuple[bytes, bytes, bytes] | None = None

        for word, freq in pre_tokens.items():
            token_seq = tuple(bytes([b]) for b in word)  # split into 1-byte tokens
            self.tokens[token_seq] += freq

    def count_pairs(self) -> Counter[tuple[bytes, bytes]]:
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()

        # For now, full count — last_merge will be used later for optimization.
        for seq, freq in self.tokens.items():  # Counter[tuple[bytes, ...]]
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += freq

        return pair_counts

    def merge_pairs(self, pair: tuple[bytes, bytes]):
        replacements: list[tuple[tuple[bytes, ...], tuple[bytes, ...]]] = []

        for token, count in self.tokens.items():
            new_token, replaced = replace_subtuple(
                token, pair, (pair[0] + pair[1],)
            )
            if replaced > 0:
                replacements.append((token, new_token))

        for old_token, new_token in replacements:
            logger.debug(f"Replacing {old_token} -> {new_token}")
            count = self.tokens.pop(old_token)
            self.tokens[new_token] += count


def replace_subtuple(
    original: tuple[bytes, ...],
    pattern: tuple[bytes, ...],
    replacement: tuple[bytes, ...],
) -> tuple[tuple[bytes, ...], int]:

    if not pattern:
        raise ValueError("pattern must be non-empty")

    result: list[bytes] = []
    i = 0
    count = 0
    plen = len(pattern)
    while i <= len(original) - plen:
        if original[i : i + plen] == pattern:
            result.extend(replacement)
            count += 1
            i += plen
        else:
            result.append(original[i])
            i += 1
    result.extend(original[i:])  # append any remaining tail
    return tuple(result), count
