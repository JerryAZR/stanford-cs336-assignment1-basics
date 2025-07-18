import logging
import os
import pprint
from collections import Counter
from typing import BinaryIO

import regex as re
from tqdm import tqdm

from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.chunk_trainer import ChunkTrainer

logger = logging.getLogger(__name__)


class BPETrainer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        train_file: str,
        vocab_size: int,
        special_tokens: list[bytes] = [b"<|endoftext|>"],
    ):
        self.train_file = train_file
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        # Dict that maps id to bytes
        self.vocab = [bytes([i]) for i in range(256)] + special_tokens
        # Map bytes to id
        self.vocab_rev = {v: k for k, v in enumerate(self.vocab)}
        # merge list
        self.merges: list[tuple[bytes, bytes]] = []

    def train(self, desired_num_chunks: int | None = None):
        with open(self.train_file, "rb") as f:
            # 1. Read and split text using config
            chunk_boundaries = find_chunk_boundaries(
                f, desired_num_chunks or os.cpu_count() or 8, self.special_tokens
            )
            logger.debug(chunk_boundaries)
            logger.debug(len(chunk_boundaries))

            trainers = [
                self.create_trainer(chunk_boundaries[i], chunk_boundaries[i + 1])
                for i in range(len(chunk_boundaries) - 1)
            ]

            # Determine how many merges we need
            initial_vocab_size = len(self.vocab)
            num_merges_needed = self.vocab_size - initial_vocab_size

            for _ in tqdm(range(num_merges_needed), desc="Training BPE"):
                counts: Counter[tuple[bytes, bytes]] = Counter()
                for trainer in trainers:
                    counts += trainer.count_pairs()
                # 2. Count tokens, apply BPE merges
                logger.debug(counts.most_common(10))
                max_pair = max(
                    counts.items(),
                    key=lambda item: (
                        item[1],
                        item[0],
                    ),  # frequency first, then lex order
                )[0]
                logger.info(
                    f"Merging {len(self.merges)}: {max_pair}; Counts {counts.most_common(3)}"
                )
                for trainer in trainers:
                    trainer.merge_pairs(max_pair)

                # 3. Build vocab
                new_vocab = max_pair[0] + max_pair[1]
                self.vocab_rev[new_vocab] = len(self.vocab)
                self.vocab.append(new_vocab)
                self.merges.append(max_pair)

            logger.debug("vocab:\n" + pprint.pformat(self.vocab))
            logger.debug("merges:\n" + pprint.pformat(self.merges))

    def split_on_special(self, text: bytes) -> list[bytes]:
        # Escape tokens to safely use in regex (note: all tokens must be bytes)
        escaped_tokens = [
            re.escape(tok) for tok in self.special_tokens
        ]  # tokens are already bytes
        pattern = b"|".join(escaped_tokens)
        # Split and filter
        return [s for s in re.split(pattern, text) if s.strip() != b""]

    def create_trainer(self, start: int, end: int) -> ChunkTrainer:
        with open(self.train_file, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start)  # still bytes
            paragraphs = self.split_on_special(chunk)  # returns list[bytes]
            logger.debug(f"<chunk>{chunk}</chunk>")

            pre_tokens: Counter[bytes] = Counter()
            for paragraph in paragraphs:
                pre_tokens += self.pre_tokenize(
                    paragraph.decode("utf-8")
                )  # must also accept bytes
            return ChunkTrainer(pre_tokens)

    def pre_tokenize(self, text: str) -> Counter[bytes]:
        vocab = [match.group().encode("utf-8") for match in re.finditer(self.PAT, text)]
        return Counter(vocab)


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, special_tokens: list[bytes]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            pattern = re.compile(b"|".join(re.escape(tok) for tok in special_tokens))
            match = pattern.search(mini_chunk)
            if match:
                chunk_boundaries[bi] = initial_position + match.start()
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    token_bytes = [token_str.encode("utf-8") for token_str in special_tokens]
    trainer = BPETrainer(input_path, vocab_size, token_bytes)
    trainer.train()
    return ({k: v for k, v in enumerate(trainer.vocab)}, trainer.merges)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("train_file", help="Path to the training text file.")
    parser.add_argument("vocab_size", type=int, help="Target vocabulary size.")
    parser.add_argument(
        "--log",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log}")
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Train
    trainer = BPETrainer(args.train_file, args.vocab_size)
    trainer.train(8)
