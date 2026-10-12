"""개념 검색용 임베딩 (계약 createEmbeddings). 저장·cosine·threshold 는 BE 책임이고, 여기서는 벡터만 만든다.

백엔드
  HashEmbedder                 글자 n-gram 해싱. 모델 파일이 필요 없는 개발·테스트용. 다른 언어 표기는 잇지 못한다
  SentenceTransformerEmbedder  고정 revision 의 다국어 문장 임베딩 모델 (예: intfloat/multilingual-e5-small)

결정성: 같은 모델 revision·입력이면 같은 벡터. 소수 여섯째 자리로 반올림해 플랫폼별 미세 오차를 줄인다.
"""

import hashlib
import math
import unicodedata
from dataclasses import dataclass

ROUND = 6


class EmbeddingError(Exception):
    """kind: too_large | invalid_output"""

    def __init__(self, kind: str, index: int | None = None):
        self.kind, self.index = kind, index
        super().__init__(kind)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0 or not math.isfinite(norm):
        raise EmbeddingError("invalid_output")
    return [round(v / norm, ROUND) for v in vec]


@dataclass
class HashEmbedder:
    dims: int = 256
    provider: str = "local"
    model: str = "char-ngram-hash"
    max_chars: int = 2000
    preprocessing_version: str = "nfc-lower-ngram1-3-v1"

    @property
    def revision(self) -> str:
        return f"hash-sha256-d{self.dims}-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i, t in enumerate(texts):
            if len(t) > self.max_chars:
                raise EmbeddingError("too_large", i)
            s = unicodedata.normalize("NFC", t).lower()
            vec = [0.0] * self.dims
            for n in (1, 2, 3):
                for k in range(len(s) - n + 1):
                    h = hashlib.sha256(s[k:k + n].encode()).digest()   # Python hash() 는 실행마다 달라 쓰지 않는다
                    idx = int.from_bytes(h[:4], "big") % self.dims
                    vec[idx] += 1.0 if h[4] & 1 else -1.0
            if not any(vec):
                vec[0] = 1.0
            out.append(_normalize(vec))
        return out


class SentenceTransformerEmbedder:
    """sentence-transformers 백엔드. `uv sync --extra embeddings` 로 설치하고, 처음 쓸 때 모델 파일을 내려받는다."""

    provider = "local"

    def __init__(self, model: str, revision: str, prefix: str = "", device: str = "cpu"):
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        path = snapshot_download(model, revision=revision)
        self.model = model
        self.revision = path.rstrip("/").rsplit("/", 1)[-1]     # 실제로 받은 commit hash
        self.prefix = prefix
        self._st = SentenceTransformer(path, device=device)
        self.dims = int(self._st.get_sentence_embedding_dimension())
        self.max_tokens = int(self._st.max_seq_length)
        self.preprocessing_version = f"prefix:{prefix.strip() or 'none'}-normalized-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        inputs = [self.prefix + t for t in texts]
        for i, t in enumerate(inputs):
            # 토큰 한도를 넘으면 모델이 몰래 잘라 의미가 바뀐다. 잘라서 성공하지 않고 413 으로 거절한다.
            if len(self._st.tokenizer(t)["input_ids"]) > self.max_tokens:
                raise EmbeddingError("too_large", i)
        vectors = self._st.encode(inputs, normalize_embeddings=False, convert_to_numpy=True, batch_size=32)
        return [_normalize([float(v) for v in vec]) for vec in vectors]
