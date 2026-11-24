from deepeval.synthesizer.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig,FiltrationConfig,ContextConstructionConfig
from deepeval.synthesizer.types import Evolution

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from deepeval.models import OpenRouterModel
llm = OpenRouterModel(
    model = "openai/gpt-4o-mini",
    temperature=0.1
)

from typing import List
from deepeval.models import DeepEvalBaseEmbeddingModel
from langchain_huggingface import HuggingFaceEmbeddings


class LocalHuggingFaceEmbedder(DeepEvalBaseEmbeddingModel):

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def load_model(self):
        return self.embeddings

    def embed_text(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        return self.embed_text(text)

    async def a_embed_texts(
        self, texts: List[str]
    ) -> List[List[float]]:
        return self.embed_texts(texts)

    def get_model_name(self):
        return "sentence-transformers/all-MiniLM-L6-v2"


#=========== Define Path ==================#
ROOT_PATH = Path(__file__).resolve().parents[2]
GOLDENS_DIR = ROOT_PATH/"data"/"evaluation"/"goldens"
DOCS_PATH = ROOT_PATH/"data"/"processed"
GOLDENS_DIR.mkdir(exist_ok=True, parents=True)

def get_docs_path(directory_path: Path | str) -> list[str]:
    directory_path = Path(directory_path)
    if not directory_path.exists() or not directory_path.is_dir():
        return []
    return [path.as_posix() for path in directory_path.glob("*.txt")]
        


filteration_config = FiltrationConfig(
    synthetic_input_quality_threshold=0.6,
    critic_model=llm
)
evaluation_config =EvolutionConfig(
    evolutions={
        Evolution.MULTICONTEXT:0.1,
        Evolution.CONCRETIZING:0.3,
        Evolution.CONSTRAINED:0.4,
        Evolution.COMPARATIVE:0.2
    }
)

local_embedder = LocalHuggingFaceEmbedder()

context_config = ContextConstructionConfig(
    critic_model=llm,
    embedder=local_embedder,
    max_contexts_per_document=2,
    context_quality_threshold=0.7,
    max_retries=1
)

#========== synthesizer ===========#
synthesizer = Synthesizer(
    model=llm,
    async_mode=True,
    max_concurrent=1,
    filtration_config=filteration_config,
    evolution_config=evaluation_config
)

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=get_docs_path(DOCS_PATH),
    max_goldens_per_context=3,
    context_construction_config=context_config
)
synthesizer.save_as(
    file_type="json",
    directory=GOLDENS_DIR.as_posix(),
    file_name="goldens_dataset"
)