from deepeval.dataset.dataset import EvaluationDataset
from deepeval.test_case.llm_test_case import LLMTestCase
from time import sleep

from dotenv import load_dotenv
load_dotenv()

from src.app.rag_workflow import workflow

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT_DIR/"data"/"evaluation"
GOLDENS_PATH = EVALUATION_DIR/"goldens"/"goldens_dataset.json"
EVALUATION_DATA_DIR = EVALUATION_DIR/"eval_dataset"
EVALUATION_DATA_DIR.mkdir(exist_ok=True,parents=True)


goldens_dataset = EvaluationDataset()
goldens_dataset.add_goldens_from_json_file(file_path=str(GOLDENS_PATH))
evaluation_dataset = EvaluationDataset()

for golden in goldens_dataset.goldens:
    final_state = workflow.invoke({"query":golden.input})
    sleep(3)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=final_state.get("response"),
        expected_output = golden.expected_output,
        retrieval_context=[doc.page_content for doc in final_state.get("retrieved_docs")]
    )
    evaluation_dataset.add_test_case(test_case=test_case)
evaluation_dataset.save_as(
    file_type="json",
    directory=EVALUATION_DATA_DIR,
    file_name="evaluation_dataset",
    include_test_cases=True
)