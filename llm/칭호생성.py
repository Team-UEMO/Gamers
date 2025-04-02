import yaml
import time

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.exceptions import OutputParserException
# from langchain_core.pydantic_v1 import BaseModel, Field

with open("칭호생성.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 모델 구상 : (데이터 형태 변환) -> 칭호 생성 -> 검증 -> 값 추출
creator = OllamaLLM(model=config['creator'])
rewriter = OllamaLLM(model=config['rewriter'])

create_prompt = PromptTemplate.from_template(
    config["create"]
)
rewrite_prompt = PromptTemplate.from_template(
    config["rewrite"]
)

create_chain = create_prompt | creator    # 1.54s
rewrite_chain = rewrite_prompt | rewriter    # 1.15s

def create_with_retry(input_dict, retry_count=3):
    def check_output_fn(output: str) -> bool:
        required_keywords = config["creator_required_keywords"]
        if all(keyword in output for keyword in required_keywords):
            return output
        raise OutputParserException("생성 결과에 필수 항목이 없습니다.")

    for _ in range(retry_count):
        generation = create_chain.invoke(input_dict)
        try:
            validated = check_output_fn(generation)
            return {"generation": validated}
        except OutputParserException as e:
            print(f"[ERROR] {e}")
    raise Exception("최대 재시도 횟수를 초과했습니다.")
validated_create_chain = RunnableLambda(lambda x: create_with_retry(x, retry_count=3))

# chain = create_chain
chain = validated_create_chain | rewrite_chain

now = time.time()
result = chain.invoke({"game_log": config["log1"]})
print(f"[INFO] Elapsed time: {time.time() - now:.2f}s")
print(f"[INFO] Output: {result}")
now = time.time()
result = chain.invoke({"game_log": config["log2"]})
print(f"[INFO] Elapsed time: {time.time() - now:.2f}s")
print(f"[INFO] Output: {result}")
now = time.time()
result = chain.invoke({"game_log": config["log3"]})
print(f"[INFO] Elapsed time: {time.time() - now:.2f}s")
print(f"[INFO] Output: {result}")