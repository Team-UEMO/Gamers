import yaml

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap

with open("칭호생성.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 모델 구상 : (데이터 형태 변환) -> 칭호 생성 -> 검증 -> 값 추출
generator = OllamaLLM(model="gemma3:latest")
generate_prompt = PromptTemplate.from_template(
    config["generate"]
)

validator = OllamaLLM(model="gemma3:latest")
validate_prompt = PromptTemplate.from_template(
    config["validate"]
)

# formatter = OllamaLLM(model="exaone-deep:latest")
# format_prompt = PromptTemplate.from_template(
#     config["format"]
# )

generate_chain = generate_prompt | generator
validate_chain = validate_prompt | validator
# format_chain = format_prompt | formatter

# chain = generate_chain
chain = generate_chain | RunnableMap({"generation": generate_chain}) | \
        validate_chain

result = chain.invoke({"game_log": config["log1"]})
print(result)