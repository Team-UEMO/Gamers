# 질문에 따라 전문가 모델을 분기해서 사용하는 구조

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnableBranch

# 전문가 모델들
summarizer = OllamaLLM(model="llama2:7b")
translator = OllamaLLM(model="mistral:latest")
analyzer = OllamaLLM(model="gemma:latest")

# 공통 프롬프트
base_prompt = PromptTemplate.from_template("{input}")

# 라우터 (Rule-based 예시)
def route_by_keyword(input: dict):
    text = input["input"]
    if "번역" in text:
        return base_prompt | translator
    elif "요약" in text:
        return base_prompt | summarizer
    else:
        return base_prompt | analyzer

router_chain = RunnableLambda(route_by_keyword)

# 실행
result = router_chain.invoke({"input": "이 문장을 요약해줘"})
# invoke()에 전달하는 딕셔너리의 key 이름은 prompt 또는 체인의 입력 변수명과 일치해야 합니다.
print(result)
