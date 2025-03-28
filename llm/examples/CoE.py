# 전문가 모델들이 순차적으로 처리하는 구조

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import JsonOutputParser

# 단계별 전문가 모델
extractor = OllamaLLM(model="llama2:7b")  # 정보 추출
reasoner = OllamaLLM(model="mistral:latest")  # 논리적 해석
summarizer = OllamaLLM(model="gemma:latest")  # 요약

# 각 단계별 프롬프트
extract_prompt = PromptTemplate.from_template("다음 문장에서 키 정보를 추출해줘:\n{input}")
reason_prompt = PromptTemplate.from_template("추출한 정보를 기반으로 의미를 해석해줘:\n{facts}")
summary_prompt = PromptTemplate.from_template("해석된 내용을 요약해줘:\n{meaning}")

# 각 단계 구성
extract_chain = extract_prompt | extractor
reason_chain = reason_prompt | reasoner
summary_chain = summary_prompt | summarizer

# CoE 스타일 순차 연결
chain = RunnableMap({"facts": extract_chain}) | \
        RunnableMap({"meaning": reason_chain}) | \
        summary_chain

# 실행
result = chain.invoke({"input": "고양이는 귀엽고, 강아지는 충성스럽다."})
print(result)
