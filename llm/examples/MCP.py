# (doc, question) 
#    ↓
# [doc_expert] → 요약 + 신뢰도  
# [chat_expert] → 응답 + 신뢰도  
#    ↓
# [Fusion Expert] ← 둘의 출력과 신뢰도를 받아, 
#                  하나의 결합 응답 생성

# fusion expert는 내가 설계한 방법으로, 병렬로 받은 두 전문가의 출력을 어떻게 결합할지는 자유롭게 결정한다.



from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnableLambda

# 모델 구성
doc_expert = OllamaLLM(model="gemma3:latest")     # 문서 요약 담당
chat_expert = OllamaLLM(model="gemma3:latest")   # 대화 대응 담당
fusion_expert = OllamaLLM(model="gemma3:latest")  # 응답 결합 담당

# 각각 다른 prompt로 문맥 구성
doc_prompt = PromptTemplate.from_template("이 문서를 요약해줘:\n\n{doc}")
chat_prompt = PromptTemplate.from_template("사용자 질문:\n{question}")

# 개별 체인
doc_chain = doc_prompt | doc_expert
chat_chain = chat_prompt | chat_expert

# MCP 스타일: 여러 모델이 서로 다른 문맥 담당
expert_outputs = RunnableMap({
    "doc_summary": doc_chain,
    "chat_response": chat_chain,
})

# Fusion expert를 위한 프롬프트
fusion_prompt = PromptTemplate.from_template("""
다음은 두 명의 전문가가 각기 응답한 결과와 해당 신뢰도 점수야.

- 문서 요약 (신뢰도 {doc_conf:.2f}): {doc_summary}
- 대화 응답 (신뢰도 {chat_conf:.2f}): {chat_response}

이 정보를 바탕으로 사용자의 질문에 가장 적절한 하나의 통합 응답을 생성해줘.
""")

# 신뢰도 평가 함수 (임시로 길이에 따라 점수 산정)
def estimate_confidence(output: str) -> float:
    return min(len(output) / 100, 1.0)  # 길이가 100 이상이면 1.0

# Fusion 단계
def build_fusion_input(outputs):
    doc_summary = outputs["doc_summary"]
    chat_response = outputs["chat_response"]
    doc_conf = estimate_confidence(doc_summary)
    chat_conf = estimate_confidence(chat_response)
    
    return {
        "doc_summary": doc_summary,
        "chat_response": chat_response,
        "doc_conf": doc_conf,
        "chat_conf": chat_conf
    }

# 전체 MCP 체인 구성: 전문가 → 병렬 출력 → Fusion expert로 통합 응답
mcp_chain = (
    expert_outputs
    | RunnableLambda(build_fusion_input)
    | (fusion_prompt | fusion_expert)
)

# 입력 예시
result = mcp_chain.invoke({
    "doc": "LangChain은 Runnable 기반과 유연한 확장 구조를 바탕으로 체인을 명확하게 구성할 수 있으며, Agent 및 Memory 시스템을 통해 복잡한 기능도 손쉽게 구현 가능합니다. 이러한 구조는 사용자가 모듈을 조립하듯 쉽게 체인을 구성할 수 있게 하여, 개발 난이도를 줄이고 빠른 실험과 적용이 가능하게 해 사용자 경험을 향상시킵니다.",
    "question": "LangChain의 장점은 뭔가요?"
})

print(result)