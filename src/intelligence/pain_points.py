import re
import pandas as pd

PAIN_POINT_KEYWORDS={
 "Delivery delays":["late delivery","delayed delivery","delivery delay","atraso","atrasada","atrasado","demorou","prazo","entrega atrasada"],
 "Product quality":["poor quality","defective","broken","damaged","qualidade","defeito","quebrado","danificado","avariado"],
 "Wrong or missing item":["wrong product","missing item","missing product","produto errado","faltou","item faltando","pedido incompleto"],
 "Customer service":["customer service","no response","support","atendimento","sem resposta","suporte","não respondem","nao respondem"],
 "Description mismatch":["not as described","different from the picture","description","descrição","descricao","diferente da foto","não corresponde","nao corresponde"],
 "Packaging problems":["poor packaging","packaging damaged","embalagem","mal embalado","embalagem danificada"],
 "Refund or payment issues":["refund","reimbursement","charged twice","reembolso","estorno","cobrado duas vezes"],
}
def detect_pain_points(reviews):
    result=reviews.copy()
    if "review_text" not in result: result["review_text"]=""
    labels=[]
    for value in result["review_text"].fillna("").astype(str).str.lower():
        found=[label for label,words in PAIN_POINT_KEYWORDS.items() if any(re.search(re.escape(word),value) for word in words)]
        labels.append("; ".join(found))
    result["pain_points"]=labels
    result["has_pain_point"]=result["pain_points"].ne("")
    return result

def summarize_pain_points(reviews):
    if "pain_points" not in reviews: reviews=detect_pain_points(reviews)
    values=reviews.loc[reviews["pain_points"].ne(""),"pain_points"]
    if values.empty:return pd.DataFrame(columns=["Pain point","Review matches"])
    return values.str.split("; ").explode().value_counts().rename_axis("Pain point").reset_index(name="Review matches")
