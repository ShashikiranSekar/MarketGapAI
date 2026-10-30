import os
from dotenv import load_dotenv
load_dotenv()

def get_llm_configuration_status():
    return bool(os.getenv("GEMINI_API_KEY","").strip())

GUIDANCE="""
You are MarketGap AI, an evidence-grounded market research assistant.
Use only supplied evidence for factual claims. Separate observed facts from hypotheses.
Do not invent market sizes, competitor facts, current trends, or profitability.
Olist is historical Brazilian e-commerce data, primarily from 2016–2018.
For each opportunity include: hypothesis, target customer, supporting evidence, MVP features,
differentiation hypothesis, monetization options, risks, low-cost validation experiment,
measurable success criteria, and confidence. If evidence is insufficient, state that clearly.
Competitor entries are user-supplied and unverified unless the evidence explicitly says otherwise.
"""

def generate_opportunity_report(evidence,focus="",industry="E-commerce"):
    key=os.getenv("GEMINI_API_KEY","").strip()
    if not key:raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env or cloud secrets.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install google-genai using requirements.txt.") from exc
    client=genai.Client(api_key=key)
    prompt=f"""{GUIDANCE}
Selected industry: {industry}
Research focus: {focus or "General opportunity discovery"}
Evidence:
{evidence[:28000]}

Structure:
1. Executive summary
2. Observed evidence
3. Up to five opportunity hypotheses in a table
4. Target customer profiles
5. Proposed MVP features
6. Monetization hypotheses
7. Risks and evidence gaps
8. Validation experiments with measurable success criteria
9. Confidence and assumptions
Do not claim external competitor research was done unless it is in the supplied evidence.
"""
    response=client.models.generate_content(model=os.getenv("GEMINI_MODEL","gemini-3.8-flash"),contents=prompt,config=types.GenerateContentConfig(temperature=0.2))
    return response.text or "No text returned by the model."
