from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-large")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")

input_text="""
    Task: The task is to find the relation between two entities in the input text. Below are some examples demonstrating the same. The examples are in English, but the test input will be in French.
    Note: The relation should be one of (has-type, from-country, has-nationality).
    Sentence: The Saskatoon Sanatorium was a tuberculosis sanatorium established in 1925 by the Saskatchewan Anti-Tuberculosis League as the second Sanatorium in the province in Wellington Park south or the Holiday Park neighborhood of Saskatoon, Saskatchewan, Canada.
    Relation: has-type
    Sentence: Bronson Beri (born 26 June 1989) is a New Zealand professional basketball player for the Nelson Giants of the National Basketball League (NBL).
    Relation: from-country
    Sentence: Amancio Ortega Gaona (Spanish pronunciation: [aˈmanθjo oɾˈteɣa ɣaˈona], born 28 March 1936) is a Spanish billionaire businessman.
    Relation: has-nationality
    Sentence: .gb is a reserved Internet country code top-level domain (ccTLD) of the United Kingdom.
    Relation: has-type
    Sentence: Gérard Paul Louis Marie-Joseph Mulliez (born 13 May 1931) is a French businessman. Gérard Mulliez was born on 13 May 1931 in Roubaix, France.
    Relation: from-country
    Sentence: Professor Makame Mbarawa> is a Tanzanian CCM politician and a nominated Member of Parliament.
    Relation: has-nationality
    
    French Sentence: Richard Challoner, né le 29 septembre 1691 à Lewes et mort le 12 janvier 1781 à Londres, est un évêque catholique anglais, figure majeure du catholicisme anglais durant une grande partie du XVIIIe siècle, et célèbre pour sa révision de la traduction de la Bible de Douai.
    Relation: 
"""

# input_text="""
#     Task: The task is to find the relation between two entities in the input text. Below are some examples demonstrating the same. The examples are in English, but the test input will be in Russian.
#     Note: The relation should be one of (has-type, from-country, has-nationality).
#     Sentence: The Saskatoon Sanatorium was a tuberculosis sanatorium established in 1925 by the Saskatchewan Anti-Tuberculosis League as the second Sanatorium in the province in Wellington Park south or the Holiday Park neighborhood of Saskatoon, Saskatchewan, Canada.
#     Relation: has-type
#     Sentence: Bronson Beri (born 26 June 1989) is a New Zealand professional basketball player for the Nelson Giants of the National Basketball League (NBL).
#     Relation: from-country
#     Sentence: Amancio Ortega Gaona (Spanish pronunciation: [aˈmanθjo oɾˈteɣa ɣaˈona], born 28 March 1936) is a Spanish billionaire businessman.
#     Relation: has-nationality
#     Sentence: .gb is a reserved Internet country code top-level domain (ccTLD) of the United Kingdom.
#     Relation: has-type
#     Sentence: Gérard Paul Louis Marie-Joseph Mulliez (born 13 May 1931) is a French businessman. Gérard Mulliez was born on 13 May 1931 in Roubaix, France.
#     Relation: from-country
#     Sentence: Professor Makame Mbarawa> is a Tanzanian CCM politician and a nominated Member of Parliament.
#     Relation: has-nationality
    
#     Russian Sentence: Джакомо Биффи (; 13 июня 1928, королевство Италия, Италия — 11 июля 2015, Болонья, Италия) — итальянский кардинал.
#     Relation: 
# """

set_seed(42)

# For the expected format of input_text, see Intended use above
inputs = tokenizer(input_text, return_tensors="pt")


# model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=400, do_sample=True, top_p=0.9, num_return_sequences=1,
#             eos_token_id=tokenizer.eos_token_id, remove_invalid_values=True, no_repeat_ngram_size=2, temperature=1.5)
outputs = model.generate(**inputs, max_new_tokens=6, num_beams=2)


print("Answer:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))