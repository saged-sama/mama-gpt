import evaluate
import json

with open("logs/summaries.json", mode="r", encoding="utf-8") as file:
    summaries = json.load(file)
    
predictions = []
bleu_refs = []
rouge_refs = []

for summary in summaries:
    article = summary.get("article", "")
    summ = summary.get("summary", "")
    
    predictions.append(summ)
    bleu_refs.append([article])
    rouge_refs.append(article)
    
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bleu_results = bleu.compute(predictions=predictions, references=bleu_refs)
rouge_results = rouge.compute(predictions=predictions, references=rouge_refs)

with open("logs/bleu_rouge_res.json", "w", encoding="utf-8") as file:
    file.write(json.dumps({"bleu_results": bleu_results, "rouge_results": rouge_results}, indent=4))
    
print(json.dumps({"bleu_results": bleu_results, "rouge_results": rouge_results}, indent=4))