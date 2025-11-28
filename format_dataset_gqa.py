import json
import csv


output = open("data/queries.csv", "w", newline="", encoding="utf-8") 
writer = csv.DictWriter(
        output,
        fieldnames=["index", "sample_id", "possible_answers", "query_type", "query", "answer", "image_name"]
    )
writer.writeheader()
datas = []
with open('data/GQA/testdev_balanced_questions.json', 'r') as f:
    data = json.load(f)
    for key, item in data.items():

        data = {
            "index": item["equivalent"][0],
            "sample_id": item["equivalent"][0],
            "possible_answers": item["fullAnswer"],
            "query_type": item["groups"]["local"],
            "query": item["question"],
            "answer": item["answer"],
            "image_name": item["imageId"] + ".jpg"
        }
        datas.append(data)
        

    writer.writerows(datas)