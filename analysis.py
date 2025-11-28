import pandas as pd
from llm import CodexModel
# df = pd.read_csv("results/gqa/trials/results_15.csv") # refine 1 imcomplete code
# logfile = "nohup.out.refine"
df = pd.read_csv("results/gqa/trials/results_16.csv") # baseline 
logfile = "nohup.out.baseline"
# df = pd.read_csv("results/gqa/trials/results_22.csv") # refine 4 test case gen
# logfile = "nohup.out"

datas = {}
for idx, row in df.iterrows():
    data = {
        "result": row["result"],
        "answer": row["answer"],
        "code": row["code"],
        "test_code": row["test_code"],
        "query": row["query"],
        "img_path": row["img_path"],
        "possible_answers": row["possible_answers"],
        "failed": False,
    }

    datas[row["id"]] = data

wrong_results = 0
correct_results = 0
for id, item in datas.items():
    if item["result"] != item["answer"]:
        wrong_results += 1
        datas[id]["wrong"] = True
    else:
        correct_results += 1
        datas[id]["wrong"] = False
failed_samples = {}
with open(logfile) as f:
    lines = f.readlines()
    for line in lines:
        if 'failed' in line:
            # print("Process failed")
            # Sample 20797830 failed with error: Expected output to be a person. Next you will see an "expected an indented block" error. 
            try:
                sample_id = line.split("Sample ")[1].split(" failed")[0]
                failed_samples[sample_id] = sample_id
                datas[int(sample_id)]["failed"] = True
            except Exception as e:
                print(e)
                continue
print(f"Number of failed samples: {len(failed_samples)}")
fp = 0
cnt = 0
fn = 0
fp_data = {}
fn_data = {}
for key, value in datas.items():
    if not value["failed"] and value["wrong"]:
        fp += 1
        # if (value["answer"] not in ["no", "yes"]) and (value["result"] not in ["no", "yes"]):
        print(value["result"], "|", value["answer"], "|", value["query"], key)
        fp_data[key] = {"result": value["result"], "answer": value["answer"], "query": value["query"]}
            # cnt += 1
    if value["failed"] and not value["wrong"]:
        fn += 1
        fn_data[key] = {"result": value["result"], "answer": value["answer"], "query": value["query"]}

# print(f"Number of false positives (passed and wrong): {fp} {cnt} {cnt/fp}")
# print(f"false positive rate: {fp / wrong_results}")

llm = CodexModel()
indistinguishable = 0
distinguishable = 0
for key, data in fp_data.items():
    result = llm.forward(data["result"], data["answer"], data["query"])
    fp_data[key]["llm"] = result[0] 
    # print(data, result)
    if result[0] == "indistinguishable":
        indistinguishable += 1
        fp_data[key]["distinguishable"] = False
        # print("inDistinguishable case:")
        # print(f"Query: {data['query']}")
        # print(f"Answer: {data['answer']}")
        # print(f"Result: {data['result']}")
        # print(f"LLM: {result[0]}")
    else:
        distinguishable += 1
        fp_data[key]["distinguishable"] = True
        # print("Distinguishable case:")
        # print(f"Query: {data['query']}")
        # print(f"Answer: {data['answer']}")
        # print(f"Result: {data['result']}")
        # print(f"LLM: {result[0]}")
print(f"Correct results: {correct_results}")
print(f"Wrong results: {wrong_results}")
print(f"Accuracy: {correct_results / (correct_results + wrong_results)}")

print(f"Indistinguishable: {indistinguishable}, Distinguishable: {distinguishable}")
print(f"Number of failed samples: {len(failed_samples)}")
print(f"Number of false positives (passed and wrong): {fp} {cnt} {cnt/fp}")
print(f"false positive rate: {fp / wrong_results}")
print(f"Number of fale negative (failed and correct): {fn}")
print(f"false negative rate: {fn / correct_results}")