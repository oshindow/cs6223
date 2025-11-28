import pandas as pd
# results/gqa/trials/results_42.csv (refine with two more examples in prompt)
# results/gqa/trials/results_43.csv more failed cases
# results/gqa/trials/results_45.csv refine prompt with list of possible answers
# df = pd.read_csv("results/gqa/trials/results_42.csv") # all cases (add two more examples in prompt)
# logfile = "nohup.out.last_all"
df = pd.read_csv("results/gqa/trials/results_46.csv") # refine with list of possible answers
logfile = "nohup.out"
print("Using logfile:", logfile)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))
invalid = 0
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
        "syntax": True,
        # "wrong": None,
    }

    datas[row["id"]] = data
    if """assert len(result.split()) in [1,2], \"Expected output to be one or two words\"\n    return result""" in row["test_code"]:
        # datas[row["id"]]["test_type"] = "short_answer"
        data["invalid"] = True 
        invalid += 1
    else:
        # datas[row["id"]]["test_type"] = "general_answer"
        data["invalid"] = False 
        # cnt
    # += 1
print(f"Number of invalid samples: {invalid}")
# print(len(datas))
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
        # if 'Batch 400' in line:
            # break
        if 'failed' in line:
            # print("Process failed")
            # Sample 20797830 failed with error: Expected output to be a person. Next you will see an "expected an indented block" error. 
            try:
                sample_id = line.split("Sample ")[1].split(" failed")[0]
                failed_samples[sample_id] = sample_id
                datas[int(sample_id)]["failed"] = True
                
                # datas[int(sample_id)]["syntax"] = line.strip()
                if 'failed with error' in line:
                    datas[int(sample_id)]["syntax"] = False
            except Exception as e:
                print(e)
                continue
        # else:

print(f"Number of failed samples: {len(failed_samples)}")

fp = 0
cnt = 0
fn = 0
tp = 0
tn = 0
fp_invalid = 0
tn_invalid = 0
tp_invalid = 0
fn_invalid = 0
syntax = 0
fp_data = {}
fn_data = {}
tp_data = {}
tn_data = {}
for key, value in datas.items():
    if value["failed"] is None:
        continue
    # if value["result"] in ["yes", "no"]:
        # continue    

    if not value["failed"] and value["wrong"]:
        fp += 1
        # if (value["answer"] not in ["no", "yes"]) and (value["result"] not in ["no", "yes"]):
        # print(value["result"], "|", value["answer"], "|", value["query"], key)
        fp_data[key] = {"result": value["result"], "answer": value["answer"], "query": value["query"], "test_code": value["test_code"]}
        if value["invalid"]:
            fp_invalid += 1
    if value["failed"] and not value["wrong"]:
        fn += 1
        fn_data[key] = {"result": value["result"], "answer": value["answer"], "query": value["query"], "test_code": value["test_code"]}
        if value["invalid"]:
            fn_invalid += 1
        
        if value["syntax"] == True:
            syntax += 1
    if not value['failed'] and not value["wrong"]:
        tp += 1
        tp_data[key] = {"result": value["result"], "answer": value["answer"], "query": value["query"], "test_code": value["test_code"]}
        if value["invalid"]:
            tp_invalid += 1
    if value["failed"] and value["wrong"]:
        tn += 1
        tn_data[key] = {"result": value["result"], "answer": value["answer"], "query": value["query"], "test_code": value["test_code"]}
        if value["invalid"]:
            tn_invalid += 1
    
print(f"Number of invalid true positives: {tp_invalid}")
print(f"Number of invalid false negatives: {fn_invalid}")
print(f"Number of invalid false positives: {fp_invalid}")
print(f"Number of invalid true negatives: {tn_invalid}")
print(f"Number of true positives: {tp} {tp/len(datas)}")
print(f"Number of false positives: {fp} {fp/len(datas)}")
print(f"Number of true negatives: {tn} {tn/len(datas)}")
print(f"Number of false negatives: {fn} {fn/len(datas)}")
print(f"Number of syntax errors in false negatives: {syntax}")
print(f"assertion erros: {fn - syntax}")
# print(len(fn_data))
# print(fn_data)
import json
with open('fn_data_46.json', 'w') as f:
    json.dump(fn_data, f, indent=4)
# print(tn_data.keys())