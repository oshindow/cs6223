import pandas as pd
from llm_mutant import CodexModel
# df = pd.read_csv("results/gqa/trials/results_15.csv") # refine 1 imcomplete code
# logfile = "nohup.out.refine"
# df = pd.read_csv("results/gqa/trials/results_16.csv") # baseline 
# logfile = "nohup.out.baseline"
# df = pd.read_csv("results/gqa/trials/results_22.csv") # refine 4 test case gen
# logfile = "nohup.out"
# df = pd.read_csv("results/gqa/trials/results_41.csv") # test 500 cases
# logfile = "nohup.out"

df = pd.read_csv("results/gqa/trials/results_42.csv") # test 500 cases
logfile = "nohup.out.last_all"

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))

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

with open('nohup.out.soundness') as f:
    for line in f:
        if 'soundness' in line:
            # Sample 20797830 Soundness: 1.0
            try:
                sample_id = line.split("key ")[1].split(" soundness")[0]
                soundness_value = float(line.split("soundness: ")[1].strip())
                assert int(sample_id) in datas
                datas[int(sample_id)]["soundness"] = soundness_value
            except Exception as e:
                print(e, line)
                continue


fp = 0
cnt = 0
fn = 0
tp = 0
tn = 0
fp_data = {}
fn_data = {}
tp_data = {}
tn_data = {}
for key, value in datas.items():
    if not value["failed"] and value["wrong"]:
        fp += 1
        # if (value["answer"] not in ["no", "yes"]) and (value["result"] not in ["no", "yes"]):
        # print(value["result"], "|", value["answer"], "|", value["query"], key)
        fp_data[key] = {"result": value["result"], "answer": value["answer"], "query": value["query"], "test_code": value["test_code"]}
            # cnt += 1
    if value["failed"] and not value["wrong"]:
        fn += 1
        fn_data[key] = {"result": value["result"], "answer": value["answer"], "query": value["query"], "test_code": value["test_code"]}

    if not value['failed'] and not value["wrong"]:
        tp += 1
        tp_data[key] = {"result": value["result"], "answer": value["answer"], "query": value["query"], "test_code": value["test_code"]}

    
    if value["failed"] and value["wrong"]:
        tn += 1
        tn_data[key] = {"result": value["result"], "answer": value["answer"], "query": value["query"], "test_code": value["test_code"]}



tp_count = 0
tn_count = 0
fp_count = 0
fn_count = 0
for key, value in datas.items():
    if value["soundness"] == 1.0:
        if key in tp_data:
            tp_count += 1
        elif key in tn_data:
            tn_count += 1
        elif key in fp_data:
            fp_count += 1
        elif key in fn_data:
            fn_count += 1
    
import json
with open('result_soundness.json', 'w', encoding='utf8') as output:
    json.dump(datas, output, indent=4)

print("precentage of tp", tp_count/len(tp_data))
print("precentage of tn", tn_count/len(tn_data))
print("precentage of fp", fp_count/len(fp_data))
print("precentage of fn", fn_count/len(fn_data))


# precentage of tp 0.714975845410628
# precentage of tn 0.33879781420765026
# precentage of fp 0.6827820186598813
# precentage of fn 0.3134715025906736