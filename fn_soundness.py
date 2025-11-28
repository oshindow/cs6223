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

df = pd.read_csv("results/gqa/trials/results_42.csv") # all cases (add two more examples in prompt)
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


print(f"Number of true positives: {tp} {tp/len(datas)}")
print(f"Number of false positives: {fp} {fp/len(datas)}")
print(f"Number of true negatives: {tn} {tn/len(datas)}")
print(f"Number of false negatives: {fn} {fn/len(datas)}")
print(len(fn_data))
# print()
llm = CodexModel()

from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno, process_guesses
# from video_segment import VideoSegment
from functools import partial

indistinguishable = 0
distinguishable = 0

queue_results = None
queues_in_ = None

print(len(fn_data))
idx = 0
for key, data in datas.items():
    if idx % 100 == 0:
        print(idx)
    idx += 1
    result = llm.forward(data["result"], data["answer"], data["query"])
    data["mutants"] = result
    queues = [queues_in_, queue_results]

    llm_query_partial = partial(llm_query, queues=queues)
    process_guesses_partial = partial(process_guesses, queues=queues)

    
    test_code = data["test_code"]
    test_code = test_code.replace("def execute_test(image):", "def execute_test(result," \
                                "llm_query, bool_to_yesno, distance, best_image_match, process_guesses):")
    
    test_code = test_code.replace("result = execute_command(image, my_fig, time_wait_between_lines, syntax)", "")
    
    # print(test_code)
    exec(compile(test_code, 'Codex', 'exec'), globals())
    
    mutants = data["mutants"]
    
    failed = 0
    passed = 0
    for mutant in mutants:
        try:
            
            output = globals()["execute_test"](mutant, llm_query_partial, bool_to_yesno, distance, best_image_match, process_guesses_partial)
            # print(output)
            passed += 1
        except Exception as e:
            # print(mutant, e)
            failed += 1
    
    soundness = passed / (passed + failed)
    print("key", key, "soundness:", soundness)
    datas[key]["soundness"] = soundness

# sorted_data = dict(
#     sorted(fn_data.items(), key=lambda x: x[1]["soundness"], reverse=True)
# )

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
    json.dump(datas, indent=4)

print("precentage of tp", tp_count/len(tp_data))
print("precentage of tn", tn_count/len(tn_data))
print("precentage of fp", fp_count/len(fp_data))
print("precentage of fn", fn_count/len(fn_data))
