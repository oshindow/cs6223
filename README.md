# CS6223 software testing

This is a repo for the codebase of NUS CS6223 course project: On Re-ranking Strategy for In-Context Learning in LLM Property-based Test Cases Generation. 

This Project is movitived by PropTest: Automatic Property Testing for Improved Visual Programming. This is the code for the paper [PropTest: Automatic Property Testing for Improved Visual Programming](https://jaywonkoo17.github.io/PropTest/) 

### [Paper (EMNLP 2024 Findings)](https://arxiv.org/pdf/2403.16921) | [Project Page](https://jaywonkoo17.github.io/PropTest/)


# Experimental results

### nohup.out for the proposed model

```bash
tail nohup.out

Accuracy at Batch 628/629: 0.4697090157417714
100%|██████████| 629/629 [32:43:04<00:00, 187.26s/it]
Saving results to results_46.csv at epoch 628
Final accuracy: 0.4697090157417714
Saving results to results_46.csv
```

### nohup.out.last_all for the baseline (advanced PropTest)

```bash
tail nohup.out.last_all

Accuracy at Batch 628/629: 0.4587374781364287
100%|██████████| 629/629 [30:27:25<00:00, 174.32s/it]
Saving results to results_42.csv at epoch 628
Final accuracy: 0.4587374781364287
Saving results to results_42.csv
```

- `result_soundness.json`: baseline results used for computing soundness
- `compute_soundness.py`: script for computing soundness
- `count_failed.py`: script for counting classification outcomes
- `draw.py`: script for plotting the confusion matrix
- `k-means.py`: script for obtaining the k-means clustering results
- `llm_mutant.py`: script for building the LLM used to generate mutants


# Environmnet

Clone recursively:
```bash
git clone --recurse-submodules https://github.com/uvavision/PropTest.git
```

After cloning:
```bash
cd PropTest
export PATH=/usr/local/cuda/bin:$PATH
bash setup.sh  # This may take a while. Make sure the vipergpt environment is active
cd GLIP
python setup.py clean --all build develop --user
cd ..
echo YOUR_OPENAI_API_KEY_HERE > api.key
```
This code was built on top of [ViperGPT](https://github.com/cvlab-columbia/viper). We follow the same installation steps as ViperGPT. For detailed installation, please refer to the [ViperGPT repository](https://github.com/cvlab-columbia/viper).

You need to download two pretrained models and store it in ```./pretrained_models```. 
You can use ```download_models.sh``` to download the models.

# Running the Code

The code can be run using the following command:
    
```
CONFIG_NAMES=your_config_name python main_batch.py
```
```CONFIG_NAMES``` is an environment variable that specifies the configuration files to use.

