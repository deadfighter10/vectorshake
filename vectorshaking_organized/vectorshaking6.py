import numpy as np
from vectorshake import Trainer, Parameters
from SetFit import train_setfit
import json
import datetime

args = Parameters()
final_results = {}
baseline_data = []
setfit_data = []
vectorshake_data = []
baseline_time = []
setfit_time = []
vectorshake_time = []
dataset_name1 = "go_emotions"

def save(fin):
    with open('results/results.json', 'w') as outfile:
        json.dump(fin, outfile, indent=4)

def load():
    with open('results/results.json', 'r') as outfile:
        data = json.load(outfile)
        return data

final_results = load()

def research(name, column_mapping=None, start=1, end=10):
    global final_results
    dataset_name = name
    baseline_data = []
    setfit_data = []
    vectorshake_data = []
    baseline_time = []
    setfit_time = []
    vectorshake_time = []
    if dataset_name not in final_results.keys():
        final_results[dataset_name] = {}
    for i in range(start, end+1):
        column_mapping = column_mapping
        trainer = Trainer(dataset_name=dataset_name, K_class=i, epochs=30, seed=42, args=args, column_mapping=column_mapping)
        time1 = datetime.datetime.now()
        acc = trainer.train_baseline()
        baseline_time.append((datetime.datetime.now() - time1).total_seconds())
        baseline_data.append(acc)
        """time2 = datetime.datetime.now()
        acc = train_setfit(dataset=dataset_name, K_class=i, epochs=30, seed=42, column_mapping=column_mapping)
        setfit_time.append((datetime.datetime.now() - time2).total_seconds())
        setfit_data.append(acc["accuracy"])"""
        trainer.search_parameters(trials=50)
        trainer.train()
        trainer.evaluate()
        time3 = datetime.datetime.now()
        trainer.train()
        acc = trainer.evaluate()["accuracy"]
        vectorshake_time.append((datetime.datetime.now() - time3).total_seconds())
        vectorshake_data.append(acc)
        print(baseline_data)
        print(vectorshake_data)
        final_results[dataset_name]["baseline"] = baseline_data
        final_results[dataset_name]["setfit"] = setfit_data
        final_results[dataset_name]["vectorshake"] = vectorshake_data
        final_results[dataset_name]["baseline_time"] = baseline_time
        final_results[dataset_name]["setfit_time"] = setfit_time
        final_results[dataset_name]["vectorshake_time"] = vectorshake_time
        save(final_results)
    final_results[dataset_name]["baseline"] = baseline_data
    final_results[dataset_name]["setfit"] = setfit_data
    final_results[dataset_name]["vectorshake"] = vectorshake_data
    final_results[dataset_name]["baseline_time"] = baseline_time
    final_results[dataset_name]["setfit_time"] = setfit_time
    final_results[dataset_name]["vectorshake_time"] = vectorshake_time
    save(final_results)


#research("go_emotions", column_mapping={"text": "text", "label": "labels"})
#research("ag_news", column_mapping={"text": "text", "label": "label"})
#research("trec", column_mapping={"text": "text", "label": "coarse_label"})

final_results = load()
#TODO Multi-run for each model, 5 SetFit and Vectorshake

def plot_results(dataset_name):
    global final_results
    eval_globals = {
        "datetime": datetime, # Keep the module available if needed
        "__builtins__": __builtins__
    }

    with open(f"results/results_{dataset_name}.txt", "r") as f:
        data = f.readlines()
        baseline_data = [i['accuracy'] for i in list(eval(data[0].strip()))]
        setfit_data = list(eval(data[1].strip()))
        vectorshake_data = list(eval(data[2].strip()))
        baseline_time_td = eval(data[3].strip(), eval_globals)  # List of timedeltas
        setfit_time_td = eval(data[4].strip(), eval_globals)  # List of timedeltas
        vectorshake_time_td = eval(data[5].strip(), eval_globals)  # List of timedeltas

        # --- Convert timedeltas to total seconds ---
        baseline_time = [td.total_seconds() for td in baseline_time_td]
        setfit_time = [td.total_seconds() for td in setfit_time_td]
        vectorshake_time = [td.total_seconds() for td in vectorshake_time_td]
        print(baseline_data)
        print(setfit_data)
        print(vectorshake_data)
        print(baseline_time)
        print(setfit_time)
        print(vectorshake_time)
        #print(json.loads(str(data[0])))

    # Plotting the results
    from matplotlib import pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(baseline_data)+1), baseline_data, marker='o', label='Baseline')
    plt.plot(range(1, len(setfit_data)+1), setfit_data, marker='o', label='SetFit')
    plt.plot(range(1, len(vectorshake_data)+1), vectorshake_data, marker='o', label='VectorShake')
    plt.title(f'Model Accuracy Comparison - {dataset_name}')
    plt.xlabel('Datapoint per Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid()
    plt.show()
    # Save the plot

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(baseline_time)+1), baseline_time, marker='o', label='Baseline')
    plt.plot(range(1, len(setfit_time)+1), setfit_time, marker='o', label='SetFit')
    plt.plot(range(1, len(vectorshake_time)+1), vectorshake_time, marker='o', label='VectorShake')
    plt.title(f'Model Training Time Comparison - {dataset_name}')
    plt.xlabel('Datapoint per Class')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid()
    plt.show()

"""plot_results("go_emotions")
plot_results("ag_news")
plot_results("trec")"""
final_results = load()

# AblationStudy.txt
def ablation_study(dataset_name, k_class, column_mapping=None, DAC_off=False):
    trainer = Trainer(dataset_name=dataset_name, K_class=k_class, epochs=30, seed=42, args=args, column_mapping=column_mapping)
    trainer.train_baseline()
    trainer.train()
    trainer.evaluate()

    is_DAC_off = DAC_off

    trainer.train()
    trainer.evaluate()

    trainer.get_best_params()
    trainer.switch_DAC_off = is_DAC_off
    trainer.train()
    metr = trainer.evaluate()

    trainer.get_best_params()
    trainer.args.lambda_mix = 0
    trainer.manual_params = True
    trainer.switch_DAC_off = is_DAC_off
    trainer.train()
    metr_mix = trainer.evaluate()

    trainer.get_best_params()
    trainer.args.lambda_contrast = 0
    trainer.manual_params = True
    trainer.switch_DAC_off = is_DAC_off
    trainer.train()
    metr_contrast = trainer.evaluate()

    trainer.get_best_params()
    trainer.args.lambda_consistency = 0
    trainer.manual_params = True
    trainer.switch_DAC_off = is_DAC_off
    trainer.train()
    metr_consist = trainer.evaluate()

    trainer.get_best_params()
    trainer.args.lambda_reconstruct = 0
    trainer.manual_params = True
    trainer.switch_DAC_off = is_DAC_off
    trainer.train()
    metr_reconstruct = trainer.evaluate()

    trainer.get_best_params()
    trainer.args.epsilon = 0
    trainer.manual_params = True
    trainer.switch_DAC_off = is_DAC_off
    trainer.train()
    metr_epsilon = trainer.evaluate()

    print("--- Ablation Study Results ---")
    print(f"Lambda Mix 0: {metr_mix}")
    print(f"Lambda Contrast 0: {metr_contrast}")
    print(f"Lambda Consistency 0: {metr_consist}")
    print(f"Lambda Reconstruct 0: {metr_reconstruct}")
    print(f"Epsilon 0: {metr_epsilon}")
    offon = "off" if is_DAC_off else "on"
    print(f"Base (every parameter on, DAC {offon}): {metr}")
    print("--- End of Ablation Study ---")
    if "ablation_study" not in final_results[dataset_name].keys():
        final_results[dataset_name]["ablation_study"] = {}

    # Save the results to the final_results dictionary
    final_results[dataset_name]["ablation_study"][f"DAC_{offon}"] = {
        "lambda_mix": metr_mix,
        "lambda_contrast": metr_contrast,
        "lambda_consistency": metr_consist,
        "lambda_reconstruct": metr_reconstruct,
        "epsilon": metr_epsilon,
        "base": metr,
        "k_class": k_class
    }
    save(final_results)

ablation_study("TREC", 5, column_mapping={"text": "text", "label": "coarse_label"}, DAC_off=False)
ablation_study("TREC", 5, column_mapping={"text": "text", "label": "coarse_label"}, DAC_off=True)
ablation_study("ag_news", 5, column_mapping={"text": "text", "label": "label"}, DAC_off=False)
ablation_study("ag_news", 5, column_mapping={"text": "text", "label": "label"}, DAC_off=True)