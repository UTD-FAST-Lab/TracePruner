# baseline runners

import json
from itertools import product

from approach.runners.rf_baseline import RandomForestBaseline
from approach.runners.rf_baseline_fixed_set import RandomForestBaselineFixedSet
from approach.runners.nn_baseline import NeuralNetBaseline
from approach.runners.nn_baseline_fixed_set import NeuralNetBaselineFixedSet
from approach.data_representation.instance_loader import load_instances
from approach.utils import plot_points

import argparse
parser = argparse.ArgumentParser(description="Run baseline models")
parser.add_argument("--baseline", type=str, default="cgpruner", help="Baseline model to run (cgpruner, autopruner, ml4cgp)")
parser.add_argument("--dataset", type=str, default="njr", help="Dataset to use (njr, other)")


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def get_balance_combinations(config_section):
    balance_basic = config_section["balance"]["basic"]
    balance_random = [
        f"{method}_{ratio}"
        for method, ratios in config_section["balance"]["random"].items()
        for ratio in ratios
    ]

    return balance_basic + balance_random


def run_cgpruner(instances):

    config = load_json("approach/runners/config.json")["cgPruner"]
    balance__variations = get_balance_combinations(config)

    all_runners = []

    # 1. with raw baseline
    runner = RandomForestBaselineFixedSet(
        instances=instances,
        raw_baseline=True,
        train_with_unknown=True,
        threshold=0.45,
        make_balance=False ,
        output_dir="approach/results/baseline/cgpruner_programwise/fix"
    )
    all_runners.append(runner)

    # 2. without raw baseline
    runner = RandomForestBaselineFixedSet(
        instances=instances,
        raw_baseline=False,
        train_with_unknown=True,
        threshold=0.45,
        make_balance=False ,
        output_dir="approach/results/baseline/cgpruner_programwise/fix"
    )

    all_runners.append(runner)
    
    # train on labeled data only
    for balance in balance__variations:
        # Convert the balance string into method and ratio if needed
        if "_" in balance:
            balance_method, balance_ratio = balance.split("_")
            make_balance = (balance_method, float(balance_ratio))
        else:
            make_balance = False
        
        runner = RandomForestBaselineFixedSet(
            instances=instances,
            raw_baseline=False,
            train_with_unknown=False,
            threshold=0.45,
            make_balance=make_balance,
            output_dir="approach/results/baseline/cgpruner_programwise/fix"
        )

        all_runners.append(runner)


    # run all the runners
    for runner in all_runners:
        runner.run()




def run_autopruner(instances):

    config = load_json("approach/runners/config.json")["autoPruner"]
    balance__variations = get_balance_combinations(config)
    all_runners = []
    # 1. with raw baseline
    runner = NeuralNetBaselineFixedSet(
        instances=instances,
        raw_baseline=True,
        train_with_unknown=True,
        make_balance=False,
        output_dir="approach/results/baseline/autopruner_programwise/finetune_fix/",
        use_trace=False,
        use_semantic=True, 
        use_static=False,
    )
    all_runners.append(runner)

    # 2. without raw baseline
    runner = NeuralNetBaselineFixedSet(
        instances=instances,
        raw_baseline=False,
        train_with_unknown=True,
        make_balance=False,
        output_dir="approach/results/baseline/autopruner_programwise/finetune_fix/",
        use_trace=False,
        use_semantic=True, 
        use_static=False,
    )
    all_runners.append(runner)

    # train on labeled data only
    for balance in balance__variations:
        # Convert the balance string into method and ratio if needed
        if "_" in balance:
            balance_method, balance_ratio = balance.split("_")
            make_balance = (balance_method, float(balance_ratio))
        else:
            make_balance = False

        runner = NeuralNetBaselineFixedSet(
            instances=instances,
            raw_baseline=False,
            train_with_unknown=False,
            make_balance=make_balance,
            output_dir="approach/results/baseline/autopruner_programwise/finetune_fix/",
            use_trace=False,
            use_semantic=True, 
            use_static=False,
        )
        all_runners.append(runner)

    # run all the runners
    for runner in all_runners:
        runner.run()


def run_ml4cgp():
    pass



def main(baseline, dataset, exp):

    if dataset == 'njr':

        instances = load_instances("njr")
        if baseline == "cgpruner":
            run_cgpruner(instances)
        elif baseline == "autopruner":
            run_autopruner(instances)
        elif baseline == "ml4cgp":
            run_ml4cgp(instances)
        else:
            raise ValueError("Invalid baseline specified")
    

    elif dataset == 'xcorp':

        if exp == '1':

            params = [
                ("doop",("v1", "39"),False),
                ("doop",("v1", "39"),True),
                ("doop",("v3", "5"),False),
                ("doop",("v3", "5"),True),
                ("doop",("v2", "0"),False),
                ("wala",("v1", "19"),False),
                ("wala",("v3", "0"),False),
                ("wala",("v1", "23"),False),
                ("opal",("v1", "0"),False),
            ]

            for param in params:
                instances = load_instances(tool=param[0], config_info=param[1], just_three=param[2])
                if baseline == "cgpruner":
                    run_cgpruner(instances)
                elif baseline == "autopruner":
                    run_autopruner(instances)
                elif baseline == "ml4cgp":
                    run_ml4cgp(instances)
                else:
                    raise ValueError("Invalid baseline specified")

        elif exp == '2':
            for tool in ['wala', 'doop']:
                instances = load_instances(tool=tool, config_info=None, just_three=True)
                if baseline == "cgpruner":
                    run_cgpruner(instances)
                elif baseline == "autopruner":
                    run_autopruner(instances)
                elif baseline == "ml4cgp":
                    run_ml4cgp(instances)
                else:
                    raise ValueError("Invalid baseline specified")

        elif exp == '3':
            instances = load_instances(tool=None, config_info=None, just_three=True)
        else:
            raise ValueError("Invalid experiment specified for xcorp dataset")
    else:
        raise ValueError("Invalid dataset specified")

    
    
    



if __name__ == "__main__":
    # add arguments to the script
    
    main(parser.parse_args().baseline)
