# baseline runners

import json
import os
from itertools import product

from approach.runners.rf_baseline import RandomForestBaseline
from approach.runners.rf_baseline_fixed_set import RandomForestBaselineFixedSet
from approach.runners.nn_baseline import NeuralNetBaseline
from approach.runners.nn_baseline_finetuned import NeuralNetBaselineFineTuned
from approach.runners.nn_baseline_fixed_set import NeuralNetBaselineFixedSet
from approach.runners.svm_runner import SVMBaseline
# from approach.data_representation.instance_loader import load_instances
from approach.data_representation.instance_loader_xcorp import load_instances    #TODO: fix this mess
from approach.utils import plot_points
from approach.constants import base

import argparse
parser = argparse.ArgumentParser(description="Run baseline models")
parser.add_argument("--baseline", type=str, default="cgpruner", help="Baseline model to run (cgpruner, autopruner, ml4cgp)")
parser.add_argument("--dataset", type=str, default="xcorp", help="Dataset to use (njr, other)")
parser.add_argument("--exp", type=str, default="1", help="Type of experiment to run (1,2,3)")


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


def run_cgpruner(instances, param=None):

    if param[0] is None:
        output_dir = f"{base}/baseline/cgpruner"
    elif param[1] is None:
        output_dir = f'{base}/baseline/cgpruner/{param[0]}'
    else:
        output_dir = f'{base}/baseline/cgpruner/{param[0]}/{param[1][0]}_{param[1][1]}_{str(param[2])}'
    os.makedirs(output_dir, exist_ok=True)

    config = load_json("approach/runners/config.json")["cgPruner"]
    balance__variations = get_balance_combinations(config)

    all_runners = []

    # 1. with raw baseline
    # runner = RandomForestBaselineFixedSet(
    # runner = RandomForestBaseline(
    #     instances=instances,
    #     raw_baseline=True,
    #     train_with_unknown=True,
    #     threshold=0.45,
    #     make_balance=False ,
    #     # output_dir="{base}/baseline/cgpruner_programwise/fix"
    #     output_dir=output_dir,
    #     all_three=param[2]
    # )
    # all_runners.append(runner)

    # 2. without raw baseline
    # runner = RandomForestBaselineFixedSet(
    # runner = RandomForestBaseline(
    #     instances=instances,
    #     raw_baseline=False,
    #     train_with_unknown=True,
    #     threshold=0.45,
    #     make_balance=False ,
    #     # output_dir="{base}/baseline/cgpruner_programwise/fix"
    #     output_dir=output_dir,
    #     all_three=param[2]
    # )

    # all_runners.append(runner)
    
    # train on labeled data only
    for balance in balance__variations:
        # Convert the balance string into method and ratio if needed
        if "_" in balance:
            balance_method, balance_ratio = balance.split("_")
            make_balance = (balance_method, float(balance_ratio))
        else:
            make_balance = False
        
        # runner = RandomForestBaselineFixedSet(
        for i in range(2):
            runner = RandomForestBaseline(
                instances=instances,
                raw_baseline=False,
                train_with_unknown=False,
                threshold=0.45,
                make_balance=make_balance,
                # output_dir="{base}/baseline/cgpruner_programwise/fix"
                output_dir=output_dir,
                all_three=param[2],
                random_split=bool(i)
            )

            all_runners.append(runner)


    # run all the runners
    for runner in all_runners:
        runner.run()




def run_autopruner(param=None):

    if param[0] is None:
        output_dir = f"{base}/baseline/autopruner"
    elif param[1] is None:
        output_dir = f'{base}/baseline/autopruner/{param[0]}'
    else:
        output_dir = f'{base}/baseline/autopruner/{param[0]}/{param[1][0]}_{param[1][1]}_{str(param[2])}'
    os.makedirs(output_dir, exist_ok=True)


    config = load_json("approach/runners/config.json")["autoPruner"]
    balance__variations = get_balance_combinations(config)

    # codebert experiments
    instances = load_instances(tool=param[0], config_info=param[1], just_three=param[2], load_semantic_features=True, model_name='codebert')
    all_runners = []
    # 1. with raw baseline
    # runner = NeuralNetBaselineFixedSet(
    # runner = NeuralNetBaseline(
    #     instances=instances,
    #     raw_baseline=True,
    #     train_with_unknown=True,
    #     make_balance=False,
    #     # output_dir="{base}/baseline/autopruner_programwise/finetune_fix/",
    #     output_dir=output_dir,
    #     use_trace=False,
    #     use_semantic=True, 
    #     use_static=False,
    #     just_three=param[2]
    # )
    # all_runners.append(runner)

    # 2. without raw baseline
    # runner = NeuralNetBaselineFixedSet(
    # runner = NeuralNetBaseline(
    #     instances=instances,
    #     raw_baseline=False,
    #     train_with_unknown=True,
    #     make_balance=False,
    #     # output_dir="{base}/baseline/autopruner_programwise/finetune_fix/",
    #     output_dir=output_dir,
    #     use_trace=False,
    #     use_semantic=True, 
    #     use_static=False,
    #     just_three=param[2]
    # )
    # all_runners.append(runner)

    # train on labeled data only
    for balance in balance__variations:
        # Convert the balance string into method and ratio if needed
        if "_" in balance:
            balance_method, balance_ratio = balance.split("_")
            make_balance = (balance_method, float(balance_ratio))
        else:
            make_balance = False

        # runner = NeuralNetBaselineFixedSet(
        runner = NeuralNetBaseline(
            instances=instances,
            raw_baseline=False,
            train_with_unknown=False,
            make_balance=make_balance,
            # output_dir="{base}/baseline/autopruner_programwise/finetune_fix/",
            output_dir=output_dir,
            use_trace=False,
            use_semantic=True, 
            use_static=False,
            just_three=param[2],
            model_name='codebert',
            random_split=False
        )
        all_runners.append(runner)

    # run all the runners
    for runner in all_runners:
        runner.run()


    # codet5 experiments
    instances = load_instances(tool=param[0], config_info=param[1], just_three=param[2], load_semantic_features=True, model_name='codet5')
    all_runners = []

    for balance in balance__variations:
        # Convert the balance string into method and ratio if needed
        if "_" in balance:
            balance_method, balance_ratio = balance.split("_")
            make_balance = (balance_method, float(balance_ratio))
        else:
            make_balance = False

        # runner = NeuralNetBaselineFixedSet(
        runner = NeuralNetBaseline(
            instances=instances,
            raw_baseline=False,
            train_with_unknown=False,
            make_balance=make_balance,
            output_dir=output_dir,
            use_trace=False,
            use_semantic=True, 
            use_static=False,
            just_three=param[2],
            model_name='codet5',
            random_split=False
        )
        all_runners.append(runner)

    # run all the runners
    for runner in all_runners:
        runner.run()


def run_finetuned(param=None):
    
    if param[0] is None:
        output_dir = f"{base}/baseline/finetuned"
    elif param[1] is None:
        output_dir = f'{base}/baseline/finetuned/{param[0]}'
    else:
        output_dir = f'{base}/baseline/finetuned/{param[0]}/{param[1][0]}_{param[1][1]}_{str(param[2])}'
    os.makedirs(output_dir, exist_ok=True)

    config = load_json("approach/runners/config.json")["autoPruner"]
    balance__variations = get_balance_combinations(config)
    # codebert experiments
    instances = load_instances(tool=param[0], config_info=param[1], just_three=param[2], load_tokens=True)
    all_runners = []
    # train on labeled data only
    for balance in balance__variations:
        # Convert the balance string into method and ratio if needed
        if "_" in balance:
            balance_method, balance_ratio = balance.split("_")
            make_balance = (balance_method, float(balance_ratio))
        else:
            make_balance = False

        runner = NeuralNetBaselineFineTuned(
            instances=instances,
            raw_baseline=False,
            train_with_unknown=False,
            make_balance=make_balance,
            output_dir=output_dir,
            use_trace=False,
            use_semantic=True, 
            use_static=False,
            just_three=param[2],
            model_name='codebert',
            random_split=False
        )
        all_runners.append(runner)

    # run all the runners
    for runner in all_runners:
        runner.run()

    # codet5 experiments
    instances = load_instances(tool=param[0], config_info=param[1], just_three=param[2], load_tokens=True)
    all_runners = []
    # train on labeled data only
    for balance in balance__variations:
        # Convert the balance string into method and ratio if needed
        if "_" in balance:
            balance_method, balance_ratio = balance.split("_")
            make_balance = (balance_method, float(balance_ratio))
        else:
            make_balance = False

        runner = NeuralNetBaselineFineTuned(
            instances=instances,
            raw_baseline=False,
            train_with_unknown=False,
            make_balance=make_balance,
            output_dir=output_dir,
            use_trace=False,
            use_semantic=True, 
            use_static=False,
            just_three=param[2],
            model_name='codet5',
            random_split=False
        )
        all_runners.append(runner)

    # run all the runners
    for runner in all_runners:
        runner.run()
    


def run_svm(param=None):

    if param[0] is None:
        output_dir = f"{base}/baseline/svm"
    elif param[1] is None:
        output_dir = f'{base}/baseline/svm/{param[0]}'
    else:
        output_dir = f'{base}/baseline/svm/{param[0]}/{param[1][0]}_{param[1][1]}_{str(param[2])}'
    os.makedirs(output_dir, exist_ok=True)

    # semantic codebert
    instances = load_instances(tool=param[0], config_info=param[1], just_three=param[2], load_semantic_features=True, model_name='codebert')
    svm_runner = SVMBaseline(instances, output_dir, kernel="rbf", nu=0.1, gamma='scale', just_three=param[2], use_semantic=True, model_name='codebert', random_split=False)
    svm_runner.run()

    # semantic codet5
    instances = load_instances(tool=param[0], config_info=param[1], just_three=param[2], load_semantic_features=True, model_name='codet5')
    svm_runner = SVMBaseline(instances, output_dir, kernel="rbf", nu=0.1, gamma='scale', just_three=param[2], use_semantic=True, model_name='codet5', random_split=False)
    svm_runner.run()

    # structured
    instances = load_instances(tool=param[0], config_info=param[1], just_three=param[2])
    svm_runner = SVMBaseline(instances, output_dir, kernel="rbf", nu=0.1, gamma='scale', just_three=param[2], use_semantic=False, random_split=False)
    svm_runner.run()


def main(args):

    if args.dataset == 'njr':

        instances = load_instances("njr")
        if args.baseline == "cgpruner":
            run_cgpruner(instances)
        elif args.baseline == "autopruner":
            run_autopruner(instances)
        elif args.baseline == "finetuned":
            run_finetuned(instances)
        else:
            raise ValueError("Invalid baseline specified")
    

    elif args.dataset == 'xcorp':

        if args.exp == '1':

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
                if args.baseline == "cgpruner":
                    instances = load_instances(tool=param[0], config_info=param[1], just_three=param[2])
                    run_cgpruner(instances, param)
                elif args.baseline == "autopruner":
                    # instances = load_instances(tool=param[0], config_info=param[1], just_three=param[2], load_semantic_features=True, model_name='codebert')
                    # run_autopruner(instances, param)
                    run_autopruner(param)
                elif args.baseline == "finetuned":
                    run_finetuned(param)
                elif args.baseline == "svm":
                    run_svm(param)
                else:
                    raise ValueError("Invalid baseline specified")

        elif args.exp == '2':
            for tool in ['wala', 'doop']:
                if args.baseline == "cgpruner":
                    instances = load_instances(tool=tool, config_info=None, just_three=False)
                    run_cgpruner(instances, param=(tool, None, False))
                elif args.baseline == "autopruner":
                    # instances = load_instances(tool=tool, config_info=None, just_three=False)
                    # run_autopruner(instances, param=(tool, None, False))
                    run_autopruner(param=(tool, None, False))
                elif args.baseline == "finetuned":
                    run_finetuned(param=(tool, None, False))
                elif args.baseline == "svm":
                    run_svm(param=(tool, None, False))
                else:
                    raise ValueError("Invalid baseline specified")

        elif args.exp == '3':
            if args.baseline == "cgpruner":
                instances = load_instances(tool=None, config_info=None, just_three=True)
                run_cgpruner(instances, param=(None, None, True))
            elif args.baseline == "autopruner":
                # run_autopruner(instances, param=(None, None, True))
                run_autopruner(param=(None, None, True))
            elif args.baseline == "finetuned":
                run_finetuned(param=(None, None, True))
            elif args.baseline == "svm":
                run_svm(param=(None, None, True))
            else:
                raise ValueError("Invalid baseline specified")
        else:
            raise ValueError("Invalid experiment specified for xcorp dataset")
    else:
        raise ValueError("Invalid dataset specified")

    
    
    



if __name__ == "__main__":
    # add arguments to the script
    
    main(parser.parse_args())
