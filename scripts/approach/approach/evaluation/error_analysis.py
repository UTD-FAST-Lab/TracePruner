import os
import pandas as pd
from collections import defaultdict


def generate_files():
    # dc1
    # inst_path = '/20TB/mohammad/xcorpus-total-recall/results/baseline/cgpruner/doop/v1_39_True/rf_programwise_0.45_trained_on_known_oversample_1.0.pkl'
    # inst_path = '/20TB/mohammad/xcorpus-total-recall/results/baseline/finetuned/doop/v1_39_True/codebert_programwise_0.5_trained_on_known.pkl'

    # oc1
    # inst_path = '/20TB/mohammad/xcorpus-total-recall/results/baseline/cgpruner/opal/v1_0_True/rf_programwise_0.45_trained_on_known_oversample_1.0.pkl'
    inst_path = '/20TB/mohammad/xcorpus-total-recall/results/baseline/finetuned/opal/v1_0_True/codebert_programwise_0.5_trained_on_known.pkl'



    # out_dir = '/20TB/mohammad/xcorpus-total-recall/error_analysis/OC1/rf'
    out_dir = '/20TB/mohammad/xcorpus-total-recall/error_analysis/OC1/bert'

    instances = pd.read_pickle(inst_path)

    program_map = defaultdict(list)
    for inst in instances:
        program_map[inst.program].append(inst)


    for program, insts in program_map.items():
        fns = []
        fps = []
        tps = []
        tns = []
        for inst in insts:
            if inst.is_known():
                if inst.get_label() == 1 and inst.get_predicted_label() == 0:
                    fns.append(inst)
                elif inst.get_label() == 0 and inst.get_predicted_label() == 1:
                    fps.append(inst)
                elif inst.get_label() == 1 and inst.get_predicted_label() == 1:
                    tps.append(inst)
                elif inst.get_label() == 0 and inst.get_predicted_label() == 0:
                    tns.append(inst)

        # write to a file for each program
        p_out_path = os.path.join(out_dir, program)
        os.makedirs(p_out_path, exist_ok=True)
        # make 4 different dfs for fns, fps, tps, tns - inst: src, offset, target, get_static_featuers - sort by confidence and then save to csv
        fns.sort(key=lambda x: x.get_confidence(), reverse=False)
        fps.sort(key=lambda x: x.get_confidence(), reverse=True)
        tps.sort(key=lambda x: x.get_confidence(), reverse=True)
        tns.sort(key=lambda x: x.get_confidence(), reverse=False)
        fns_df = pd.DataFrame([{
            'src': inst.src,
            'offset': inst.offset,
            'target': inst.target,
            # 'static_features': inst.get_static_featuers(),
            'confidence': inst.get_confidence()
        } for inst in fns])
        fps_df = pd.DataFrame([{
            'src': inst.src,
            'offset': inst.offset,
            'target': inst.target,
            # 'static_features': inst.get_static_featuers(),
            'confidence': inst.get_confidence()
        } for inst in fps])
        tps_df = pd.DataFrame([{
            'src': inst.src,
            'offset': inst.offset,
            'target': inst.target,
            # 'static_features': inst.get_static_featuers(),
            'confidence': inst.get_confidence()
        } for inst in tps])
        tns_df = pd.DataFrame([{
            'src': inst.src,
            'offset': inst.offset,
            'target': inst.target,
            # 'static_features': inst.get_static_featuers(),
            'confidence': inst.get_confidence()
        } for inst in tns])
        fns_df.to_csv(os.path.join(p_out_path, 'fns.csv'), index=False)
        fps_df.to_csv(os.path.join(p_out_path, 'fps.csv'), index=False)
        tps_df.to_csv(os.path.join(p_out_path, 'tps.csv'), index=False)
        tns_df.to_csv(os.path.join(p_out_path, 'tns.csv'), index=False)


import pandas as pd
import os
from matplotlib_venn import venn2
import matplotlib.pyplot as plt

MODELS = ["rf", "bert"]
PROGRAMS = ["axion", "batik", "xerces", "jasml"]
TYPES = ["tps", "fps", "fns", "tns"]
BASE_DIR = "/20TB/mohammad/xcorpus-total-recall/error_analysis/OC1"  # adjust if needed


def load_results():
    all_data = {}
    for model in MODELS:
        all_data[model] = {}
        for program in PROGRAMS:
            prog_path = os.path.join(BASE_DIR, model, program)
            all_data[model][program] = {}
            for typ in TYPES:
                file_path = os.path.join(prog_path, f"{typ}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df["type"] = typ
                    df["program"] = program
                    df["model"] = model
                    all_data[model][program][typ] = df
                else:
                    print(f"[!] Missing: {file_path}")
                    all_data[model][program][typ] = pd.DataFrame(columns=["src", "offset", "target", "confidence", "type", "program", "model"])
    return all_data


def compare_errors(all_data, output_dir="analysis"):
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    for program in PROGRAMS:
        for err_type in ["fps", "fns"]:
            df_rf = all_data["rf"][program][err_type]
            df_bert = all_data["bert"][program][err_type]

            set_rf = set(tuple(row[:3]) for row in df_rf[["src", "offset", "target"]].values)
            set_bert = set(tuple(row[:3]) for row in df_bert[["src", "offset", "target"]].values)

            only_rf = set_rf - set_bert
            only_bert = set_bert - set_rf
            both = set_rf & set_bert

            summary_rows.append({
                "program": program,
                "error_type": err_type,
                "rf_count": len(set_rf),
                "bert_count": len(set_bert),
                "both": len(both),
                "only_rf": len(only_rf),
                "only_bert": len(only_bert)
            })

            # Save comparison CSV
            pd.DataFrame(list(only_rf), columns=["src", "offset", "target"]).to_csv(f"{output_dir}/{program}_{err_type}_only_rf.csv", index=False)
            pd.DataFrame(list(only_bert), columns=["src", "offset", "target"]).to_csv(f"{output_dir}/{program}_{err_type}_only_bert.csv", index=False)
            pd.DataFrame(list(both), columns=["src", "offset", "target"]).to_csv(f"{output_dir}/{program}_{err_type}_both.csv", index=False)

            # Venn Diagram
            plt.figure()
            venn2([set_rf, set_bert], set_labels=("RF", "BERT"))
            plt.title(f"{program.upper()} - {err_type.upper()} Comparison")
            plt.savefig(f"{output_dir}/{program}_{err_type}_venn.png")
            plt.close()

    # Save summary
    pd.DataFrame(summary_rows).to_csv(f"{output_dir}/error_summary.csv", index=False)
    print(f"✅ Analysis complete. Summary and plots saved in '{output_dir}/'")


def plot_confidence_distribution():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Setup
    root = '/20TB/mohammad/xcorpus-total-recall/error_analysis/OC1'
    models = ['rf', 'bert']
    error_types = ['fps', 'fns']

    # Collect confidence scores
    confidence_data = []

    for model in models:
        model_path = os.path.join(root, model)
        for program in os.listdir(model_path):
            prog_path = os.path.join(model_path, program)
            if not os.path.isdir(prog_path):
                continue
            for err_type in error_types:
                path = os.path.join(prog_path, f'{err_type}.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    if 'confidence' not in df.columns:
                        raise ValueError(f"Missing 'confidence' column in {path}")
                    for _, row in df.iterrows():
                        confidence_data.append({
                            'model': model.upper(),
                            'program': program,
                            'error_type': 'False Positive' if err_type == 'fps' else 'False Negative',
                            'confidence': row['confidence']
                        })

    # Convert to DataFrame
    df_conf = pd.DataFrame(confidence_data)

    # # Plot
    # plt.figure(figsize=(8, 5))
    # sns.violinplot(data=df_conf, x='error_type', y='confidence', hue='model', split=True, inner="quart", palette='Set2')
    # # plt.title("Confidence Distribution on False Predictions")
    # plt.xlabel("Error Type")
    # plt.ylabel("Confidence Score")
    # plt.legend(title='Model')
    # plt.grid(True)
    # plt.tight_layout()
    plt.figure(figsize=(5, 4)) 

    sns.violinplot(data=df_conf, x='error_type', y='confidence', hue='model', 
                split=True, inner="quart", palette='Set2')

    # Add font sizes to labels and title
    # plt.title("Confidence Distribution on False Predictions", fontsize=16)
    plt.xlabel("Error Type", fontsize=16)
    plt.ylabel("Confidence Score", fontsize=16)

    # Increase font size for the axis ticks (the numbers)
    plt.tick_params(axis='both', which='major', labelsize=13)

    # Increase font size for the legend
    plt.legend(title='Model', title_fontsize='14', fontsize='13')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(root, 'confidence_distribution.png'))



if __name__ == "__main__":
    # generate_files()
    
    # all_data = load_results()
    # compare_errors(all_data, output_dir='/20TB/mohammad/xcorpus-total-recall/error_analysis/OC1/analysis')

    plot_confidence_distribution()