# CG-Pruner
This repository is the artifact accompanying the following paper:
<Add citation after submission, to respect double blind policy>

Since the CG-pruner is a machine-learning based tool, it involves two phases: the training phase and then the test phase. The following instructions detail steps to complete these phases as well as run the tool to produce the call-graph for an single program.

There are two ways to use the artifact:

Option1) (Recommended) Download the Virtualbox VM image, and import it using the "File->Import Appliance" option in VirtualBox. The commands from this Readme that are labeled as "full command on the VM", can be copy pasted as is into the Terminal on the VM. The username and password for the VM are both 'artifact'.
No further installation or downloads are needed for this option.

Option2) Manual downloads and installation.
Download the *code* and *data* folders.
Also download the NJR-1 benchmarks are located at (https://zenodo.org/record/4839913).
Download and install all the dependencies listed below, and then
follow the rest of the steps. 

## Downloading Dependencies

Skip this step if you are using the VirtualBox VM image, since all the dependencies are already installed.

Dependencies:
- Python3
- Java-8
- scikit-learn (tested with v 1.0.1)
- numpy (tested with v1.21.5)
- scipy (tested with v1.7.3)
- javaq (https://github.com/ucla-pls/javaq) (tested with commit 33e767761c6b8b6813cd01041ae2fb021fb0efd6)

## Train and generate a pruner
In this step we will train a cg-pruner. The training data is again available in the accompanying dataset. Run the following command (estimated-time: 6 minutes):

```
$ cd code/generate-pruner
$ python3 generate_cg_pruner.py <callgraphs-folder> <training-programs-list> <config-file> <learned-model-file>
```

- *callgraphs-folder:* Is a directory with one sub-directory per training program. Use the *full-callgraphs-set* folder from the dataset.
- *training-programs-list:* Is a list of the training-programs. Use *train_programs.txt* from the dataset.
- *config-file:* Use *configs/wala.config* for WALA. This lists some configuration information.
- *learned-model-file:* Is the name of the trained-model to be output. Choose any name, but use a .joblib extension.

For example, this would be the full command on the VM.
```
$ cd code/generate-pruner
$ python3 generate_cg_pruner.py ../../data/full_callgraphs_set ../../data/train_programs.txt ../configs/wala.config ../wala.joblib
```

(NOTE: The accompanying dataset already has some learned classifiers in the *learned_classifiers* folder. Use *wala.joblib* from there, in case you want to skip this step.)

## Evaluate on test-set
In this step we will evaluate the learned classifier on the test-programs.
The evaluation mainly uses the concepts of recall and precision, with
a dynamic analysis providing the ground-truth.
Kindly see the introduction and implementation sections of the paper for the definitions of these concepts.
Run the following command (estimated-time: 1-2 minutes):

```
$ cd code/final-experiments/main_result
$ python3 get_main_result.py <callgraphs-folder> <config-file> <learned-model-file> <test-programs-list>
```
- *callgraphs-folder:* Same as above
- *test-programs-list:* Is a list of the test-programs. Use *test_programs.txt* from the dataset.
- *config-file:* Same as above
- *learned-model-file:* Use the .joblib file created in the previous step if you generated your own cg-pruner. Otherwise use the *wala.joblib* file from the *learned_classifiers* folder in the dataset.

For example, this would be the full command on the VM.
```
$ cd code/final-experiments/main_result
$ python3 get_main_result.py ../../../data/full_callgraphs_set ../../configs/wala.config ../../wala.joblib ../../../data/test_programs.txt > result.csv
```

##### Understanding the output
Redirect the output text to a csv file, and open it using Excel, Google Sheets
or Mac Numbers for better viewing. The following results are output in text format:

1. *Per-program baseline:* This is the precision and recall scores for each of the individual programs in the test-set using the original tool (Wala in this case).
2. *Baseline:* This is the average precision, recall and f-measure over the test-set
3. *New per-program precision-recall at 0.45 threshold:* This lists the precision and recall scores for each individual program 
4. *Area under the curve:* This is the area under the precision-recall curve.
5. *Precision-recall curve points:* This shows the precision-recall trade-off points in the main result of the paper.
6. *Monomorphic calls:* This is the precision and recall scores for a monomorphic callsites client before and after using CG-pruner.

Additionally, two folders are generated. We are mainly interested in the *walaCgPrunerCallgraphs* which has the pruned call-graphs.

##### Running null-pointer analysis on the pruned call-graphs (optional, estimated-time:45 minutes)
The *wala-npa* folder has a null-pointer analysis. We can run this analysis with and without the CG-pruner to see the difference in the null-warnings produced.
But first you will have to add the Wala jars in *wala-npa* to your classpath and compile
the null-pointer analysis using *javac nullpointeranalysis/\*.java*

For example, this would be the full command on the VM.
```
$ cd code/wala-npa
$ export CLASSPATH=$CLASSPATH:com.ibm.wala.core-1.5.5.jar:com.ibm.wala.shrike-1.5.5.jar:com.ibm.wala.util-1.5.5.jar
$ javac nullpointeranalysis/*.java
```

Then we can run null pointer analysis either with or without the call-graph pruner .

```
$ cd final-experiments/null_pointer_experiment
$ python3 runNPAonTestSet.py <projects-folder> <results-folder> <test-programs-list> <pruned-callgraphs>
```

- *projects-folder:* Use the *full_programs_set* folder from the dataset.
- *results-folder:* Is an empty folder to store the results.
- *test-programs-list:* Is a list of the test-programs. Use *test_programs.txt* from the dataset.
- *pruned-callgraphs:* Is the folder with the pruned-callgraphs. If you want to use the original WALA callgraph, leave this option blank. Else, use the pruned-callgraphs generated in the previous step.

For example, this would be the full command on the VM.
```
$ cd code
$ mkdir npa_original_results
$ mkdir npa_pruned_results
$ cd final-experiments/null_pointer_experiment
$ python3 runNPAonTestSet.py ../../../njr-1_dataset/june2020_dataset ../../npa_original_results ../../../data/test_programs.txt
$ python3 runNPAonTestSet.py ../../../njr-1_dataset/june2020_dataset ../../npa_pruned_results ../../../data/test_programs.txt ../main_result/walaCgPrunerCallgraphs
```

Compare the number of null-pointer errors reported with and without the call-graph pruner.
The one with the pruned call-graph will have fewer errors.

## Evaluate on a single-program (optional)
Instead of running on the entire test-set, you may want to run the tool on
a single program to generate its pruned call-graph.
To do this, first run WALA to generate the original call-graph

```
$ cd code/run-on-single-program
$ python3 run_wala.py --input <jarfile> --output <output-callgraph> --main <main-class>
```
- *jarfile:* The jarfile to be analyzed. The *sample-program* folder has a jarfile which can be used.
- *output-callgraph:* The name of the file with the output callgraph.
- *main-class:* The name of the class with the main function. For the *sample-program*, this is located in the *info/mainclassname* file.

For example, this would be the full command on the VM.
```
$ cd code/run-on-single-program
$ python3 run_wala.py --input sample-program/jarfile/jar1.jar --output wala_cg.csv --main praxis.PraxisController
```

The next step is to run the CG-pruner on this output callgraph

```
$ python3 run_cg_pruner.py --output <output-callgraph> --input <input-callgraph> --classifier <learned-model-file> --cutoff <cutoff-value>
```
- *output-callgraph:* The name of the file with the output callgraph.
- *input:* The name of the file with the input callgraph. This is the Wala call-graph generated in the previous step.
- *learned-model-file:* Use the .joblib file created in the previous step if you generated your own cg-pruner. Otherwise use the *wala.joblib* file from the *learned_classifiers* folder in the dataset.
- *cutoff-value:* Kindly see the introduction of the paper to understand what
this means. A value of 0.6 for the sample program will show a decrease in the total number of edges due to pruning.

For example, this would be the full command on the VM.
```
$ cd code/run-on-single-program
$ python3 run_cg_pruner.py --output wala_pruned_cg.csv --input wala_cg.csv --classifier ../wala.joblib --cutoff 0.45
```

## Understanding the Directory Structure of the code

1) callgraph-feature-generation: Scripts used to compute features over the call-graphs.
2) common: contains utility files used by some of the scripts
3) config: contains one configuration file per tool that was experimented with. Used by some scripts
4) final-experiments: scripts to run all the experiments used in the paper.
5) generate-pruner: code to generate a pruner. Outputs a joblib file.
6) run-on-single-program: scripts to run the tool on a single Jar file, instead of running the experiments.
7) stats: Some statistics computed on the benchmark set
8) wala-npa: A null-pointer analysis implemented in Wala