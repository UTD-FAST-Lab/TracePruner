#!/bin/bash

# CLI Args:
# $1 -> Path to the dataset

jar_file_path="target/ml4cg_sa-1.0-SNAPSHOT-shaded.jar"

#XCorpus
/usr/lib/jvm/java-1.11.0-openjdk-amd64/bin/java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "/20TB/mohammad/cg_dataset/static_cg/xcorpus2" -j "/20TB/mohammad/cg_dataset/ml4cgp_study_data/xcorpus/xcorpus_sel_programs2.txt" --confID 0 --handleZeroLengthArray --handleStaticInit
# java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/xcorpus/xcorpus_jars_w_deps" -j "$1/xcorpus/xcorpus_sel_programs2.txt" --confID $2 --handleZeroLengthArray --handleStaticInit
# java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/xcorpus/xcorpus_jars_w_deps" -j "$1/xcorpus/xcorpus_sel_programs.txt" -1cfa

#YCorpus
# java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/ycorpus/gh_projects_processed_w_deps_v7-5" -j "$1/ycorpus/ycorpus_sel_programs.txt" -0cfa
# java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/ycorpus/gh_projects_processed_w_deps_v7-5" -j "$1/ycorpus/ycorpus_sel_programs.txt" -1cfa

# #NJR-1
# java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/njr1/cgs/replicate/0cfa" -j "$1/njr1/cgs/njr1_programs_2.txt" 
# java -Xmx90g -cp $jar_file_path dev.c0pslab.analysis.CGGenRunner -o "$1/njr1/cgs/1cfa" -j "$1/njr1/cgs/njr1_programs.txt" -1cfa