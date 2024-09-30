import os

DRIVER_PROGRAM = "WalaOriginal"
BENCHMARKS_FOLDER = "../../../dataset-high-precision-callgraphs/full_programs_set"
RESULTS_FOLDER = "1cfa_timings"
JAVAC_COMMAND = "javac"
JAVA_COMMAND = "java"
SKIP_COMPLETED = True #skips if the output file is already there.
FILE_WITH_MAIN_CLASS = "info/mainclassname"
CLASSPATH = "../../wala-npa/com.ibm.wala.shrike-1.5.5.jar:../../wala-npa/com.ibm.wala.core-1.5.5.jar:../../wala-npa/com.ibm.wala.util-1.5.5.jar:."
#add wala jars to classpath. Also add current folder
#class_path_string = f'{WALA_CORE_JAR}:{WALA_SHRIKE_JAR}:{WALA_UTIL_JAR}:$CLASSPATH'
#os.environ["CLASSPATH"] = class_path_string
#os.system("echo $CLASSPATH")    

#Loop through the benchmarks
for benchmark in os.listdir(BENCHMARKS_FOLDER):
    print(benchmark)
    if (SKIP_COMPLETED):
        if os.path.exists(os.path.join(RESULTS_FOLDER,(benchmark+".txt"))):
            print("skipping completed benchmark.")
            continue
    benchmark_path = os.path.join(BENCHMARKS_FOLDER,benchmark)
    #skip non-directories
    if not os.path.isdir(benchmark_path):
        continue
    #Get jar file
    jarfile = ''
    for file in os.listdir(os.path.join(benchmark_path,"jarfile")):
        if file.endswith(".jar"):
            jarfile = file
    jarfile_path = os.path.join(benchmark_path,("jarfile/" + jarfile))

    #get main class name
    mainclassname_file = os.path.join(benchmark_path,FILE_WITH_MAIN_CLASS)
    with open(mainclassname_file) as fp:
        mainclass_name = fp.read().splitlines()[0]

    #execute wala on the jar
    wala_command = (JAVA_COMMAND
        + " -cp " + CLASSPATH
        + " " + DRIVER_PROGRAM
        + " -classpath"
        + " " + jarfile_path
        + " -mainclass"
        + " " + mainclass_name
        + " > " + RESULTS_FOLDER
        + "/" + benchmark + ".txt"
    )
    os.system(wala_command)

