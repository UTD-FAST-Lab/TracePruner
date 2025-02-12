# mohammad rafieian


import os
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

TMP_DIR = "tmp"

WALA_CORE_JAR = "tools/wala/com.ibm.wala.core-1.5.9.jar"
WALA_SHRIKE_JAR = "tools/wala/com.ibm.wala.shrike-1.5.9.jar"
WALA_UTIL_JAR = "tools/wala/com.ibm.wala.util-1.5.9.jar"

JCG_JAR = "/home/mohammad/projects/CallGraphPruner/cats/jcg_wala_testadapter/target/scala-2.12/JCG-WALA-Test-Adapter-assembly-1.0.jar"
TESTCASES_PATH = "/home/mohammad/projects/CallGraphPruner/cats/testcaseJars"

is_default = True
WALA_DEFAULT_CONFIG = '-reflectionSetting NONE -cgalgo ZERO_CFA -sensitivity 1 -handleStaticInit true -useConstantSpecificKeys false -handleZeroLengthArray true -useLexicalScopingForGlobals false -useStacksForLexcialScoping false'

agents = (
        '/home/mohammad/projects/CallGraphPruner/agents/branch-j8/target/branch-j8-1.0-SNAPSHOT-jar-with-dependencies.jar',
        '/home/mohammad/projects/CallGraphPruner/agents/call-graph-j8/target/call-graph-j8-1.0-SNAPSHOT-jar-with-dependencies.jar',
        '/home/mohammad/projects/CallGraphPruner/agents/variable-j8/target/variables-1.0-SNAPSHOT-jar-with-dependencies.jar',
        '/home/mohammad/projects/CallGraphPruner/agents/integrated-j8/target/integrated-j8-1.0-SNAPSHOT-jar-with-dependencies.jar',
        '/home/mohammad/projects/CallGraphPruner/agents/integrated-edge-seperate-j8/target/integrated-edge-seperate-j8-1.0-SNAPSHOT-jar-with-dependencies.jar' 
)


def get_mainclass(tc):
    
    config_path = os.path.join(TESTCASES_PATH, f'{tc}.conf')

    with open(config_path, 'r') as file:
        data = json.load(file)
        return data.get("main")


def run_wala(tc, mainclass):

    command = f'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -javaagent:{agents[4]}=logLevel=method,agentLevel=cg -jar {JCG_JAR} 0-CFA {tc} {mainclass} '
    command += f' > /home/mohammad/projects/CallGraphPruner/data/cgs/{tc}.txt'

    os.system(command)


def main():

    tc_names = [os.path.splitext(f)[0] for f in os.listdir(TESTCASES_PATH) if f.endswith('.jar') and os.path.isfile(os.path.join(TESTCASES_PATH, f))]

    for tc in tc_names:
        if 'VC4' in tc:
            mainclass = get_mainclass(tc)
            run_wala(tc, mainclass)



if __name__ == '__main__':
    #Create temporary directory if it doesn't exist
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    # #Add jars to classpath
    # class_path_string = f'{WALA_CORE_JAR}:{WALA_SHRIKE_JAR}:{WALA_UTIL_JAR}:.'
    # os.environ["CLASSPATH"] = class_path_string

    main()

