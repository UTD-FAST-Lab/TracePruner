# mohammad rafieian


import os
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, subprocess
import argparse




WALA_DRIVER = "/home/mohammad/projects/TracePruner/scripts/trace-generation/driver/wala-project_scg/target/wala-project_scg-1.0-SNAPSHOT-jar-with-dependencies.jar"






def run_wala():
	
		
	mainclass = 'vc.Class'
	jar_file = '/home/mohammad/projects/TracePruner/scripts/trace-generation/VC4.jar'
	output_file = f'/home/mohammad/projects/TracePruner/scripts/trace-generation/jcg_1obj.csv'


	command = f'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -jar {WALA_DRIVER} -classpath {jar_file} -mainclass {mainclass} -output {output_file} -resolveinterfaces true -reflection false -analysis 1obj '
	

	os.system(command)



def main():
	
    run_wala()

	

if __name__ == '__main__':

	main()
