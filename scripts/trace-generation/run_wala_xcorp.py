# mohammad rafieian


import os
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading, subprocess
import argparse




WALA_DRIVER = "/home/mohammad/projects/TracePruner/scripts/trace-generation/driver/wala-project_scg/target/wala-project_scg-1.0-SNAPSHOT-jar-with-dependencies.jar"


jar_files = {
	'axion':'/20TB/mohammad/xcorpus-total-recall/jarfiles/axion/final.jar',
	'batik':'/20TB/mohammad/xcorpus-total-recall/jarfiles/batik/final.jar',
	'jasml':'/20TB/mohammad/xcorpus-total-recall/jarfiles/jasml/final.jar',
	'xerces':'/20TB/mohammad/xcorpus-total-recall/jarfiles/xerces/final.jar',
}



def run_wala():
	
	
	mainclass = 'Entrypoint'

	for key in jar_files:
		jar_file = jar_files[key]
		output_file = f'/20TB/mohammad/xcorpus-total-recall/static_cgs/{key}/wala0cfa_none.csv'

		command = f'/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -jar {WALA_DRIVER} -classpath {jar_file} -mainclass {mainclass} -output {output_file} -resolveinterfaces true -reflection false -analysis 0cfa '
	
	
		os.system(command)



def main():
    run_wala()

	

if __name__ == '__main__':

	main()
