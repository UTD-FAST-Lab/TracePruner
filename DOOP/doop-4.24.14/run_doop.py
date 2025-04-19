import os

# doop_file = "/home/mohammad/projects/CallGraphPruner/DOOP/doop-4.24.14/doop"
programs_list = '/home/mohammad/projects/CallGraphPruner/data/programs/all_programs.txt'
data_dir = '/home/mohammad/projects/CallGraphPruner_data/njr-1_dataset/june2020_dataset'

analyses = ['context-insensitive', '1-call-site-sensitive']


def get_mainclass(program):
	mainclassname_path = os.path.join(data_dir, program, 'info', "mainclassname")
	with open(mainclassname_path, 'r') as mainfile:
		mainclass = mainfile.readline().strip()
	return mainclass

def main():
    with open(programs_list, 'r') as f:
        for i, line in enumerate(f):
            program = line.strip()
            print(f"Running DOOP on {program}")

            program_path = os.path.join(data_dir, program, 'jarfile', f'{program}.jar')
            mainclass = get_mainclass(program)

            command = f"./doop -a {analyses[1]} -i {program_path} --main {mainclass} --platform java_8  --id {program}_1cfa"
            os.system(command)
            print(f"DOOP finished for {program}")

            if i == 5:
                break


if __name__ == "__main__":
    main()