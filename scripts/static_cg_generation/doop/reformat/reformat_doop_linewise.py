
import csv
import re
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess


def java_type_to_descriptor(jtype):
    jtype = jtype.strip()
    # Array types
    if jtype.endswith("[]"):
        return "[" + java_type_to_descriptor(jtype[:-2])
    # Primitives
    primitives = {
        "void": "V", "int": "I", "float": "F", "double": "D",
        "long": "J", "boolean": "Z", "char": "C", "short": "S", "byte": "B"
    }
    if jtype in primitives:
        return primitives[jtype]
    # Object types
    return f"L{jtype.replace('.', '/')};"


def format_signature_wala(raw_sig):
    """
    Convert from:
      java.io.File: java.lang.String[] list()
    To:
      java/io/File.list:()[Ljava/lang/String;
    """
    if ':' not in raw_sig:
        return raw_sig

    class_part, rest = raw_sig.split(':', 1)
    class_part = class_part.strip().replace('.', '/')

    match = re.match(r'\s*(\S+)\s+([<>a-zA-Z0-9_$]+)\((.*)\)', rest.strip())
    if not match:
        print(f"Warning: Unable to parse signature: {raw_sig}")
        return raw_sig  # fallback

    return_type, method_name, param_str = match.groups()
    return_type = java_type_to_descriptor(return_type)

    params = param_str.split(',') if param_str.strip() else []
    param_types = ''.join([java_type_to_descriptor(p.strip()) for p in params])

    return f"{class_part}.{method_name}:({param_types}){return_type}"


def parse_callgraph_line(line):
    parts = line.strip().split('\t')
    if len(parts) != 4:
        return None

    caller_field = parts[1]  # e.g. <class: sig>/some.site/42
    callee_field = parts[3]  # e.g. <class: sig>

    # Extract the method part and the offset
    if '/' not in caller_field:
        return None

    callparts = caller_field.split('/')
    if len(callparts) < 2:
        return None
    callsite = callparts[1]

    sig_part, offset = caller_field.rsplit('/', 1)

    # Find the real closing '>' of the signature
    paren_close = sig_part.find(')')
    if paren_close == -1:
        return None
    gt_index = sig_part.find('>', paren_close)
    if gt_index == -1:
        return None

    method_sig = sig_part[1:gt_index]  # remove angle brackets
    target_sig = callee_field.strip('<>')

    method_sig = format_signature_wala(method_sig)
    target_sig = format_signature_wala(target_sig)

    return method_sig, callsite, offset, target_sig


def calculate_pc(method, callsite, offset, program_dir, line_number):
    '''run the jar file that calculates the pc'''

    program_name = program_dir.split('_')[0]

    classandname, signature = method.split(':')
    classname, methodname = classandname.split('.')

    extractor_jar_path = '/home/mohammad/projects/TracePruner/scripts/static_cg_generation/doop/reformat/DoopRunnerWithPC/target/DoopRunnerWithPC-1.0-SNAPSHOT-jar-with-dependencies.jar'

    inputJar = f'/20TB/mohammad/xcorpus-total-recall/jarfiles/{program_name}/final.jar'
    callerMethod = methodname
    callerClass = classname
    callerSignature = signature
    jdkPath = '/usr/lib/jvm/java-1.8.0-openjdk-amd64/jre/lib/rt.jar'
    declaredTgt = callsite
    number = offset
    output_dir = f'/home/mohammad/projects/TracePruner/scripts/static_cg_generation/doop/reformat/pcs/{program_dir}_{line_number}.txt'
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    args = [
        "java", "-jar", extractor_jar_path,
        inputJar,
        callerMethod,
        callerClass,
        callerSignature,
        jdkPath,
        declaredTgt,
        number,
        output_dir
    ]

    try:
        # print(f"[START] Running subprocess for line {program_dir} - {line_number}")
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # print(f"[END] Subprocess for line {program_dir}-{line_number} completed successfully")

        with open(output_dir, 'r') as f:
            lines = f.readlines()
        
        pc = lines[0]
        pc = pc.strip()

        # delete the output file after reading
        os.remove(output_dir)

        return pc

    except subprocess.CalledProcessError:
        # print(f"DOOP failed for config for program {program_dir} at line {line_number}")
        return None


def process_program(input_programs_dir, output_dir, program_dir):
    program_input_path = os.path.join(input_programs_dir, program_dir)
    input_file = os.path.join(program_input_path, 'CallGraphEdge.csv')
    output_file_dir = os.path.join(output_dir, program_dir)
    os.makedirs(output_file_dir, exist_ok=True)
    output_file = os.path.join(output_file_dir, 'CallGraphEdge.csv')

    print(f"[+] Processing {input_file}...")

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping...")
        return

    with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['method', 'offset', 'target'])

        for line_number, line in enumerate(fin):
            parsed = parse_callgraph_line(line)
            if parsed:
                method, callsite, offset, target = parsed
                if method.startswith('register-finalize'):
                    continue
                # if offset is not a valid integer, skip the line
                if not offset.isdigit():
                    # print(f"Skipping line with invalid offset")
                    continue
                
                pc = calculate_pc(method, callsite, offset, program_dir, line_number)
                if pc is None:
                    # print(f"Failed to calculate PC for {method} at line {line_number} in {program_dir}")
                    continue
                writer.writerow([method, pc, target])
                fout.flush()

    print(f"[✓] WALA-style call graph written to {output_file}")


def main():
    config_version = 'v3'
    doop_data_dir = f'/20TB/mohammad/xcorpus-total-recall/static_cgs/doop/{config_version}'
    input_programs_dir = f'{doop_data_dir}/out'
    output_dir = os.path.join(doop_data_dir, 'reformatted-v2')
    os.makedirs(output_dir, exist_ok=True)

    programs = [d for d in os.listdir(input_programs_dir) if os.path.isdir(os.path.join(input_programs_dir, d))]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for program_dir in programs:
            futures.append(executor.submit(process_program, input_programs_dir, output_dir, program_dir))

        for future in futures:
            future.result()


if __name__ == "__main__":
    main()