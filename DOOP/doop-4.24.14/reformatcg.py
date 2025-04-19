import csv
import re
import os

# === JVM stdlib filters ===
stdlib_patterns = [
    r"java/awt/.*", r"javax/swing/.*", r"sun/awt/.*", r"sun/swing/.*",
    r"com/sun/.*", r"sun/.*", r"org/netbeans/.*", r"org/openide/.*",
    r"com/ibm/crypto/.*", r"com/ibm/security/.*", r"org/apache/xerces/.*",
    r"dalvik/.*", r"java/io/ObjectStreamClass.*", r"apple/.*", r"com/apple/.*",
    r"jdk/.*", r"org/omg/.*", r"org/w3c/.*", r"java/util/.*", r"java/io/.*",
    r"java/nio/.*", r"java/net/.*", r"java/math/.*", r"java/text/.*",
    r"java/sql/.*", r"java/security/.*", r"java/time/.*", r"javax/.*",
    r"org/graalvm/.*", r"java/.*"
]
stdlib_regex = re.compile(r"^(?:" + "|".join(stdlib_patterns) + r")")



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

    method_field = parts[1]  # e.g., <java.io.File: java.lang.String[] list()>/java.io.FileSystem.list/0

    # Step 1: find the closing ')' that ends the method param list
    paren_close = method_field.find(')')
    if paren_close == -1:
        return None

    # Step 2: find the first '>' AFTER the closing ')'
    gt_index = method_field.find('>', paren_close)
    if gt_index == -1:
        return None

    method_sig = method_field[1:gt_index]  # remove leading '<', slice up to '>'
    
    # Step 2: Extract offset from the last '/'
    offset_match = method_field.rsplit('/', 1)
    if len(offset_match) < 2:
        return None
    offset = offset_match[1]

    # Step 3: Extract target method signature
    target_sig = parts[3].strip('<>')

    # Step 4: Format the method signature
    method_sig = format_signature_wala(method_sig)
    target_sig = format_signature_wala(target_sig)

    return method_sig, offset, target_sig





def main():

    analysis_type = 'context-insensitive'  # or '1-call-site-sensitive'

    for program_dir in os.listdir('results'):
        if not os.path.isdir(os.path.join('results', program_dir)):
            continue

        input_file = os.path.join('results', program_dir, analysis_type, 'java_8', program_dir, 'CallGraphEdge.csv')  #fix naming
        output_file = os.path.join('results', program_dir, f'doop_{analysis_type}.csv')

        if not os.path.exists(input_file):
            print(f"Warning: Input file {input_file} does not exist.")
            continue

        print(f"[+] Processing {input_file}...")

        # Check if the output file already exists
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping...")
            continue


        # === Main Processing ===
        with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerow(['method', 'offset', 'target'])

            for line in fin:
                parsed = parse_callgraph_line(line)
                if parsed:
                    method, offset, target = parsed
                    caller_class = method.split('.')[0]
                    target_class = target.split('.')[0]
                    if stdlib_regex.match(caller_class) or stdlib_regex.match(target_class):
                        continue
                    writer.writerow([method, offset, target])

        print(f"[✓] WALA-style call graph written to {output_file}")


if __name__ == "__main__":
    main()