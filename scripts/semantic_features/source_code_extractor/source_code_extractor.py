import os
import javalang
import re

# === Descriptor parsing ===

JVM_PRIMITIVES = {
    'B': 'byte',
    'C': 'char',
    'D': 'double',
    'F': 'float',
    'I': 'int',
    'J': 'long',
    'S': 'short',
    'Z': 'boolean',
    'V': 'void'
}

def parse_type(descriptor, i):
    array_dims = 0
    while descriptor[i] == '[':
        array_dims += 1
        i += 1

    if descriptor[i] in JVM_PRIMITIVES:
        base_type = JVM_PRIMITIVES[descriptor[i]]
        i += 1
    elif descriptor[i] == 'L':
        j = descriptor.find(';', i)
        base_type = descriptor[i + 1:j].replace('/', '.')
        i = j + 1
    else:
        raise ValueError(f"Unknown type descriptor at: {descriptor[i:]}")

    return base_type + '[]' * array_dims, i

def parse_descriptor(descriptor):
    args_part, return_part = re.match(r'\((.*)\)(.+)', descriptor).groups()
    param_types = []

    i = 0
    while i < len(args_part):
        parsed_type, i = parse_type(args_part, i)
        param_types.append(parsed_type)

    return_type, _ = parse_type(return_part, 0)
    return param_types, return_type

# def parse_descriptor(descriptor):
#     args_part, return_part = re.match(r'\((.*)\)(.+)', descriptor).groups()
#     param_types = []

#     i = 0
#     while i < len(args_part):
#         c = args_part[i]
#         if c in JVM_PRIMITIVES:
#             param_types.append(JVM_PRIMITIVES[c])
#             i += 1
#         elif c == 'L':
#             j = args_part.find(';', i)
#             param = args_part[i+1:j].replace('/', '.')
#             param_types.append(param)
#             i = j + 1
#         elif c == '[':
#             array_type, jump = '', 0
#             while args_part[i + jump] == '[':
#                 array_type += '[]'
#                 jump += 1
#             base_type, rest = parse_descriptor('(' + args_part[i + jump:] + ')')[0][0], jump
#             param_types.append(base_type + array_type)
#             i += jump + (1 if args_part[i + jump] in JVM_PRIMITIVES else args_part[i + jump:].find(';') + 1)
#         else:
#             raise ValueError(f"Unknown descriptor format at: {args_part[i:]}")
    
#     return_type = JVM_PRIMITIVES.get(return_part, return_part.replace('/', '.')) if return_part in JVM_PRIMITIVES else return_part.replace('/', '.')
#     return param_types, return_type


# === Source extractor ===

def find_nested_class(class_node, nested_class_name):
    for member in class_node.body:
        if isinstance(member, javalang.tree.ClassDeclaration) and member.name == nested_class_name:
            return member
    return None

def get_type_str(t):
    if isinstance(t, javalang.tree.BasicType) or isinstance(t, javalang.tree.ReferenceType):
        name = t.name
        dims = '[]' * len(t.dimensions)
        return name + dims
    return str(t)

# def find_method_source_code(project_dir, class_path, method_name, descriptor):
#     class_name = class_path.split('/')[-1]
#     class_path_dot = class_path.replace('/', '.')
#     param_types, return_type = parse_descriptor(descriptor)
#     nested_class_name = None
#     if '$' in class_name:
#         # Handle nested classes
#         parts = class_name.split('$')
#         class_name = parts[0]
#         nested_class_name = parts[1]

#     for root, _, files in os.walk(project_dir):
#         for file in files:
#             if file.endswith(".java") and class_name in file:
#                 filepath = os.path.join(root, file)
#                 try:
#                     with open(filepath, 'r') as f:
#                         code = f.read()

#                     tree = javalang.parse.parse(code)
#                     for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
#                         if class_node.name != class_name:
#                             continue
                        
#                         methods = class_node.methods
#                         if method_name == "<init>":
#                             methods = class_node.constructors

#                         for method in methods:
#                             method_param_types = [
#                                 p.type.name if isinstance(p.type, javalang.tree.ReferenceType) else p.type
#                                 for p in method.parameters
#                             ]
#                             # Compare short names only
#                             if method_param_types != [p.split('.')[-1] for p in param_types]:
#                                 continue
#                             if method_name != "<init>":
#                                 if method.return_type:
#                                     rt = method.return_type.name if isinstance(method.return_type, javalang.tree.ReferenceType) else method.return_type
#                                     if rt != return_type.split('.')[-1].replace(';',''):
#                                         continue
#                             print(method_name)
#                             return extract_method_source(filepath, method)
#                 except Exception as e:
#                     print(f"Failed parsing {filepath}: {e}")
#     return None

def find_method_source_code(project_dir, class_path, method_name, descriptor):
    # Handle inner classes
    path_parts = class_path.split('/')
    class_file_name = path_parts[-1].split('$')[0] + '.java'
    class_name_parts = path_parts[-1].split('$')
    outer_class_name = class_name_parts[0]
    inner_class_name = class_name_parts[1] if len(class_name_parts) > 1 else None

    param_types, return_type = parse_descriptor(descriptor)
    print(return_type)

    for root, _, files in os.walk(project_dir):
        for file in files:
            if file == class_file_name:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        code = f.read()

                    tree = javalang.parse.parse(code)
                    for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
                        if class_node.name != outer_class_name:
                            continue                      
                        if inner_class_name:
                            class_node = find_nested_class(class_node, inner_class_name)
                            if not class_node:
                                continue

                        methods = class_node.methods
                        if method_name == "<init>":
                            methods = class_node.constructors

                        for method in methods:
                            method_param_types = [
                                p.type.name if isinstance(p.type, javalang.tree.ReferenceType) else p.type
                                for p in method.parameters
                            ]
                            if method_param_types != [p.split('.')[-1] for p in param_types]:
                                continue
                            if method_name != "<init>":
                                if method.return_type:
                                    actual = get_type_str(method.return_type)
                                    expected = return_type.split('.')[-1].replace(';', '')
                                    if actual != expected:
                                        continue

                            print(f"[✓] Found match for method: {method_name}")
                            return extract_method_source(filepath, method)

                except Exception as e:
                    print(f"[!] Failed parsing {filepath}: {e}")
    return None

def extract_method_source(filepath, method_node):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_line = method_node.position.line - 1
    brace_count = 0
    method_lines = []
    started = False

    for i in range(start_line, len(lines)):
        line = lines[i]
        method_lines.append(line)
        brace_count += line.count('{') - line.count('}')
        if '{' in line:
            started = True
        if started and brace_count == 0:
            break

    # Include annotations (if any)
    annotation_lines = []
    for i in range(start_line - 1, -1, -1):
        line = lines[i].strip()
        if line.startswith('@'):
            annotation_lines.insert(0, lines[i])
        elif line == '':
            continue
        else:
            break

    return ''.join(annotation_lines + method_lines)

# === Example usage ===

if __name__ == "__main__":
    project_root = "/20TB/mohammad/xcorpus-total-recall/source_codes/axion-1.0-M2/src"
    input_signature = "org/axiondb/engine/BaseRow.getIdentifier:()I"

    class_path, method_sig = input_signature.split('.')
    method_name, descriptor = method_sig.split(':')

    print(f"Searching for method: {method_name} with descriptor: {descriptor} in class: {class_path}")

    source = find_method_source_code(project_root, class_path, method_name, descriptor)
    if source:
        print("[✓] Method found:")
        print(source)
    else:
        print("[!] Method not found.")
