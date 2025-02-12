import json
import os

TRACE_INDO_PATH = '../data/diff-sim/info'


def get_area(decoded_edges):
    '''return the unique methods, classes and pakcages in the selected diff ranage'''
    u_packages = set()
    u_classes = set()
    u_methods = set()

    for edge in decoded_edges:
        package = edge[0].split('.')[:-2]
        u_packages.add('.'.join(package))
        u_classes.add(edge[0].split('.')[-2])
        u_methods.add(edge[0].split('.')[-1])

        package = edge[1].split('.')[:-2]
        u_packages.add('.'.join(package))
        u_classes.add(edge[1].split('.')[-2])
        u_methods.add(edge[1].split('.')[-1])
        

    u_mathods_list = list(u_methods)
    u_classes_list = list(u_classes)
    u_packages_list = list(u_packages)
    
    json_data = json.dumps({'u_methods': u_mathods_list, 'u_classes': u_classes_list, 'u_package': u_packages_list})

    with open('mt3vc1_area.json', 'w') as f:
        f.write(json_data)

def compare(d1: dict, d2: dict):
    '''Compare the info in two dictionaries.'''
    
    result = {}
    
    for key in d1.keys() & d2.keys():  # Ensure both dictionaries have the key
        set1, set2 = set(d1[key]), set(d2[key])
        
        # Calculate similarity percentage
        intersection = set1 & set2
        similarity = (len(intersection) / max(len(set1), len(set2))) * 100 if max(len(set1), len(set2)) > 0 else 100
        
        # Find different values
        only_in_d1 = set1 - set2
        only_in_d2 = set2 - set1
        
        result[key] = {
            'similarity_percentage': similarity,
            'only_in_d1': list(only_in_d1),
            'only_in_d2': list(only_in_d2)
        }
    

    return result



def compare_info(file1, file2):
    """Decode an encoded file back to text format."""
   
    file1_path = os.path.join(TRACE_INDO_PATH, file1)
    file2_path = os.path.join(TRACE_INDO_PATH, file2)

    f1_data = {}
    f2_data = {}

    with open(file1_path, 'r') as f1:
        f1_data = json.load(f1)
    
    with open(file2_path, 'r') as f2:
        f2_data = json.load(f2)


    with open('../data/diff-sim/info/compare-mt3vc1-mt2vc4.json', 'w') as fout:
        fout.write(json.dumps(compare(f1_data, f2_data)))
    
    
    


def main():
    compare_info("mt3vc1_area.json","mt2vc4_area.json")


if __name__ == '__main__':
    main()