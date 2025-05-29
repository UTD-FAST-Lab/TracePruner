from allpairspy import AllPairs
import pandas as pd

# Configuration space
wala_configuration_space = {
    'analysis': {'levels': ['0cfa', '1cfa', '2cfa', '1obj', '2obj', 'rta'], 'default': '0cfa'},
    'reflection': {'levels': [
        "FULL", "APPLICATION_GET_METHOD", "NO_FLOW_TO_CASTS",
        "NO_FLOW_TO_CASTS_APPLICATION_GET_METHOD", "NO_METHOD_INVOKE",
        "NO_FLOW_TO_CASTS_NO_METHOD_INVOKE", "ONE_FLOW_TO_CASTS_NO_METHOD_INVOKE",
        "ONE_FLOW_TO_CASTS_APPLICATION_GET_METHOD", "MULTI_FLOW_TO_CASTS_APPLICATION_GET_METHOD",
        "NO_STRING_CONSTANTS", "STRING_ONLY", "NONE"
    ], 'default': 'FULL'},
    'handleStaticInit': {'levels': ['true', 'false'], 'default': 'true'},
    'useConstantSpecificKeys': {'levels': ['true', 'false'], 'default': 'false'},
    'handleZeroLengthArray': {'levels': ['true', 'false'], 'default': 'true'},
    'useLexicalScopingForGlobals' : {'levels': ['true', 'false'], 'default': 'false'},
    'useStacksForLexicalScoping' : {'levels': ['true', 'false'], 'default': 'false'}
}

# Extract the keys and value lists
keys = list(wala_configuration_space.keys())
values = [wala_configuration_space[k] for k in keys]

# Generate 2-way combinations
pairwise_combinations = list(AllPairs(values))

# Convert to list of dictionaries (key-value format)
test_cases = [dict(zip(keys, comb)) for comb in pairwise_combinations]

# Display a few examples

df = pd.DataFrame(test_cases)
df.to_csv("wala_pairwise_testcases.csv", index=False)

# with open('2-way-covered.csv', 'w') as f:
#     for i, case in enumerate(test_cases):
        


