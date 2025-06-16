import pandas as pd

# Configuration space with default values
wala_configuration_space = {
    'analysis': {
        'levels': [
            '0cfa', 
            '1cfa', 
            '2cfa', 
            '1obj', 
            '2obj', 
            'rta', 
            "VANILLA_1cfa",
            "VANILLA_2cfa",
            "VANILLA_1obj",
            "VANILLA_2obj",
            "ZEROONE_CFA",
            "VANILLA_ZEROONECFA",
            "ZEROONE_CONTAINER_CFA",
            "VANILLA_ZEROONE_CONTAINER_CFA",
            "ZERO_CONTAINER_CFA"
            "RTA",
        ],
        'default': '0cfa'},

    'reflection': {
        'levels': [
            "FULL", "APPLICATION_GET_METHOD", "NO_FLOW_TO_CASTS",
            "NO_FLOW_TO_CASTS_APPLICATION_GET_METHOD", "NO_METHOD_INVOKE",
            "NO_FLOW_TO_CASTS_NO_METHOD_INVOKE", "ONE_FLOW_TO_CASTS_NO_METHOD_INVOKE",
            "ONE_FLOW_TO_CASTS_APPLICATION_GET_METHOD", "MULTI_FLOW_TO_CASTS_APPLICATION_GET_METHOD",
            "NO_STRING_CONSTANTS", "STRING_ONLY", "NONE"
        ], 
        'default': 'FULL'},

    'handleStaticInit': {'levels': ['true', 'false'], 'default': 'true'},
    'useConstantSpecificKeys': {'levels': ['true', 'false'], 'default': 'false'},
    'handleZeroLengthArray': {'levels': ['true', 'false'], 'default': 'true'},
    'useLexicalScopingForGlobals': {'levels': ['true', 'false'], 'default': 'false'},
    'useStacksForLexicalScoping': {'levels': ['true', 'false'], 'default': 'false'}
}

        

# wala_configuration_space = {
#     'analysis': {
#         'levels': [
#             "1-call-site-sensitive",
#             "1-call-site-sensitive+heap",
#             "1-object-1-type-sensitive+heap",
#             "1-object-sensitive",
#             "1-object-sensitive+heap",
#             "1-type-sensitive",
#             "1-type-sensitive+heap",
#             "2-call-site-sensitive+2-heap",
#             "2-call-site-sensitive+heap",
#             "2-object-sensitive+2-heap",
#             "2-object-sensitive+heap",
#             "2-type-object-sensitive+2-heap",
#             "2-type-object-sensitive+heap",
#             "2-type-sensitive+heap",
#             "3-object-sensitive+3-heap",
#             "3-type-sensitive+2-heap",
#             "3-type-sensitive+3-heap",
#             "adaptive-2-object-sensitive+heap",
#             "basic-only",
#             "context-insensitive",
#             "context-insensitive-plus",
#             "context-insensitive-plusplus",
#             "data-flow",
#             "dependency-analysis",
#             "fully-guided-context-sensitive",
#             "micro",
#             "partitioned-2-object-sensitive+heap",
#             "selective-2-object-sensitive+heap",
#             "sound-may-point-to",
#             "sticky-2-object-sensitive"
#         ],
#         'default': "context-insensitive"
#     },

#     # All other binary options
#     'cs-library': {'levels': ['false', 'true'], 'default': 'false'},
#     'no-merge-library-objects': {'levels': ['false', 'true'], 'default': 'false'},
#     'no-merges': {'levels': ['false', 'true'], 'default': 'false'},
#     'data-flow-goto-lib': {'levels': ['false', 'true'], 'default': 'false'},
#     'data-flow-only-lib': {'levels': ['false', 'true'], 'default': 'false'},
#     'heapdl-nostrings': {'levels': ['false', 'true'], 'default': 'false'},
#     'only-precise-native-strings': {'levels': ['false', 'true'], 'default': 'false'},
#     'distinguish-reflection-only-string-constants': {'levels': ['false', 'true'], 'default': 'false'},
#     'distinguish-string-buffers-per-package': {'levels': ['false', 'true'], 'default': 'false'},
#     'reflection': {'levels': ['false', 'true'], 'default': 'false'},
#     'reflection-classic': {'levels': ['false', 'true'], 'default': 'false'},
#     'reflection-high-soundness-mode': {'levels': ['false', 'true'], 'default': 'false'},
#     'Xgenerics-pre': {'levels': ['false', 'true'], 'default': 'false'},
#     'Xreflection-coloring': {'levels': ['false', 'true'], 'default': 'false'},
#     'Xreflection-context-sensitivity': {'levels': ['false', 'true'], 'default': 'false'}
# }


# Step 1: Create base (default) config
default_config = {
    key: spec['default'] for key, spec in wala_configuration_space.items()
}

# Step 2: Generate 1-change configs
one_change_configs = []

for key, spec in wala_configuration_space.items():
    default_value = spec['default']
    for val in spec['levels']:
        if val != default_value:
            config = default_config.copy()
            config[key] = val
            one_change_configs.append(config)

# Step 3: Export to CSV
df = pd.DataFrame(one_change_configs)
df.to_csv("doop_1change_configs.csv", index=False)

print(f"Generated {len(one_change_configs)} 1-change configurations.")
