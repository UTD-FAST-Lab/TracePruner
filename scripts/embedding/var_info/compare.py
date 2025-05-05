import os
import pandas as pd


for program in os.listdir("/20TB/mohammad/data/variables"):

    df1 = pd.read_csv(f"/20TB/mohammad/data/var_repr/{program}_full_info.csv")

    # df1 = pd.read_csv("/20TB/mohammad/data/var_repr/urlead5353366_Zeldon_BigData_Class_tgz-pJ8-LocalTestsJ8_full_info.csv")
    # df1 = pd.read_csv("/20TB/mohammad/data/var_repr/urlead5353366_Zeldon_BigData_Class_tgz-pJ8-LocalTestsJ8_full_info.csv")
    # df2 = pd.read_csv("/20TB/mohammad/data/cg_embeddings/sum/tfidf/urlead5353366_Zeldon_BigData_Class_tgz-pJ8-LocalTestsJ8.csv")
    df2 = pd.read_csv(f"/20TB/mohammad/data/cg_embeddings/sum/tfidf/{program}.csv")

    # according to src_node, offset, target_node in df1 and method, offset, target in df2 give me the unique rows in each dataframes
    df1_unique = df1[~df1[['src_node', 'offset', 'target_node']].apply(tuple, 1).isin(df2[['method', 'offset', 'target']].apply(tuple, 1))]
    df2_unique = df2[~df2[['method', 'offset', 'target']].apply(tuple, 1).isin(df1[['src_node', 'offset', 'target_node']].apply(tuple, 1))]

    df1_unique = df1_unique[['src_node', 'offset', 'target_node']]
    df2_unique = df2_unique[['method', 'offset', 'target']]

    # print("df1 unique rows: ", df1_unique)
    # print("df2 unique rows: ", df2_unique)

    print(program)
    for i in df2_unique.iterrows():
        print(i[1]['method'], i[1]['offset'], i[1]['target'])


    # print()
    # for i in df1_unique.iterrows():
    #     print(i[1]['src_node'], i[1]['offset'], i[1]['target_node'])

    print(
        "-------------------------------------------------------------------------------------"
    )