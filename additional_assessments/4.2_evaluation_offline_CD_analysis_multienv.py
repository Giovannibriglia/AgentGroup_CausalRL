import global_variables
from scripts.utils.merge_causal_graphs import MergeCausalGraphs
from scripts.utils.test_causal_table import TestCausalTable
import json

DIR_RESULTS = 'OfflineCD_MultiEnv_4x4'

DIR_RESULTS = f'{global_variables.GLOBAL_PATH_REPO}/Results/{DIR_RESULTS}'

merging = MergeCausalGraphs(dir_results=DIR_RESULTS)
merging.start_merging()
out_causal_graph = merging.get_merged_causal_graph()
with open(f'{global_variables.PATH_CAUSAL_GRAPH_OFFLINE}', 'w') as json_file:
    json.dump(out_causal_graph, json_file)

merging.start_cd()
out_causal_table = merging.get_causal_table()

test = TestCausalTable(out_causal_table, global_variables.get_possible_actions)
test.do_check()

out_causal_table.to_excel(f'{global_variables.GLOBAL_PATH_REPO}/scripts/utils/ground_truth_causal_table.xlsx')
out_causal_table.to_pickle(f'{global_variables.PATH_CAUSAL_TABLE_OFFLINE}')
