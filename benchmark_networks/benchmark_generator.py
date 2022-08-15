import bm
import networkx
import numpy as np
model_list = ["StdMerge","StdGrow","StdMixed"]
for model_type in model_list:
    std_merge_model = bm.get_model(model_type, p_in=.5, p_out=0.05, n=20,q=4)
    std_merge_comms = []
    std_merge_graphs = []
    output_file_prefix = model_type
    for t in range(100):
        model = std_merge_model
        qi = std_merge_model.comms(t)
	print(qi)
	std_merge_comms.append(qi)
        graph_t = std_merge_model.graph(t)
        bm.write_temporal_commlist(model, output_file_prefix+'.tcomms', qi, t)
        bm.write_temporal_edgelist(model, graph_t, output_file_prefix+'.tgraph', t)
