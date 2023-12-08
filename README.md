# Backward Influence Maximization

1. First, run  the SampleSubgraph.py, SampleSubgraph_graph, GenerateNodeFeature.py  in SampleFeature floder in sequence, which generate the config.G and config.trueP.

2. Secondly, run the Louvain.py in community detection floder, which generates config.community.

3. Then, run the link_prediction.py file to generate the link_prediction model which used to predict whether nodes without edges have links with each other, and run the temp.py file to generate config.pred. (we have provided the link_prediction model for each dataset)

4. Finally, run the comparison.py to show the results.