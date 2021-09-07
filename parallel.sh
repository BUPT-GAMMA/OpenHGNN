#!/bin/bash
# execute: ./parallel.h 2
# 2 means gpu
#aggrs=(gcnconv gatconv sageconv ginconv)
aggrs=(gcnconv)
datasets=(HGBn-ACM HGBn-DBLP HGBn-Freebase HGBn-IMDB)
models=(homo_GNN relation_HGNN mp_GNN)
#models=(homo_GNN relation_HGNN)
ran=(1 2)
for aggr in ${aggrs[*]}; do
  for i in ${ran[*]}; do
    yaml="yamlpath: ${aggr}_${i}.yaml"
    echo ${yaml}
    python space4hgnn/generate_yaml.py -a ${aggr} -s ${i}
    for dataset in ${datasets[*]}; do
    	for model in ${models[*]}; do
        {
        para="-a ${aggr} -s ${i} -m ${model} -d ${dataset} -g $1 -t node_classification"
        echo ${para}
        python run.py ${para}
        }&
      done
      wait
    done
  done
done

