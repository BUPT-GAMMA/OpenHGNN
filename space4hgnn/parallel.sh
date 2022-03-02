#!/bin/bash
# execute: ./parallel.sh 2
# 2 means gpu
if [ ! -n "$1" ]; then
  echo "gpu is empty"
  exit 0
fi
if [ ! -n "$2" ]; then
  echo "repeat is empty"
  exit 0
fi
if [ ! -n "$3" ]; then
  echo "key is empty"
  exit 0
fi
if [ ! -n "$4" ]; then
  echo "value is empty"
  exit 0
fi
if [ ! -n "$5" ]; then
  echo "task type is empty"
  exit 0
fi
if [ ! -n "$6" ]; then
  echo "config file is empty"
  exit 0
fi
if [ ! -n "$7" ]; then
  echo "predict file is empty"
  exit 0
fi
aggrs=(gcnconv gatconv sageconv ginconv)
#aggrs=(gcnconv)
if [ "$5" == "node_classification" ]
then
  datasets=(HGBn-ACM HGBn-DBLP HGBn-IMDB HGBn-Freebase HNE-PubMed)
  #datasets=(HGBn-ACM)
elif [ "$5" == "link_prediction" ]
then
  #datasets=(HGBl-amazon HGBl-LastFM HGBl-PubMed)
  datasets=(HGBl-ACM HGBl-DBLP HGBl-IMDB)
elif [ "$5" == "recommendation" ]
then
  datasets=(DoubanMovie)
else
  echo "The task name is wrong!"
fi

subgraphs=(homo metapath relation)
ran=(1 2)
for aggr in ${aggrs[*]}; do
  for i in ${ran[*]}; do
    echo "===================================================================================="
    file="./space4hgnn/config/$6/$3/${aggr}_${i}.yaml"
    if [ ! -f ${file} ]
    then
      echo "yaml not exists"
      python ./space4hgnn/generate_yaml.py -a ${aggr} -s ${i} -k $3 -c $6
    fi
    for dataset in ${datasets[*]}; do
    	for subgraph in ${subgraphs[*]}; do
        {
        if [ ${subgraph} == "homo" ]
        then
          para="-a ${aggr} -s ${i} -m homo_GNN -d ${dataset} -g $1 -t $5 -k $3 -v $4 -r $2 -c $6 -p $7"
        else
          para="-a ${aggr} -s ${i} -m general_HGNN -u ${subgraph} -d ${dataset} -g $1 -t $5 -k $3 -v $4 -r $2 -c $6 -p $7"
        fi
        echo "***************************************************************************************"
        echo ${para}
        python ./space4hgnn.py ${para}
        }&
      done
      wait
    done
  done
done

