import os, glob
import pandas as pd
import argparse

def all_path(dirname):
    list = []
    postfix = set(['csv'])
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            #if True:
            if apath.split('.')[-1] in postfix:
                try:
                    # with open(filelistlog, 'a+') as fo:
                    #     fo.writelines(apath)
                    #     fo.write('\n')
                    list.append(apath)
                except:
                    pass
    return list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictfile', '-p', default='predict', type=str, help='value')
    #parser.add_argument('--files', '-f', default='predict', type=str, help='value')

    args = parser.parse_args()
    all_files = all_path(args.predictfile)
    print(all_files)

    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True, sort=False)
    df_merged.to_csv("{}/result.csv".format(args.predictfile))