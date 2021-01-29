import configparser
import os


class Config(object):
    def __init__(self, file_path, model, dataset):
        conf = configparser.ConfigParser()
        data_path = os.getcwd()
        try:
            conf.read(file_path)
        except:
            print("failed!")
        # training dataset path
        self.model = model
        self.dataset = dataset
        self.path = {'output_modelfold': './output/model/',
                     'input_fold': './dataset/'+self.dataset+'/',
                     'temp_fold': './output/temp/'+self.model+'/'}

        if model == "NSHE":
            self.dim_size = {}
            self.dim_size['emd'] = conf.getint("NSHE", "emd_dim")
            self.dim_size['context'] = conf.getint("NSHE", "context_dim")
            self.dim_size['project'] = conf.getint("NSHE", "project_dim")

            self.lr = conf.getfloat("NSHE", "learning_rate")
            self.max_epoch = conf.getint("NSHE", "max_epoch")
            self.num_e_neg = conf.getint("NSHE", "num_e_neg")
            self.num_ns_neg = conf.getint("NSHE", "num_ns_neg")
            self.norm_emd_flag = conf.get("NSHE", "norm_emd_flag")


        else:
            pass
