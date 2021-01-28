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
            self.relation_category = conf.get("RHINE", "relation_category")

            self.data_set = conf.get("Model_Setup", "data_set")
            self.combination = conf.get("RHINE", "combination")
            self.link_type = conf.get("RHINE", "link_type")
            self.mode = conf.get("Model_Setup", "mode")
            self.IRs_nbatches = conf.getint("RHINE", "IRs_nbatches")
            self.ARs_nbatches = conf.getint("RHINE", "ARs_nbatches")

            self.margin = conf.getint("RHINE", "margin")
            self.ent_neg_rate = conf.getint("Model_Setup", "ent_neg_rate")
            self.rel_neg_rate = conf.getint("Model_Setup", "rel_neg_rate")
            self.evaluation_flag = conf.get("Model_Setup", "evaluation_flag")
            self.log_on = conf.getint("Model_Setup", "log_on")
            self.exportName = conf.get("Model_Setup", "exportName")
            if self.exportName == 'None':
                self.importName = None
            self.importName = conf.get("Model_Setup", "importName")
            if self.importName == 'None':
                self.importName = None
            self.export_steps = conf.getint("Model_Setup", "export_steps")
            self.opt_method = conf.get("Model_Setup", "opt_method")
            self.optimizer = conf.get("Model_Setup", "optimizer")
            if self.optimizer == 'None':
                self.optimizer = None
            self.weight_decay = conf.get("Model_Setup", "weight_decay")

        else:
            pass
