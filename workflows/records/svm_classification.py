'''
Created on Aug 17, 2011

@author: rca

This workflow works on the record, it should get the fulltext of the
marc record, transform it into vectors, and classify using SVM. Result
is stored inside the obj instance
'''

from workflow.patterns import IF_NOT, ENG_GET
from workflow.config import config_reader as cfg
from components.records import svm_classifier

cfg.init('DESY')

workflow = [
            IF_NOT(ENG_GET('#svm_fmap'), [
                       svm_classifier.init(cfg.svm_model, cfg.svm_train, cfg.svm_predict,
                                           train_dir=cfg.train_dir, test_dir=cfg.test_dir,
                                           output_dir=cfg.basedir),
                       svm_classifier.build_feature_map_from_dir
                   ]),
            svm_classifier.classify_record,
            ]
