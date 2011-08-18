'''
Created on Aug 18, 2011

@author: rca

Example workflow which is not working (yet).

Normally, it is executed as: run_workflow.py workflows/DESY
'''


from workflow.patterns import IF_NOT, ENG_GET, WHILE
from workflow.config import config_reader as cfg
from components.records import svm_classifier
from workflows.collections import create_records
from workflows.records import svm_classification
cfg.init('DESY.ini')

workflow = [
            init_params(),
            WHILE(get_input_sources, [
                    create_records.workflow,
                    WHILE(get_records, [
                          svm_classification.workflow,
                          bibclassify_classification.workflow,
                        ])
                  ]),

            ]
