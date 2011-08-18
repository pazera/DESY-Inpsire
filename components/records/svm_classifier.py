
from workflow.engine import get_logger
import os


log = get_logger('workflow.svm_classifier')

# requires lucene to be installed
try:
    from lucene import DefaultSimilarity, TermPositionVector, \
                       Term, IndexSearcher,Field
except:
    log.warning('lucene is not installed, please do not use the lucene specific tasks')

from libs import svm_classifier


def init(svm_model,
         svm_train,
         svm_predict,
         indexer='#indexer'):

    def _x(obj, eng):
        eng.setVar('#svm_train', svm_train)
        eng.setVar('#svm_predict', svm_predict)
        eng.setVar('#svm_model', svm_model)
        if indexer:
            eng.setVar('#indexer', indexer)
    return _x

def _build_feature_map_from_index(
                      eng_input='#indexer',
                      eng_output = '#fmap',
                      fieldname='text',
                      max_tokens_perdoc=10000,
                      max_f_perdoc=100,
                      min_f_perindex=-1,
                      max_f_perindex=-1,
                      min_token_len=-1,

                      ):
    """Generates the map of tokens->ids; it is used to translate
    words into the position inside the vector.

    @keyword eng_input: (str) this task expects the lucene indexer
            object to find under this key
    @keyword eng_output: (str) results are put inside the engine
            under this key
    @keyword fieldname: (str) name of the index to pull words from
    @keyword max_tokens_perdoc: (int) can be used to limit the number
            of features that we accept. Useful for huge documents.
    @keyword min_f_perindex: (int) include only words that are present
            in at least x documents
    @keyword max_f_perindex: (int) include only words that are present
            in at most x documents
    """

    def x(obj, eng):

        tmap = eng.getVar(eng_output, {}) # holds token2id mappings
        indexer = eng.getVar(eng_input)
        reader = indexer.getReader()
        searcher = IndexSearcher(reader)
        docs = reader.numDocs()

        for i in xrange(docs):
            tfv = reader.getTermFreqVector(i, fieldname)
            if tfv:
                terms = tfv.getTerms()
                frequencies = tfv.getTermFrequencies()
                for (t,f,x) in zip(terms,frequencies,xrange(max_tokens_perdoc)):
                    if len(t) >= min_token_len and f <= max_f_perdoc:
                        df= searcher.docFreq(Term(fieldname, t)) # number of docs with the given term
                        if df > min_f_perindex and df <= max_f_perindex:
                            tmap.setdefault(t, len(tmap)+1)

    x.__name__ = '_build_feature_map_from_index'
    return x

build_feature_map_from_index = _build_feature_map_from_index()



def build_feature_map_from_dir(obj, eng):
    dirpath = eng.getVar('#dirpath')
    if not dirpath:
        raise Exception('Missing parameter: dirpath')

    eng.setVar(svm_classifier.build_feature_map_from_dir(dirpath))

def train_model(obj, eng):
    svm_train = eng.getVar('#svm_train')
    train_dir = eng.getVar('#train_dir')
    train_cat_index = train_dir + '.txt'
    target_file = eng.getVar('#svm_model')
    svm_classifier.train_model(svm_train, train_dir, train_cat_index, target_file)


def classify_record(obj, eng):
    # get data from the record
    # TODO
    text = 'bla bla bla'
    fmap = eng.getVar('#svm_fmap')
    svm_predict = eng.getVar('#svm_predict')
    model = eng.getVar('#svm_model')
    output_dir = eng.getVar('#smv_output_dir')

    vector = svm_classifier.vectorize_text(fmap, text)
    result = svm_classifier.classify_vector(svm_predict, model, vector, output_dir)

    #TODO: save result into the record
    obj['$result'] = result
