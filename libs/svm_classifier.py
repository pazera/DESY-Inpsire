'''
Created on Aug 18, 2011

@author: rca
'''
import os
import tempfile

def main(svm_train, svm_predict,
         train_dir, train_cat_index,
         test_dir, test_cat_index,
         output_dir):

    model = output_dir + '/model'
    if not os.path.exists(model):
        fmap = train_model(svm_train, train_dir, train_cat_index, model)
    else:
        fmap = build_feature_map_from_dir(train_dir)

    correct = wrong = 0
    correct_results = load_categories(test_cat_index)
    for filename, correct_result in correct_results.items():
        f = os.path.join(test_dir, filename)
        vector = vectorize_file(fmap, f)
        result = classify_vector(svm_predict, model, vector, output_dir)
        if result == correct_result:
            correct += 1
        else:
            wrong += 1


    print 'correct/wrong = %s/%s (%s)' % (correct, wrong, correct/len(correct_results))


def classify_vector(svm_predict, svm_model, vector, output_dir):
    filepath = output_dir + tempfile.gettempprefix()
    fi = open(filepath, 'w')
    _write_line(fi, '0', vector)
    fi.close()
    result_file = filepath + '.output'
    run_classification(svm_predict, svm_model, filepath, result_file)
    res = load_svm_results(result_file)
    os.remove(filepath)
    os.remove(result_file)
    return res


def load_svm_results(filepath):
    fi = open(filepath, 'r')
    out = []
    for c in fi:
        out.append(c.strip())
    fi.close()
    return out


def run_classification(svm_predict_binary, svm_model, test_file, result_file):
    os.system('%s %s %s %s' % (svm_predict_binary, test_file, svm_model, result_file))

def _format_vector(vector):
    out = []
    for vi, vv in zip(range(1, len(vector)+1), vector):
        if vv:
            out.append('%d:%f' % (vi, vv))
    return ' '.join(out)

def _write_line(fo, category, vector):
    fo.write(category)
    fo.write(' ')
    fo.write(_format_vector(vector))
    fo.write('\n')

def train_model(svm_train, train_dir, train_cat_index, target_file):
    fmap = build_feature_map_from_dir(train_dir)
    correct_results = load_categories(train_cat_index)
    train_data = target_file + '.data'
    fo = open(train_data, 'w')

    for filename, category in correct_results.items():
        vector = vectorize_file(fmap, os.path.join(train_dir, filename))
        _write_line(fo, category, vector)
    fo.close()

    run_training(svm_train, train_data, target_file)

    return fmap

def run_training(svm_train, train_data, target_file):
    os.system('%s %s %s' % (svm_train, train_data, target_file))

def _streamer(filepath):
    fi = open(filepath, 'r')
    text = fi.read().encode('utf8')
    fi.close()
    return text

def _tokenizer(text):
    return text.split()

def _normalizer(token):
    return token

def _filter(token):
    if token and len(token) > 2:
        return True

def build_feature_map_from_dir(dirpath,
                                streamer=_streamer,
                                tokenizer=_tokenizer,
                                normalizer=_normalizer,
                                filterizer=_filter):
    tset = set()
    tmap = {}
    files = os.listdir(dirpath)
    for f in files:
        for token in filter(filterizer, map(normalizer, tokenizer(streamer(os.path.join(dirpath, f))))):
            tset.add(token)

    for t in tset:
        tmap[t] = len(tmap)
    return tmap


def vectorize_file(fmap, filepath, streamer=_streamer, tokenizer=_tokenizer, normalizer=_normalizer, filterizer=_filter):
    return vectorize_text(fmap, streamer(filepath), tokenizer=tokenizer, normalizer=normalizer, filterizer=filterizer)

def vectorize_text(fmap, text, tokenizer=_tokenizer, normalizer=_normalizer, filterizer=_filter):
    vector = [0] * len(fmap)
    for token in filter(filterizer, map(normalizer, tokenizer(text))):
        if token in fmap:
            vector[fmap[token]] = 1
    return vector

def load_categories(filepath):
    fi = open(filepath, 'r')
    fcat = {}
    for line in fi:
        line = line.strip()
        if line:
            filename, cat = line.split('\t')
            fcat[filename] = cat
    fi.close()
    return fcat


if __name__ == '__main__':
    base = '/x/dev/liblinear/'
    svm_train = base + 'train'
    svm_predict = base + 'predict'
    basedir = '/x/dev/workspace/DESY-Inpsire/tests/data/svm/'
    test_dir = basedir + 'test'
    test_cat = test_dir + '.txt'
    train_dir = basedir + 'train'
    train_cat = train_dir + '.txt'
    try:
        os.remove(basedir + 'model')
    except:
        pass
    main(svm_train, svm_predict, train_dir, train_cat, test_dir, test_cat, basedir)
