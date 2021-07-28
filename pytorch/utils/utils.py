# File utils
# ydawei@umich.edu

import os
import pickle
import datetime
import time
import argparse
import hashlib
import subprocess


class Dotable(dict):
    '''
        Access a dict through attributes. For example:
        a = {'name': 'John', 'email': 'john@example.com'}
        dotable = Dotable(a)
        dotable.name      # == 'John'
        dotable.email     # == 'john@example.com'
        dotable.to_dict() # == {'name': 'John', 'email': 'john@example.com'}
    '''
    __getattr__= dict.__getitem__

    def __init__(self, d=None):
        super().__init__()
        if d:
            self.update(d)

    def update(self, d, dic_list=[]):
        dic = self
        for i in dic_list:
            dic = dic[i]
        for k, v in d.items():
            if isinstance(dic.get(k), dict):
                dic[k].update(v)
            else:
                dic[k] = dic.parse(v)

    def to_dict(self):
        return self.parse_inv(self)

    def parse_inv(self, value):
        if isinstance(value, dict):
            return {k: self.parse_inv(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.parse_inv(v) for v in value]
        elif isinstance(value, (int, float, str, type(None))):
            return value
        else:
            raise TypeError('unknown type ' + str(type(value)))


    @classmethod
    def parse(cls, v):
        if isinstance(v, dict):
            return cls(v)
        elif isinstance(v, list):
            return [cls.parse(i) for i in v]
        elif isinstance(v, (int, float, str, type(None))):
            return v
        else:
            raise TypeError('unknown type ' + str(type(v)))


class Lazy:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.func(*self.args, **self.kwargs)


def exec_print(cmd):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print('[{}] {}' .format(st, ' '.join(cmd)))
    subprocess.run(cmd, check=True)


def list_all_folders(folder):
    return [os.path.join(folder, o) for o in os.listdir(folder) if os.path.isdir(os.path.join(folder, o))]


def list_files(folder, ext, recursive=False, cache_folder=None, full_path=True):
    folder = os.path.abspath(folder)

    if cache_folder is not None:
        assert isinstance(cache_folder, str)
        hash_obj = hashlib.sha256(folder.encode())
        FILENAME_CACHE = os.path.join(cache_folder, hash_obj.hexdigest())
        if recursive:
            FILENAME_CACHE += '-R'
        FILENAME_CACHE += ext + '.bin'

        try:
            with open(FILENAME_CACHE, 'rb') as f:
                return pickle.load(f)
        except IOError:
            pass

    filenames = []
    if recursive:
        for root, dummy, fnames in os.walk(folder):
            for fname in fnames:
                if fname.endswith(ext):
                    if full_path:
                        filenames.append(os.path.join(root, fname))
                    else:
                        filenames.append(fname)
    else:
        for fname in os.listdir(folder):
            if fname.endswith(ext):
                if full_path:
                    filenames.append(os.path.join(folder, fname))
                else:
                    filenames.append(fname)

    if cache_folder is not None:
        makedir_if_not_exist(cache_folder)
        with open(FILENAME_CACHE, 'wb') as f:
            pickle.dump(filenames, f, True)

    return filenames


def list_all_files_w_ext(folder, ext, recursive=False, cache=False, full_path=True):
    '''
        List all the files in a folder with specified extension.
        :param folder: which folder to list.
        :param ext: the extension that the file name ends with.
        :param recursive: whether recursively scan the subfolders.
        :param cache: whether save the file lists to a cache folder
    '''
    if type(ext) == str:
        return list_files(folder, ext, recursive, cache, full_path)

    return [fname for extension in ext for fname in list_files(folder, extension, recursive, cache, full_path)]


def makedir_if_not_exist(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except FileExistsError:
        pass
    return directory

def remove_if_exist(file):
    try:
        if os.path.exists(file):
            os.remove(file)
    except FileNotFoundError:
        pass
    return file


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False



def setattr_dot(dct, k, v):
    '''
        set_attr_dot(dct, 'k', 'v')           =>         dct.k == 'v'
        set_attr_dot(dct, 'k1.k2', 'v')       =>         dct.k1.k2 == 'v'
        set_attr_dot(dct, 'k1.k2.k3', 'v')    =>         dct.k1.k2.k3 == 'v'
    '''
    sp = k.split('.', 1)
    if len(sp) > 1:
        k1, k2 = sp
        dct[k1] = dct.get(k1) or {}
        setattr_dot(dct[k1], k2, v)
    else:
        dct[k] = v



class StoreDictKeyPair(argparse.Action):
    '''
        A tool to parse key-value pairs in argparse.
        For example, the command `python train.py -a attr1=value1,attr2=[value2,{attr3=value3}]`
        will yield a dict:
        {attr1: value1, attr2: [value,2 {attr3: value3}]}
    '''
    def __init__(self, *args, **kwargs):
        super(StoreDictKeyPair, self).__init__(*args, **kwargs)
        self.kv_dict = {}

    def parse_dict(self, val):
        dct = dict()
        remain = val
        while remain:
            # Find first '='
            pos = remain.index('=')
            k = remain[:pos]
            if remain[pos+1] == '{': # dict, find the matching'}'
                count = 0
                for m in range(pos + 2, len(remain)):
                    if remain[m] == '}' and count == 0: break
                    if remain[m] == '{': count += 1
                    if remain[m] == '}': count -= 1

                nxt = m
                setattr_dot(dct, k, self.parse_dict(remain[pos+2:nxt]))
                nxt += 1

            elif remain[pos+1] == '[': # list, find ']'
                nxt = remain.index(']')
                setattr_dot(dct, k, self.parse_list(remain[pos+2:nxt]))
                nxt += 1
            else:
                try:
                    nxt = remain.index(',')
                    v = remain[pos+1:nxt]
                except ValueError:
                    nxt = len(remain) - 1
                    v = remain[pos+1:]

                setattr_dot(dct, k, self.parse_single(v))
            remain = remain[nxt+1:]
        return dct


    def parse_list(self, val):
        lst = list()
        remain = val
        while remain:
            if remain[0] == '[':
                # Find the matching ']'
                count = 0
                for m in range(1, len(remain)):
                    if remain[m] == ']' and count == 0: break
                    elif remain[m] == '[': count += 1
                    elif remain[m] == ']': count -= 1
                assert count == 0, 'Brackets do not match!'
                nxt = m
                lst.append(self.parse_list(remain[1:nxt]))
            elif remain[0] == '{':
                # Find the matching '}'
                count = 0
                for m in range(1, len(remain)):
                    if remain[m] == '}' and count == 0: break
                    elif remain[m] == '{': count += 1
                    elif remain[m] == '}': count -= 1
                assert count == 0, 'Brackets do not match!'
                nxt = m
                lst.append(self.parse_dict(remain[1:nxt]))
            else:
                try:
                    nxt = remain.index(',')
                    v = remain[:nxt]
                except ValueError:
                    nxt = len(remain) - 1
                    v = remain

                lst.append(self.parse_single(v))

            remain = remain[nxt+1:]

        return lst

    def parse_single(self, v):
        if v.upper() == 'TRUE':
            return True
        elif v.upper() == 'FALSE':
            return False
        elif v.isdigit():
            return int(v)
        elif isfloat(v):
            return float(v)
        return v


    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            for value in values:
                self.kv_dict.update(self.parse_dict(value))
        elif type(values) == str:
            self.kv_dict.update(self.parse_dict(values))
        else:
            assert False
        setattr(namespace, self.dest, self.kv_dict)


def recursive_update_dict(original, to_update, strict=False, _prefix=''):
    '''
        Update the nested dict `original` with the values from another
        nested dict `to_update`.
        :param strict: True if enforce that the keys from `to_update` exist `original`
        :param _prefix: internal use, to handle dotted key values like 'attr1.attr2'.
    '''
    updated = {}
    for k, v in original.items():
        updated[k] = v

    if not to_update:
        return updated

    for k, v in to_update.items():
        if isinstance(v, dict) and k in original and original[k] is not None:
            updated[k] = recursive_update_dict(original[k], v, strict=strict, _prefix=_prefix + k + '.')
        elif not strict or k in original:
            updated[k] = v
        else:
            raise ValueError('Invalid key: {}'.format(_prefix + k))

    return updated



class CommandSerialization(object):
    '''
        Extract the arguments and generate the command string.
        Example:
        >>> s = CommandSerialization('python train.py')
        >>> s.serialize({attr1: 'value1', attr2: {attr3: value3}})
        'python train.py --attr1=value1 --attr2={attr3=value3}'

    '''
    def __init__(self, init_cmd):
        self.init_cmd = init_cmd


    @staticmethod
    def serialize_single(value, outermost):
        if isinstance(value, dict):
            string = CommandSerialization.serialize_dict(value, not outermost)
        elif isinstance(value, list):
            string = CommandSerialization.serialize_list(value)
        else:
            string = str(value)

        if outermost and type(value) in [dict, list]:
            string = '\'' + string + '\''

        return string

    @staticmethod
    def serialize_list(args):
        return '[' + ','.join([CommandSerialization.serialize_single(arg, False) for arg in args if arg is not None]) + ']'


    @staticmethod
    def serialize_dict(kwargs, bracket=False):
        string = ','.join(['{}={}'.format(k, CommandSerialization.serialize_single(v, False)) for k, v in kwargs.items() if v is not None])
        if bracket:
            string = '{' + string + '}'
        return string


    def serialize(self, kwargs):
        string = self.init_cmd + ' '
        for k, v in kwargs.items():
            if v is None:
                continue
            elif type(v) in [list, dict] and not v:
                continue
            elif v is False:
                continue
            elif v is True:
                string += '--{} '.format(k)
            else:
                string += '--{} {} '.format(k, self.serialize_single(v, True))

        return string[:-1]


def format_sheet_row(args, keys, delimeter='\t'):
    '''
    Example:
    >>> args = {attr1: value1, attr2: {attr3: value3, attr4: value4}}
    >>> keys = ['attr1', 'attr2.attr4']
    >>> format_sheet_row(args, keys, '///')
    attr1///attr2.attr4
    value1///value4
    '''

    values = []
    for key in keys:
        components = key.split('.')
        value = args
        for component in components:
            value = value[component]

        values.append(str(value))

    keys = delimeter.join(keys)
    values = delimeter.join(values)

    return keys, values


def format_option_text(args, options):
    indices, values = format_sheet_row(args, options, delimeter='/', separator='.')
    indices_list = indices.split('/')[:-1]
    values_list = values.split('/')[:-1]
    assert len(indices_list) == len(values_list)

    string = ''
    for name, value in zip(indices_list, values_list):
        name_splits = name.split('.')
        name = '.'.join(name_splits[::-1])
        string += name + '=' + value + '\n'

    return string
