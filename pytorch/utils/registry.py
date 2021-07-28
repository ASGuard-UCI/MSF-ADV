
'''
A factory for indexing class using strings.
Example:
>>> @factory.register('ClassType', 'ClassName')
    class Name(object): pass

    instance = factory.create('ClassType', 'ClassName')()
'''
__name_map = dict()

def create(typename, classname):
    return __name_map[typename][classname]

def register(typename, classname):
    '''Return a function which registers the subclass with given @name.'''
    print('registering {}:{}'.format(typename, classname))
    if typename not in __name_map.keys():
        __name_map[typename] = dict()

    def register_name_fn(subcls):
        __name_map[typename][classname] = subcls
        return subcls

    if classname.isalnum():
        return register_name_fn
    else:
        raise ValueError('Invalid name {}. Must follow isalnum()=True. Not registered.'.format(classname))
