def singleton_generator():
    instances = {}

    def singleton_decorator(class_):
        def wrapper(*args, **kwargs):
            if class_.__name__ not in instances:
                instances[class_.__name__] = class_(*args, **kwargs)
                return instances[class_.__name__]
            else:
                return instances[class_.__name__]

        return wrapper

    singleton_decorator.all = instances
    return singleton_decorator


singleton = singleton_generator()
