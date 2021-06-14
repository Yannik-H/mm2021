from mmdet.datasets import PIPELINES


@PIPELINES.register_module()
class MyTransform:

    def __call__(self, results):
        print(type(results))
        for key in results:
            print(key, type(results[key]))
        return results