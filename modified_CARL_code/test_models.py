import timm
from pprint import pprint

def print_options():
    model_names = timm.list_models(pretrained=True)
    pprint(model_names)

def test_load_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    # print(model)
    for i,c in enumerate(model.children()):
        # print(i)
        print(c)
        # print('---')


if __name__ == '__main__':
    test_load_model('vit_small_patch16_224.dino')