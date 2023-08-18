import timm
from pprint import pprint
import torch.nn as nn


def print_options():
    model_names = timm.list_models(pretrained=True)
    pprint(model_names)



def test_load_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    # model.eval()
    print(model)
    # for i,c in enumerate(model.children()):
    #     print(i)
    #     print(c)
    #     print('---')



# test split model into frozen and trainable parts
# layer specifies which layer to freeze up to
def test_split_model(model_name, layer):
    # TODO
    model = timm.create_model(model_name, pretrained=True)
    # parts = list(model.children())

    # backbone = parts[:4]
    # post_layers = parts[5:]
    # t_blocks = parts[4]
    # t_blocks = list(t_blocks.children())
    # nb = len(t_blocks)

    # t_blocks_freeze = t_blocks[:nb-fine_blocks]
    # t_blocks_fine = t_blocks[nb-fine_blocks:]

    # backbone += t_blocks_freeze
    # res_finetune = t_blocks_fine + post_layers

    parts = list(model.children())
    t_blocks = list(parts[4].children())
    backbone = parts[:4] + t_blocks[:layer]
    finetune = t_blocks[layer:] + parts[5:]
    backbone = nn.Sequential(*backbone)
    res_finetune = nn.Sequential(*finetune)

    print(backbone)
    print('---')
    print(res_finetune)

    # print(len(t_blocks[:layer]))
    # print(len(t_blocks[layer:]))

    return



def other_tests(model_name):
    model = timm.create_model(model_name, pretrained=True)
    # print('num_prefix_tokens')
    # print(model.num_prefix_tokens)
    print(model)
    for i,c in enumerate(model.modules()):
        print(i)
        print(c)
        print('---')


# freeze the parameters of a timm-loaded dino model up to a specified layer
def freeze_dino(model, layer, silent=False):
    fc = 0
    fb = 0
    for i,c in enumerate(model.children()):
        if i < 4:
            # input layers
            for p in c.parameters():
                p.requires_grad = False
                fc += 1
        else:
            # transformer blocks
            for b_idx, b in enumerate(c.children()):
                if b_idx == layer: break
                for p in b.parameters():
                    p.requires_grad = False
                    fc += 1
                fb += 1
            break
    if not silent:
        print('frozen block count: ' + str(fb))
        print('frozen param count: ' + str(fc))



# https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch/blob/master/README.md
def freeze_test(model_name, layer):
    model = timm.create_model(model_name, pretrained=True)
    freeze_dino(model, layer)



# count the number of blocks in a ViT model:
def count_blocks(model_name):
    model = timm.create_model(model_name, pretrained=True)
    print('===')
    print(model_name)
    print('blocks: %i'%len(model.blocks))


if __name__ == '__main__':
    # print_options()
    # test_load_model('vit_small_patch16_224.dino')
    # test_split_model('vit_small_patch16_224.dino', 3)
    # test_split_model('vit_small_patch16_224.dino', 9)
    # other_tests('vit_small_patch16_224.dino')
    # freeze_test('vit_small_patch16_224.dino', 11)

    MODEL_NAMES = ['vit_small_patch16_224.dino', 'vit_small_patch8_224.dino', 'vit_base_patch16_224.dino', 
        'vit_base_patch8_224.dino', 'vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
        'vit_large_patch14_dinov2.lvd142m', 'vit_giant_patch14_dinov2.lvd142m']
    for MN in MODEL_NAMES:
        count_blocks(MN)
