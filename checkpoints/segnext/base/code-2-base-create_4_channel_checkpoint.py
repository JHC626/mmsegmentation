from osgeo import gdal
import torch
from mmengine import Config
from mmseg.apis import init_model
from mmengine.runner.checkpoint import save_checkpoint, CheckpointLoader


"""
ATL_笔记：
    新权重的012通道是原始预训练的BGR, 3456789通道是kaiming_normal_初始化的
    也就只是变了第一层patch_embed.proj.weight的权重，后面的都没变呀。

    1. 从beiv2_large_patch16_224_pt1k_ft21k.pth中提取patch_embed.proj.weight
    2. 生成一个新的patch_embed.proj.weight，形状为(1024, 10, 16, 16)
    3. 将新的patch_embed.proj.weight覆盖到checkpoint中
    4. 保存新的checkpoint

"""

if __name__ == '__main__':

    weight_name = 'patch_embed1.proj.0.weight'

    checkpoint_path = './segnext-base-3chan.pth'
    checkpoint_3channel = CheckpointLoader.load_checkpoint(checkpoint_path, map_location='cpu')
    save_path_10channel = './segnext-base-4chan.pth'
    if 'module' in checkpoint_3channel:
        checkpoint_3channel = checkpoint_3channel['module']   
    elif 'model' in checkpoint_3channel:
        checkpoint_3channel = checkpoint_3channel['model'] 
    elif 'state_dict' in checkpoint_3channel:
        checkpoint_3channel = checkpoint_3channel['state_dict']   

    # 保存一个中间权重，防止覆盖啥的
    cp_back_path = checkpoint_path.replace('.pth', '_back.pth')
    torch.save(checkpoint_3channel, cp_back_path)
    checkpoint_10channel = CheckpointLoader.load_checkpoint(checkpoint_path, map_location='cpu')

    if 'module' in checkpoint_10channel:
        checkpoint_10channel = checkpoint_10channel['module']   
    elif 'model' in checkpoint_10channel:
        checkpoint_10channel = checkpoint_10channel['model']   
    elif 'state_dict' in checkpoint_10channel:
        checkpoint_10channel = checkpoint_10channel['state_dict']   

    print('开始处理')
    with torch.no_grad():
        print(f"原始 checkpoint.{weight_name} 形状{checkpoint_3channel[weight_name].shape}")
        new_weight = torch.zeros((32, 4, 3, 3))
        checkpoint_10channel[weight_name] = new_weight
        print('[ATL-LOG] 3channel value 0 chan\n')
        print(checkpoint_3channel[weight_name][:, 0])


        # 新的权重初始化，用kaiming_normal_
        torch.nn.init.kaiming_normal_(checkpoint_10channel[weight_name], mode='fan_out', nonlinearity='relu')
        print(f"修改通道后 checkpoint.{weight_name} 形状{checkpoint_10channel[weight_name].shape}")
        print('[ATL-LOG] 10channel value 0 chan\n')
        print(checkpoint_10channel[weight_name][:, 0])

        # 3channel的权重是RGB，图像是BGR，换一下。  #RGB   B G R Nir
        checkpoint_10channel[weight_name][:, 0] = checkpoint_3channel[weight_name][:, 2]
        checkpoint_10channel[weight_name][:, 1] = checkpoint_3channel[weight_name][:, 1]
        checkpoint_10channel[weight_name][:, 2] = checkpoint_3channel[weight_name][:, 0]
        checkpoint_10channel[weight_name][:, 3] = checkpoint_3channel[weight_name][:, 2]


        print(f"覆盖3通道后 checkpoint.{weight_name} 形状{checkpoint_10channel[weight_name].shape}")
        
        
        if checkpoint_10channel[weight_name][:, 0].all() == checkpoint_3channel[weight_name][:, 2].all():
            print('通道1覆盖成功')
        if checkpoint_10channel[weight_name][:, 1].all() == checkpoint_3channel[weight_name][:, 1].all():
            print('通道2覆盖成功')
        if checkpoint_10channel[weight_name][:, 2].all() == checkpoint_3channel[weight_name][:, 0].all():
            print('通道3覆盖成功')

        torch.save(checkpoint_10channel, save_path_10channel)
        print('处理完成')