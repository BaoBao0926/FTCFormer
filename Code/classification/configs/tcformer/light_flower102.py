cfg = dict(
    model='tcformer_light',
    drop_path=0.1,
    clip_grad=None,
    # output_dir='work_dirs/tcformer_light_flower102_test',
)

"""
python main.py --config configs/tcformer/tcformer_light_flower102.py

1.work_dirs/tcformer_light_flower102_1: 这个是只训练了300个epoch,但是看起来还没有收敛
Test: Total time: 0:00:39 (0.5155 s / it)
* Acc@1 53.017 Acc@5 78.257 loss 2.077
Accuracy of the network on the 6149 test images: 53.0%, Max accuracy: 53.72%, Training time 4:28:14

2.work_dirs/tcformer_light_flower102_2: 1卡在跑, 1000个epoch
{"train_lr": 1.0686187278161592e-05, "train_loss": 2.4584259390830994, "test_loss": 1.29529910989396, "test_acc1": 72.90616364354214, "test_acc5": 88.5835095584092, "epoch": 937, "n_parameters": 13770985}
Max accuracy: 72.91%
"""