from torch.utils.tensorboard import SummaryWriter
import os

def create_tb_logger(root_dir, tb_log_dir='tensorboard'):
    return SummaryWriter(log_dir=os.path.join(root_dir, tb_log_dir))

if __name__ == "__main__":
    cfg = {}
    cfg["ckpt"] = None
    cfg["num_epochs"] = 40
    cfg["ckpt_save_interval"] = 5
    cfg["batch_size"] = 100
    cfg["pdrop"] = 0.5
    cfg["grad_norm_clip"] = 5.0
    cfg["num_networks"] = 30

    data_dir = os.path.join("./..", "data", "boston")

    output_dir = os.path.join("./..", "output", "uci_regression")
    ckpt_dir = os.path.join(output_dir, 'ckpts')

    print("Prepare model")
    from model.fc import FC
    from utils.dataset import *

    bos_train = BostonDataset(os.path.join(data_dir, "boston_train.csv"))
    train_loader_bos = torch.utils.data.DataLoader(bos_train, batch_size=cfg["batch_size"], num_workers=2)

    model = FC(bos_train.input_dim, cfg["pdrop"])
    model.cuda()

    cur_ckpt = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e{}".format(40)))
    print("loading checkpoint ckpt_e{}".format(40))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])
    model.eval()

    writer = SummaryWriter('./../runs/experiment')
    x, y = next(iter(bos_train))
    # writer.add_graph(model, x.cuda())
    # writer.close()

    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

    writer.close()